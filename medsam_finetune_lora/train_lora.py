"""
train_lora.py
-------------
MedSAM2 + LoRA finetuning 학습 파이프라인.

데이터 구조 (root_dir 기준):
    result_resolution/
        output1/frames/frame_00001.png ...
        output10/frames/frame_00001.png ...
        output20/frames/frame_00001.png ...
        output30/frames/frame_00001.png ...
    propagated_masks/
        output1/frame_00001.png ...
        output10/frame_00001.png ...
        output20/frame_00001.png ...
        output30/frame_00001.png ...

output 번호로 이미지-마스크가 1:1 매칭됨.

실행:
    python train_lora.py \\
        --root_dir /path/to/data \\
        --medsam2_dir /path/to/MedSAM2 \\
        --checkpoint MedSAM2/checkpoints/MedSAM2_latest.pt \\
        --output_dir ./checkpoints \\
        --epochs 50 --batch_size 4 --rank 4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.ndimage import distance_transform_edt


# ──────────────────────────────────────────────────────────────────────────────
# 0. 경로 설정 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def setup_path(medsam2_dir: str):
    """MedSAM2 소스 경로를 sys.path 에 추가."""
    p = str(Path(medsam2_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터셋
# ──────────────────────────────────────────────────────────────────────────────

class SMASDataset(Dataset):
    """
    초음파 B-mode 이미지 + SMAS 마스크 데이터셋.

    이미지: result_resolution/output{N}/frames/frame_XXXXX.png  (grayscale)
    마스크: propagated_masks/output{N}/frame_XXXXX.png          (binary, 0/255)
    """

    def __init__(
        self,
        root_dir: str,
        img_size: int = 512,
        clip_percentile: float = 0.5,
        augment: bool = False,
    ):
        self.img_size = img_size
        self.clip_lo = clip_percentile / 100.0
        self.clip_hi = 1.0 - clip_percentile / 100.0
        self.augment = augment

        # (image_path, mask_path) 쌍 수집
        self.pairs = []
        masks_root = Path(root_dir) / "propagated_masks"
        frames_root = Path(root_dir) / "result_resolution"

        if not masks_root.exists():
            raise FileNotFoundError(f"propagated_masks 폴더 없음: {masks_root}")
        if not frames_root.exists():
            raise FileNotFoundError(f"result_resolution 폴더 없음: {frames_root}")

        for output_dir in sorted(masks_root.iterdir()):
            if not output_dir.is_dir():
                continue
            output_name = output_dir.name          # e.g. "output1"
            frame_dir = frames_root / output_name / "frames"
            if not frame_dir.exists():
                continue
            for mask_file in sorted(output_dir.glob("frame_*.png")):
                frame_file = frame_dir / mask_file.name
                if frame_file.exists():
                    self.pairs.append((str(frame_file), str(mask_file)))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"데이터 쌍을 찾지 못했습니다. 경로 확인: {root_dir}"
            )
        print(f"[Dataset] {len(self.pairs)}개 이미지-마스크 쌍 로드")

        # ImageNet 정규화 (grayscale → 3ch 복사 후 적용)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.pairs)

    def _preprocess_image(self, img_arr: np.ndarray) -> torch.Tensor:
        """
        numpy [H, W] grayscale → torch [3, img_size, img_size] float32
        전처리:
          1) percentile clip (0.5% ~ 99.5%)
          2) [0, 1] 정규화
          3) 3채널 복사
          4) resize to img_size
          5) ImageNet 정규화
        """
        lo = np.percentile(img_arr, self.clip_lo * 100)
        hi = np.percentile(img_arr, self.clip_hi * 100)
        img_arr = np.clip(img_arr, lo, hi)
        if hi > lo:
            img_arr = (img_arr - lo) / (hi - lo)
        else:
            img_arr = np.zeros_like(img_arr, dtype=np.float32)

        img_arr = img_arr.astype(np.float32)
        # [H, W] → [3, H, W]
        img_t = torch.from_numpy(img_arr).unsqueeze(0).repeat(3, 1, 1)
        # resize
        img_t = F.interpolate(
            img_t.unsqueeze(0), size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        # ImageNet 정규화
        img_t = self.normalize(img_t)
        return img_t

    def _preprocess_mask(self, mask_arr: np.ndarray) -> torch.Tensor:
        """
        numpy [H, W] binary mask → torch [1, img_size, img_size] float32 {0, 1}
        """
        mask_arr = (mask_arr > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_t = F.interpolate(
            mask_t, size=(self.img_size, self.img_size),
            mode="nearest"
        ).squeeze(0)  # [1, img_size, img_size]
        return mask_t

    def __getitem__(self, idx):
        frame_path, mask_path = self.pairs[idx]

        img_arr = np.array(Image.open(frame_path).convert("L"))
        mask_arr = np.array(Image.open(mask_path).convert("L"))

        img_t = self._preprocess_image(img_arr)
        mask_t = self._preprocess_mask(mask_arr)

        if self.augment:
            img_t, mask_t = self._random_augment(img_t, mask_t)

        return img_t, mask_t

    def _random_augment(self, img_t, mask_t):
        """학습용 간단한 augmentation (좌우 반전)."""
        if torch.rand(1).item() > 0.5:
            img_t = torch.flip(img_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])
        return img_t, mask_t


def split_dataset(dataset: SMASDataset, val_ratio: float = 0.2, seed: int = 42):
    """전체 데이터를 train / val 로 분할."""
    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    rng = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=rng)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Loss 함수
# ──────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, 1, H, W]  — sigmoid 적용 전
        targets : [B, 1, H, W]  — {0, 1}
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.flatten(1)
        targets_flat = targets.flatten(1)
        inter = (probs_flat * targets_flat).sum(1)
        denom = probs_flat.sum(1) + targets_flat.sum(1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice Loss (1:1 비율)."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 단일 이미지 forward pass
# ──────────────────────────────────────────────────────────────────────────────

def forward_single_image(model, images: torch.Tensor):
    """
    단일 이미지 세그멘테이션 forward pass (메모리 없음, 프롬프트 없음).

    Args:
        model  : SAM2Base (LoRA 적용됨)
        images : [B, 3, 512, 512]  float32

    Returns:
        high_res_masks : [B, 1, 512, 512]  (sigmoid 전 logits)
    """
    B = images.shape[0]
    C = model.hidden_dim  # 256

    # 1) 이미지 인코딩 (forward_image 가 conv_s0/s1 도 적용함)
    backbone_out = model.forward_image(images)

    # 2) Backbone feature 준비 (FPN 처리, flatten)
    _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(
        backbone_out
    )
    # vision_feats[-1] : [HW, B, C] = [1024, B, 256]

    # 3) 메모리 없이 첫 프레임 처리 (directly_add_no_mem_embed 경로)
    #    is_init_cond_frame=True, output_dict 비어 있음
    empty_output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    pix_feat = model._prepare_memory_conditioned_features(
        frame_idx=0,
        is_init_cond_frame=True,
        current_vision_feats=vision_feats,
        current_vision_pos_embeds=vision_pos_embeds,
        feat_sizes=feat_sizes,
        output_dict=empty_output_dict,
        num_frames=1,
    )
    # pix_feat : [B, 256, 32, 32]

    # 4) High-res features (backbone_fpn[0]=128x128, backbone_fpn[1]=64x64)
    #    (이미 forward_image 에서 conv_s0/s1 적용됨)
    high_res_features = [
        backbone_out["backbone_fpn"][0],   # [B, 32, 128, 128]
        backbone_out["backbone_fpn"][1],   # [B, 64, 64, 64]
    ]

    # 5) SAM 헤드 forward (프롬프트 없음 → padding point label=-1 자동 사용)
    sam_outputs = model._forward_sam_heads(
        backbone_features=pix_feat,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=high_res_features,
        multimask_output=False,
    )
    # sam_outputs = (low_res_multimasks, high_res_multimasks, ious,
    #                low_res_masks, high_res_masks, obj_ptr, object_score_logits)
    high_res_masks = sam_outputs[4]  # [B, 1, 512, 512]
    return high_res_masks


# ──────────────────────────────────────────────────────────────────────────────
# 4. 학습 / 검증 루프
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device, warmup_scheduler=None):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred_masks = forward_single_image(model, images)   # [B, 1, 512, 512]
            loss = criterion(pred_masks, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device):
    """
    검증 루프.
    Returns dict: {dice, iou, hausdorff, precision, recall, loss}
    """
    model.eval()
    criterion = BCEDiceLoss()

    dice_list, iou_list, hd_list, prec_list, rec_list, loss_list = [], [], [], [], [], []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        pred_logits = forward_single_image(model, images)    # [B, 1, 512, 512]
        loss = criterion(pred_logits, masks)
        loss_list.append(loss.item())

        pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()

        for i in range(images.shape[0]):
            p = pred_bin[i, 0].cpu().numpy()
            g = masks[i, 0].cpu().numpy()

            inter = (p * g).sum()
            p_sum = p.sum()
            g_sum = g.sum()

            # Dice
            dice = (2 * inter + 1e-6) / (p_sum + g_sum + 1e-6)
            dice_list.append(dice)

            # IoU
            union = p_sum + g_sum - inter
            iou = (inter + 1e-6) / (union + 1e-6)
            iou_list.append(iou)

            # Precision / Recall
            prec = (inter + 1e-6) / (p_sum + 1e-6)
            rec  = (inter + 1e-6) / (g_sum + 1e-6)
            prec_list.append(prec)
            rec_list.append(rec)

            # Hausdorff Distance (95th percentile)
            hd = _hausdorff_95(p, g)
            hd_list.append(hd)

    return {
        "loss":      float(np.mean(loss_list)),
        "dice":      float(np.mean(dice_list)),
        "iou":       float(np.mean(iou_list)),
        "hausdorff": float(np.mean(hd_list)),
        "precision": float(np.mean(prec_list)),
        "recall":    float(np.mean(rec_list)),
    }


def _hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    95th percentile Hausdorff distance.
    pred, gt: binary numpy array [H, W]
    """
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(max(pred.shape))  # 패널티

    dist_pred = distance_transform_edt(1 - pred)
    dist_gt   = distance_transform_edt(1 - gt)

    hd_pred_to_gt = dist_gt[pred > 0]
    hd_gt_to_pred = dist_pred[gt > 0]

    all_dists = np.concatenate([hd_pred_to_gt, hd_gt_to_pred])
    return float(np.percentile(all_dists, 95))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Warmup Scheduler
# ──────────────────────────────────────────────────────────────────────────────

class LinearWarmupScheduler:
    """첫 warmup_steps 스텝 동안 lr 을 0 → base_lr 로 선형 증가."""

    def __init__(self, optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        # lr 을 0 으로 초기화
        for pg in optimizer.param_groups:
            pg["lr"] = 0.0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            ratio = self.current_step / self.warmup_steps
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * ratio


# ──────────────────────────────────────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MedSAM2 + LoRA SMAS Finetuning")
    p.add_argument("--root_dir",    required=True,
                   help="데이터 루트 (result_resolution/, propagated_masks/ 포함)")
    p.add_argument("--medsam2_dir", required=True,
                   help="MedSAM2 소스 디렉토리 경로")
    p.add_argument("--checkpoint",  required=True,
                   help="MedSAM2 체크포인트 .pt 경로")
    p.add_argument("--config",      default="configs/sam2.1_hiera_t512.yaml",
                   help="SAM2 config 파일 (medsam2_dir 기준 상대 경로)")
    p.add_argument("--output_dir",  default="./lora_checkpoints",
                   help="학습 결과 저장 디렉토리")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--rank",        type=int, default=4,  help="LoRA rank")
    p.add_argument("--lora_alpha",  type=float, default=4.0)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--warmup_steps",type=int, default=100)
    p.add_argument("--val_ratio",   type=float, default=0.2)
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--train_decoder", action="store_true",
                   help="mask decoder 도 함께 학습")
    p.add_argument("--resume",      default=None,
                   help="이어서 학습할 LoRA 체크포인트 경로")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 경로 설정 ──
    setup_path(args.medsam2_dir)
    from sam2.build_sam import build_sam2
    from lora_medsam2 import apply_lora_to_model, get_trainable_params
    from lora_medsam2 import save_lora_checkpoint, load_lora_checkpoint

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 디바이스: {device}")

    # ── 모델 로드 ──
    print("[Train] 모델 로딩 중...")
    config_path = str(Path(args.medsam2_dir) / args.config)
    model = build_sam2(
        config_file=config_path,
        ckpt_path=args.checkpoint,
        device=device,
    )
    model = model.to(device)

    # ── LoRA 적용 ──
    apply_lora_to_model(model, r=args.rank, lora_alpha=args.lora_alpha)
    if args.resume:
        load_lora_checkpoint(model, args.resume)

    # ── 데이터셋 ──
    full_dataset = SMASDataset(
        root_dir=args.root_dir,
        img_size=args.img_size,
        augment=False,  # split 후 train set 에만 augment 적용
    )
    train_set, val_set = split_dataset(full_dataset, val_ratio=args.val_ratio)
    # train set 에 augment 활성화
    train_set.dataset.augment = True

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"[Train] train: {len(train_set)}, val: {len(val_set)}")

    # ── Optimizer / Loss / Scaler ──
    param_groups = get_trainable_params(model, train_decoder=args.train_decoder)
    for pg in param_groups:
        pg["lr"] = args.lr

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    criterion = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    warmup_sched = LinearWarmupScheduler(optimizer, warmup_steps=args.warmup_steps)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── 학습 루프 ──
    best_dice = 0.0
    log_path = Path(args.output_dir) / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,dice,iou,hausdorff,precision,recall\n")

    print("[Train] 학습 시작")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            warmup_scheduler=warmup_sched if epoch == 1 else None,
        )
        metrics = validate(model, val_loader, device)

        # warmup 이후 cosine lr decay
        if epoch > 1 or args.warmup_steps == 0:
            lr_scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"Dice={metrics['dice']:.4f} | "
            f"IoU={metrics['iou']:.4f} | "
            f"HD={metrics['hausdorff']:.2f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # 로그 저장
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{metrics['loss']:.6f},"
                f"{metrics['dice']:.6f},{metrics['iou']:.6f},"
                f"{metrics['hausdorff']:.4f},"
                f"{metrics['precision']:.6f},{metrics['recall']:.6f}\n"
            )

        # 베스트 체크포인트 저장
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            ckpt_path = str(Path(args.output_dir) / "best_lora.pt")
            save_lora_checkpoint(model, ckpt_path)
            print(f"  → Best model 저장 (Dice={best_dice:.4f}): {ckpt_path}")

        # 마지막 epoch 체크포인트
        if epoch == args.epochs:
            save_lora_checkpoint(
                model, str(Path(args.output_dir) / "last_lora.pt")
            )

    print(f"\n[Train] 완료! Best Dice = {best_dice:.4f}")
    print(f"  체크포인트: {args.output_dir}/best_lora.pt")
    print(f"  로그:       {log_path}")


if __name__ == "__main__":
    main()
