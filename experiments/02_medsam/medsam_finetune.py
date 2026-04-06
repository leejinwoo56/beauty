"""
medsam_finetune.py

output_segmentation_video/result*/frames + masks 를 이용해
MedSAM (SAM ViT-B) 을 SMAS 층 검출용으로 fine-tuning.

제외 폴더: result6, result7, result9, result16, result17, result19

실행 예시
---------
python medsam_finetune.py \
    --data_root output_segmentation_video \
    --checkpoint medsam_vit_b.pth \
    --save_dir checkpoints \
    --epochs 20 \
    --batch_size 2 \
    --lr 1e-4
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from segment_anything import sam_model_registry

# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE_RESULTS = {6, 7, 9, 16, 17, 19}
SAM_INPUT_SIZE  = 1024   # SAM 이미지 인코더 입력 크기
SAM_MASK_SIZE   = 256    # SAM mask decoder 출력 크기
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# RLE 디코더 (mask_to_rle 의 역변환)
# ─────────────────────────────────────────────────────────────────────────────

def rle_to_mask(rle: dict) -> np.ndarray:
    """
    {"size": [H, W], "counts": [...]} → (H, W) bool 마스크.
    build_layer_masks 에서 Fortran-order(column-major) 로 인코딩했으므로
    동일하게 디코딩.
    """
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0  # counts 는 항상 0(배경) 부터 시작
    for run in counts:
        flat[pos:pos + run] = val
        pos += run
        val ^= 1  # 0 ↔ 1 교대
    return flat.reshape((w, h)).T.astype(bool)  # Fortran → C order


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────name───────
# ─────────────────────────────────────────────────────────────────────────────

class SMASDataset(Dataset):
    """
    result*/frames/{name}.png + result*/masks/{name}.json 쌍 로드.

    __getitem__ 반환:
        image_tensor : (3, 1024, 1024) float32, [0, 1] 정규화
        mask_tensor  : (1, 256, 256)   float32, 0/1 이진 마스크
        box_tensor   : (4,)            float32, [x1, y1, x2, y2] in 1024 scale
    """

    def __init__(self, data_root: str, exclude: set = EXCLUDE_RESULTS):
        self.samples = []   # list of (frame_path, mask_path)

        base = Path(data_root)
        for result_dir in sorted(base.glob("result*")):
            if not result_dir.is_dir():
                continue
            # 번호 파싱
            try:
                num = int(result_dir.name.replace("result", ""))
            except ValueError:
                continue
            if num in exclude:
                continue

            frame_dir = result_dir / "frames"
            mask_dir  = result_dir / "masks"
            if not frame_dir.exists() or not mask_dir.exists():
                continue

            for frame_path in sorted(frame_dir.glob("*.png")):
                mask_path = mask_dir / (frame_path.stem + ".json")
                if mask_path.exists():
                    self.samples.append((frame_path, mask_path))

        print(f"[Dataset] 총 {len(self.samples)}개 샘플 로드 (제외: result{sorted(exclude)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, mask_path = self.samples[idx]

        # ── 이미지 로드 및 전처리 ─────────────────────────────────────────
        img_bgr = cv2.imread(str(frame_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        orig_h, orig_w = img_rgb.shape[:2]

        # min-max 정규화 → [0, 1]
        lo, hi = img_rgb.min(), img_rgb.max()
        if hi > lo:
            img_rgb = (img_rgb - lo) / (hi - lo)
        else:
            img_rgb = np.zeros_like(img_rgb)

        # 1024×1024 리사이즈
        img_1024 = cv2.resize(img_rgb, (SAM_INPUT_SIZE, SAM_INPUT_SIZE),
                              interpolation=cv2.INTER_CUBIC)
        image_tensor = torch.from_numpy(img_1024.transpose(2, 0, 1))  # (3, H, W)

        # ── 마스크 로드 및 전처리 ─────────────────────────────────────────
        with open(mask_path) as f:
            ann = json.load(f)

        rle  = ann["annotations"][0]["segmentation"]
        bbox = ann["annotations"][0]["bbox"]          # [x, y, w, h]

        mask_bool = rle_to_mask(rle)                  # (orig_h, orig_w) bool

        # 256×256 리사이즈 (SAM mask decoder 출력 크기)
        mask_256 = cv2.resize(mask_bool.astype(np.uint8),
                              (SAM_MASK_SIZE, SAM_MASK_SIZE),
                              interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_256).float().unsqueeze(0)  # (1, 256, 256)

        # ── Box prompt: 1024 스케일로 변환 ──────────────────────────────
        bx, by, bw, bh = bbox
        sx = SAM_INPUT_SIZE / orig_w
        sy = SAM_INPUT_SIZE / orig_h
        # bbox가 빈 경우 이미지 전체를 box로 사용
        if bw == 0 or bh == 0:
            box = np.array([0, 0, SAM_INPUT_SIZE, SAM_INPUT_SIZE], dtype=np.float32)
        else:
            box = np.array([bx * sx, by * sy,
                            (bx + bw) * sx, (by + bh) * sy], dtype=np.float32)
        box_tensor = torch.from_numpy(box)  # (4,)

        return image_tensor, mask_tensor, box_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """pred, target: (B, 1, H, W), sigmoid 적용 전 logit."""
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


def seg_loss(pred_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice Loss + BCE Loss."""
    bce = F.binary_cross_entropy_with_logits(pred_logit, target)
    dice = dice_loss(pred_logit, target)
    return bce + dice


# ─────────────────────────────────────────────────────────────────────────────
# Training / Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0

    with torch.set_grad_enabled(train):
        for images, masks, boxes in loader:
            images = images.to(device)          # (B, 3, 1024, 1024)
            masks  = masks.to(device)           # (B, 1, 256, 256)
            boxes  = boxes.to(device)           # (B, 4)

            # ── 이미지 인코딩 (freeze) ───────────────────────────────────
            with torch.no_grad():
                image_embeddings = model.image_encoder(images)  # (B, 256, 64, 64)

            # ── Box prompt 인코딩 ─────────────────────────────────────────
            # SAM prompt_encoder 는 boxes를 (B, 1, 4) 형태로 받음
            boxes_1n4 = boxes.unsqueeze(1)  # (B, 1, 4)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None,
                boxes=boxes_1n4,
                masks=None,
            )

            # ── Mask 디코딩 ──────────────────────────────────────────────
            low_res_logits, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            # low_res_logits: (B, 1, 256, 256)

            loss = seg_loss(low_res_logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Dice score (metric용)
            with torch.no_grad():
                pred = (torch.sigmoid(low_res_logits) > 0.5).float()
                inter = (pred * masks).sum(dim=(1, 2, 3))
                union = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                batch_dice = ((2 * inter + 1e-6) / (union + 1e-6)).mean().item()

            total_loss += loss.item()
            total_dice += batch_dice

    n = len(loader)
    return total_loss / n, total_dice / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="output_segmentation_video")
    parser.add_argument("--checkpoint", required=True,
                        help="MedSAM 또는 SAM ViT-B checkpoint (.pth)")
    parser.add_argument("--save_dir",   default="checkpoints")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.15,
                        help="검증 셋 비율")
    parser.add_argument("--num_workers",type=int,   default=0)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 모델 로드 ─────────────────────────────────────────────────────────
    print(f"[Model] {args.model_type} checkpoint: {args.checkpoint}")
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model.to(device)

    # 이미지 인코더는 freeze, prompt encoder + mask decoder 만 학습
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    trainable = list(model.prompt_encoder.parameters()) + \
                list(model.mask_decoder.parameters())
    print(f"[Model] trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 데이터셋 ─────────────────────────────────────────────────────────
    dataset = SMASDataset(args.data_root)
    if len(dataset) == 0:
        print("[Error] 로드된 샘플이 없습니다. 경로를 확인하세요.")
        return

    val_size   = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"[Dataset] train={train_size}, val={val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    # ── 체크포인트 저장 경로 ──────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_dice  = 0.0
    best_epoch = 0

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss,   val_dice   = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
              f"val   loss={val_loss:.4f}   dice={val_dice:.4f}")

        # 매 epoch 저장
        ckpt_path = save_dir / f"smas_epoch{epoch:03d}.pth"
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_dice":    val_dice,
        }, ckpt_path)

        # best 체크포인트 갱신
        if val_dice > best_dice:
            best_dice  = val_dice
            best_epoch = epoch
            best_path  = save_dir / "smas_best.pth"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_dice":    val_dice,
            }, best_path)
            print(f"  → best 갱신 (epoch={epoch}, val_dice={val_dice:.4f})")

    print(f"\n[Done] best epoch={best_epoch}, val_dice={best_dice:.4f}")
    print(f"       best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
