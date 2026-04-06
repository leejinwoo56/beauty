"""
train_lora_unimatch.py
----------------------
UniMatch V2 + MedSAM2 LoRA Semi-Supervised 학습 파이프라인.

데이터 구조:
    annotation/          ← 직접 주석한 고품질 GT (labeled)
        output1/frame_00001.png, frame_00008.png ...
        output10/ ...
        output20/ ...
        output30/ ...
        output42/ ...
        output44/ ...
    result_resolution/   ← 원본 프레임 전체 (labeled + unlabeled)
        output{N}/frames/frame_00001.png ...

학습 방식 (UniMatch V2):
    L_total = (L_sup + L_unsup) / 2

    L_sup   : labeled frame → forward → BCE+Dice(GT)
    L_unsup : unlabeled weak aug → EMA pseudo-label (sigmoid>0.95)
              unlabeled strong aug x2 → student → confidence-masked BCE

실행:
    python train_lora_unimatch.py

    # 경로 직접 지정
    python train_lora_unimatch.py \\
        --ann_root annotation \\
        --frames_root result_resolution \\
        --lora_ckpt lora_checkpoints/best_lora.pt  # 기존 체크포인트로 warm-start
"""

import argparse
import os
import sys
import math
import time
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent / "MedSAM2"))


# ──────────────────────────────────────────────────────────────────────────────
# 0. 경로 설정
# ──────────────────────────────────────────────────────────────────────────────

LABELED_OUTPUTS  = ["output1", "output10", "output20", "output30","output42"]
ANN_ROOT         = "propagated_masks"
FRAMES_ROOT      = "result_resolution"
OUTPUT_DIR       = "../ train_lora_unimatch_using42_propagated_US_train_decoder"
MEDSAM2_DIR      = "../MedSAM2"
SAM2_CFG         = "configs/sam2.1_hiera_t512.yaml"
SAM2_CKPT        = "./MedSAM2/checkpoints/MedSAM2_US_Heart.pt"
CONF_THRESH      = 0.95
EMA_DECAY        = 0.996


# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터셋
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _load_gray(path: str, img_size: int, clip_pct: float = 0.5) -> torch.Tensor:
    """grayscale PNG → [3, img_size, img_size] float32 (학습과 동일 전처리)."""
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    lo  = np.percentile(arr, clip_pct)
    hi  = np.percentile(arr, 100 - clip_pct)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-6)
    t   = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
    t   = F.interpolate(t.unsqueeze(0), size=(img_size, img_size),
                        mode="bilinear", align_corners=False).squeeze(0)
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return norm(t)


def _load_mask(path: str, img_size: int) -> torch.Tensor:
    """binary mask PNG → [1, img_size, img_size] float32 {0,1}."""
    arr = (np.array(Image.open(path).convert("L")) > 127).astype(np.float32)
    t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t   = F.interpolate(t, size=(img_size, img_size), mode="nearest").squeeze(0)
    return t


# ── CutMix ──────────────────────────────────────────────────────────────────

def obtain_cutmix_box(img_size: int, p: float = 0.5,
                      size_min: float = 0.02, size_max: float = 0.4,
                      ratio_min: float = 0.3):
    """
    CutMix 박스 (x1, y1, x2, y2) 반환. p 확률로 유효, 아니면 (0,0,0,0).
    """
    if random.random() > p:
        return 0, 0, 0, 0
    area = img_size * img_size
    for _ in range(10):
        size   = random.uniform(size_min, size_max) * area
        ratio  = random.uniform(ratio_min, 1 / ratio_min)
        w = int(math.sqrt(size * ratio))
        h = int(math.sqrt(size / ratio))
        if w > img_size or h > img_size:
            continue
        x1 = random.randint(0, img_size - w)
        y1 = random.randint(0, img_size - h)
        return x1, y1, x1 + w, y1 + h
    return 0, 0, 0, 0


# ── Strong Augmentation (초음파용) ──────────────────────────────────────────

def strong_aug(img_t: torch.Tensor) -> torch.Tensor:
    """
    [3, H, W] tensor에 strong augmentation 적용.
    초음파(grayscale) 특성에 맞게: brightness/contrast + gaussian blur
    """
    # brightness/contrast jitter (grayscale이므로 3채널 동일하게)
    if random.random() < 0.8:
        factor_b = random.uniform(0.6, 1.4)
        factor_c = random.uniform(0.6, 1.4)
        img_t = img_t * factor_c + factor_b - 0.5
        img_t = img_t.clamp(img_t.min(), img_t.max())

    # gaussian blur
    if random.random() < 0.5:
        sigma  = random.uniform(0.1, 2.0)
        kernel = max(3, int(2 * math.ceil(2 * sigma) + 1))
        kernel = kernel if kernel % 2 == 1 else kernel + 1
        blur   = transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)
        img_t  = blur(img_t)

    return img_t


class SMASLabeledDataset(Dataset):
    """
    annotation/output{N}/*.png (GT) ↔ result_resolution/output{N}/frames/*.png (원본)
    직접 주석한 고품질 데이터만 사용.
    """

    def __init__(self, ann_root: str, frames_root: str,
                 labeled_outputs: list, img_size: int = 512):
        self.img_size = img_size
        self.pairs = []  # (frame_path, mask_path)

        for output_name in labeled_outputs:
            ann_dir   = Path(ann_root) / output_name
            frame_dir = Path(frames_root) / output_name / "frames"
            if not ann_dir.exists() or not frame_dir.exists():
                print(f"[Labeled] 건너뜀: {output_name}")
                continue
            for mask_file in sorted(ann_dir.glob("*.png")):
                frame_file = frame_dir / mask_file.name
                if frame_file.exists():
                    self.pairs.append((str(frame_file), str(mask_file)))

        if not self.pairs:
            raise RuntimeError(f"labeled 데이터 없음. ann_root={ann_root} 확인 필요")
        print(f"[Labeled]  {len(self.pairs)}개 (annotation 기준 GT)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frame_path, mask_path = self.pairs[idx]
        img  = _load_gray(frame_path, self.img_size)
        mask = _load_mask(mask_path, self.img_size)

        # weak augmentation (horizontal flip only)
        if random.random() < 0.5:
            img  = torch.flip(img,  dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        return img, mask


class SMASUnlabeledDataset(Dataset):
    """
    result_resolution/output{N}/frames/ 전체 프레임 중
    labeled_pairs 에 없는 프레임 = unlabeled 데이터.

    반환: (img_w, img_s1, img_s2, cutmix_box1, cutmix_box2)
        img_w  : weak aug (flip only)
        img_s1 : strong aug + CutMix box1
        img_s2 : strong aug + CutMix box2
    """

    def __init__(self, frames_root: str, labeled_pairs: set,
                 img_size: int = 512):
        self.img_size = img_size
        self.frames   = []

        for output_dir in sorted(Path(frames_root).iterdir()):
            frame_dir = output_dir / "frames"
            if not frame_dir.exists():
                continue
            for f in sorted(frame_dir.glob("frame_*.png")):
                if str(f) not in labeled_pairs:
                    self.frames.append(str(f))

        print(f"[Unlabeled] {len(self.frames)}개 "
              f"(result_resolution 전체 - labeled {len(labeled_pairs)}개 제외)")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path  = self.frames[idx]
        img_w = _load_gray(path, self.img_size)

        # horizontal flip (weak)
        if random.random() < 0.5:
            img_w = torch.flip(img_w, dims=[-1])

        # strong aug x2
        img_s1 = strong_aug(img_w.clone())
        img_s2 = strong_aug(img_w.clone())

        # CutMix 박스
        b1 = obtain_cutmix_box(self.img_size)
        b2 = obtain_cutmix_box(self.img_size)

        return img_w, img_s1, img_s2, torch.tensor(b1), torch.tensor(b2)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Loss 함수
# ──────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).flatten(1)
        t = targets.flatten(1)
        inter = (p * t).sum(1)
        denom = p.sum(1) + t.sum(1)
        return (1 - (2 * inter + self.smooth) / (denom + self.smooth)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Forward pass (train_lora.py와 동일 로직, 독립 구현)
# ──────────────────────────────────────────────────────────────────────────────

def forward_single(model, images: torch.Tensor) -> torch.Tensor:
    """[B,3,512,512] → [B,1,512,512] logits (sigmoid 전)."""
    B      = images.shape[0]
    device = images.device

    backbone_out = model.forward_image(images)
    _, vision_feats, vision_pos_embeds, feat_sizes = \
        model._prepare_backbone_features(backbone_out)

    empty = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    pix_feat = model._prepare_memory_conditioned_features(
        frame_idx=0, is_init_cond_frame=True,
        current_vision_feats=vision_feats,
        current_vision_pos_embeds=vision_pos_embeds,
        feat_sizes=feat_sizes,
        output_dict=empty, num_frames=1,
    )

    high_res_features = [backbone_out["backbone_fpn"][0],
                         backbone_out["backbone_fpn"][1]]

    H, W = feat_sizes[-1]
    sparse = model.default_prompt_embedding.to(device).expand(B, -1, -1)
    dense  = (model.sam_prompt_encoder.no_mask_embed.weight
              .reshape(1, -1, 1, 1).expand(B, -1, H, W))

    (low_res, _, _, _) = model.sam_mask_decoder(
        image_embeddings=pix_feat,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    low_res = low_res.float()
    return F.interpolate(low_res, size=(model.image_size, model.image_size),
                         mode="bilinear", align_corners=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. EMA 업데이트
# ──────────────────────────────────────────────────────────────────────────────

def update_ema(student: nn.Module, teacher: nn.Module, step: int):
    ratio = min(1 - 1 / (step + 1), EMA_DECAY)
    with torch.no_grad():
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.copy_(p_t * ratio + p_s.detach() * (1 - ratio))
        for b_s, b_t in zip(student.buffers(), teacher.buffers()):
            b_t.copy_(b_t * ratio + b_s.detach() * (1 - ratio))


# ──────────────────────────────────────────────────────────────────────────────
# 5. 검증
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    dice_list, iou_list, hd_list, prec_list, rec_list, loss_list = [], [], [], [], [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits  = forward_single(model, images)
        loss_list.append(criterion(logits, masks).item())
        pred    = (torch.sigmoid(logits) > 0.5).float()

        for i in range(images.shape[0]):
            p = pred[i, 0].cpu().numpy()
            g = masks[i, 0].cpu().numpy()
            inter = (p * g).sum()
            p_sum, g_sum = p.sum(), g.sum()
            dice_list.append((2 * inter + 1e-6) / (p_sum + g_sum + 1e-6))
            iou_list.append((inter + 1e-6) / (p_sum + g_sum - inter + 1e-6))
            prec_list.append((inter + 1e-6) / (p_sum + 1e-6))
            rec_list.append((inter + 1e-6) / (g_sum + 1e-6))
            hd_list.append(_hd95(p, g))

    return {
        "loss":      float(np.mean(loss_list)),
        "dice":      float(np.mean(dice_list)),
        "iou":       float(np.mean(iou_list)),
        "hausdorff": float(np.mean(hd_list)),
        "precision": float(np.mean(prec_list)),
        "recall":    float(np.mean(rec_list)),
    }


def _hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(max(pred.shape))
    d1 = distance_transform_edt(1 - pred)[gt > 0]
    d2 = distance_transform_edt(1 - gt)[pred > 0]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# ──────────────────────────────────────────────────────────────────────────────
# 6. 체크포인트 (lora_medsam2.py 와 분리)
# ──────────────────────────────────────────────────────────────────────────────

def save_ckpt(model: nn.Module, path: str):
    state = {k: v for k, v in model.state_dict().items()
             if "lora_" in k or "sam_mask_decoder" in k
             or "default_prompt_embedding" in k}
    torch.save(state, path)
    print(f"[Ckpt] 저장: {path} ({len(state)}개 텐서)")


def load_ckpt(model: nn.Module, path: str):
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Ckpt] 로드: {path}")
    if missing:
        print(f"  missing    : {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  unexpected : {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. 학습 루프
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(student, teacher, labeled_loader, unlabeled_loader,
                optimizer, criterion, device, global_step, scaler):
    student.train()
    teacher.eval()

    total_loss = total_sup = total_unsup = 0.0
    n_batches  = min(len(labeled_loader), len(unlabeled_loader))

    labeled_iter   = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for _ in range(n_batches):
        # ── 데이터 로드 ──
        img_x, mask_x = next(labeled_iter)
        img_u_w, img_u_s1, img_u_s2, box1, box2 = next(unlabeled_iter)

        img_x    = img_x.to(device)
        mask_x   = mask_x.to(device)
        img_u_w  = img_u_w.to(device)
        img_u_s1 = img_u_s1.to(device)
        img_u_s2 = img_u_s2.to(device)
        B_u = img_u_w.shape[0]

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):

            # ── Supervised loss ──
            pred_x  = forward_single(student, img_x)
            loss_sup = criterion(pred_x, mask_x)

            # ── Pseudo-label (EMA teacher, no grad) ──
            with torch.no_grad():
                pred_u_w   = forward_single(teacher, img_u_w)   # [B, 1, H, W]
                conf_u_w   = torch.sigmoid(pred_u_w)            # confidence
                pseudo_u_w = (conf_u_w > CONF_THRESH).float()   # pseudo-label

            # ── CutMix 적용 ──
            img_u_s1, pseudo_s1, conf_s1 = _apply_cutmix(
                img_u_s1, pseudo_u_w, conf_u_w, box1)
            img_u_s2, pseudo_s2, conf_s2 = _apply_cutmix(
                img_u_s2, pseudo_u_w, conf_u_w, box2)

            # ── Student: strong aug 예측 ──
            pred_u_s1 = forward_single(student, img_u_s1)
            pred_u_s2 = forward_single(student, img_u_s2)

            # ── Confidence-masked BCE (신뢰도 낮은 픽셀 제외) ──
            loss_u_s1 = _masked_bce(pred_u_s1, pseudo_s1, conf_s1)
            loss_u_s2 = _masked_bce(pred_u_s2, pseudo_s2, conf_s2)
            loss_unsup = (loss_u_s1 + loss_u_s2) / 2.0

            loss = (loss_sup + loss_unsup) / 2.0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()

        # ── EMA 업데이트 ──
        update_ema(student, teacher, global_step)
        global_step += 1

        total_loss  += loss.item()
        total_sup   += loss_sup.item()
        total_unsup += loss_unsup.item()

    n = max(n_batches, 1)
    return total_loss / n, total_sup / n, total_unsup / n, global_step


def _apply_cutmix(img_s, pseudo, conf, box):
    """CutMix: img_s의 box 영역을 배치 내 flip(0) 이미지로 교체."""
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    img_s    = img_s.clone()
    pseudo   = pseudo.clone()
    conf     = conf.clone()
    img_flip = img_s.flip(0)
    ps_flip  = pseudo.flip(0)
    cf_flip  = conf.flip(0)

    for i in range(img_s.shape[0]):
        x1i, y1i, x2i, y2i = x1[i].item(), y1[i].item(), x2[i].item(), y2[i].item()
        if x2i > x1i and y2i > y1i:
            img_s[i, :, y1i:y2i, x1i:x2i]  = img_flip[i, :, y1i:y2i, x1i:x2i]
            pseudo[i, :, y1i:y2i, x1i:x2i] = ps_flip[i, :, y1i:y2i, x1i:x2i]
            conf[i, :, y1i:y2i, x1i:x2i]   = cf_flip[i, :, y1i:y2i, x1i:x2i]

    return img_s, pseudo, conf


def _masked_bce(logits, pseudo, conf):
    """신뢰도 >= CONF_THRESH 픽셀에만 BCE 적용."""
    mask_valid = (conf >= CONF_THRESH).squeeze(1)          # [B, H, W]
    n_valid    = mask_valid.float().sum().clamp(min=1)
    bce_map    = F.binary_cross_entropy_with_logits(
        logits.squeeze(1), pseudo.squeeze(1), reduction="none")
    return (bce_map * mask_valid.float()).sum() / n_valid


# ──────────────────────────────────────────────────────────────────────────────
# 8. 메인
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UniMatch V2 + MedSAM2 LoRA Semi-supervised")
    p.add_argument("--ann_root",      default=ANN_ROOT)
    p.add_argument("--frames_root",   default=FRAMES_ROOT)
    p.add_argument("--medsam2_dir",   default=MEDSAM2_DIR)
    p.add_argument("--sam2_ckpt",     default=SAM2_CKPT)
    p.add_argument("--output_dir",    default=OUTPUT_DIR)
    p.add_argument("--lora_ckpt",     default=None,
                   help="기존 LoRA 체크포인트로 warm-start (선택)")
    p.add_argument("--epochs",        type=int, default=70)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--num_workers",   type=int, default=8)
    p.add_argument("--rank",          type=int, default=4)
    p.add_argument("--lora_alpha",    type=float, default=4.0)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--n_prompt_tokens", type=int, default=2)
    p.add_argument("--val_ratio",     type=float, default=0.2)
    p.add_argument("--img_size",      type=int, default=512)
    p.add_argument("--train_decoder", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    medsam2_path = str(Path(args.medsam2_dir).resolve())
    if medsam2_path not in sys.path:
        sys.path.insert(0, medsam2_path)

    from sam2.build_sam import build_sam2
    from medsam_finetune_lora.lora_medsam2 import (apply_lora_to_model, add_default_prompt,
                                                   get_trainable_params)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 디바이스: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark     = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32    = True

    # ── 모델 (student) ──
    print("[Train] 모델 로딩...")
    student = build_sam2("configs/sam2.1_hiera_t512.yaml",
                         args.sam2_ckpt, device=device).to(device)

    apply_lora_to_model(student, r=args.rank, lora_alpha=args.lora_alpha)
    add_default_prompt(student, n_tokens=args.n_prompt_tokens)

    if args.lora_ckpt:
        load_ckpt(student, args.lora_ckpt)
        print(f"[Train] warm-start: {args.lora_ckpt}")

    # ── EMA teacher (student 복사, grad 없음) ──
    teacher = deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("[Train] EMA teacher 초기화 완료")

    # ── 데이터셋 ──
    labeled_ds = SMASLabeledDataset(
        ann_root=args.ann_root,
        frames_root=args.frames_root,
        labeled_outputs=LABELED_OUTPUTS,
        img_size=args.img_size,
    )

    # labeled 프레임 경로 집합 (unlabeled에서 제외)
    labeled_paths = {p for p, _ in labeled_ds.pairs}

    unlabeled_ds = SMASUnlabeledDataset(
        frames_root=args.frames_root,
        labeled_pairs=labeled_paths,
        img_size=args.img_size,
    )

    # labeled → train/val 분할
    n_val   = max(1, int(len(labeled_ds) * args.val_ratio))
    n_train = len(labeled_ds) - n_val
    train_labeled, val_labeled = torch.utils.data.random_split(
        labeled_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    print(f"[Train] labeled train={n_train}, val={n_val}, "
          f"unlabeled={len(unlabeled_ds)}")

    pw = args.num_workers > 0
    labeled_loader = DataLoader(
        train_labeled, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=pw,
        prefetch_factor=8 if pw else None,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=pw,
        prefetch_factor=8 if pw else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_labeled, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=pw,
        prefetch_factor=8 if pw else None,
    )

    # ── Optimizer / Loss / Scaler ──
    param_groups = get_trainable_params(student, train_decoder=args.train_decoder)
    for pg in param_groups:
        pg["lr"] = args.lr
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    criterion = BCEDiceLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # polynomial LR decay (UniMatch V2 방식)
    total_iters = args.epochs * min(len(labeled_loader), len(unlabeled_loader))
    def poly_lr(step):
        return max((1 - step / total_iters) ** 0.9, 1e-6 / args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

    # ── 학습 루프 ──
    best_dice  = 0.0
    global_step = 0
    log_path   = Path(args.output_dir) / "train_log_unimatch.csv"

    with open(log_path, "w") as f:
        f.write("epoch,total_loss,sup_loss,unsup_loss,"
                "val_loss,dice,iou,hausdorff,precision,recall\n")

    print("[Train] UniMatch V2 학습 시작\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        total_l, sup_l, unsup_l, global_step = train_epoch(
            student, teacher, labeled_loader, unlabeled_loader,
            optimizer, criterion, device, global_step, scaler)

        lr_scheduler.step()
        metrics = validate(student, val_loader, device, criterion)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={total_l:.4f} (sup={sup_l:.4f} unsup={unsup_l:.4f}) | "
            f"val_loss={metrics['loss']:.4f} | "
            f"Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f} | "
            f"HD={metrics['hausdorff']:.2f} | "
            f"Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | "
            f"{elapsed:.1f}s"
        )

        with open(log_path, "a") as f:
            f.write(f"{epoch},{total_l:.6f},{sup_l:.6f},{unsup_l:.6f},"
                    f"{metrics['loss']:.6f},{metrics['dice']:.6f},"
                    f"{metrics['iou']:.6f},{metrics['hausdorff']:.4f},"
                    f"{metrics['precision']:.6f},{metrics['recall']:.6f}\n")

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            ckpt_path = str(Path(args.output_dir) / "best_lora_unimatch.pt")
            save_ckpt(student, ckpt_path)
            print(f"  → Best 저장 (Dice={best_dice:.4f}): {ckpt_path}")

    save_ckpt(student, str(Path(args.output_dir) / "last_lora_unimatch.pt"))
    print(f"\n[완료] Best Dice={best_dice:.4f}")
    print(f"  체크포인트: {args.output_dir}/best_lora_unimatch.pt")
    print(f"  로그:       {log_path}")


if __name__ == "__main__":
    main()
