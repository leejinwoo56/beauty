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
              unlabeled strong aug x1 → encode → Complementary Dropout
              → feat_s1, feat_s2 → decode → confidence-masked BCE x2

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

# ──────────────────────────────────────────────────────────────────────────────
# 0. 경로 설정
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent      # medsam_finetune_lora/
_ROOT = _HERE.parent                          # beauty/

sys.path.insert(0, str(_ROOT / "MedSAM2"))
sys.path.insert(0, str(_HERE))

# annotation: labeling_data/annotation/output{N}/frame_XXXXX.png
# frames    : analog/analog_result_24_v1/output{N}/frames/frame_XXXXX.png
# output1, output10 → validation 전용 (같은 영상 내 프레임 leakage 방지)
# output20, output30, output42, output44 → 학습 전용
TRAIN_OUTPUTS    = ["output20", "output30","output42","output44"]
VAL_OUTPUTS      = ["output1","output10"]
EXCLUDE_OUTPUTS  = []   # 학습/검증 모두에서 제외할 output 목록. 예: ["output47"]
ANN_ROOT        = str(_ROOT / "labeling_data" / "propagated_masks")
FRAMES_ROOT     = str(_ROOT / "analog" / "analog_result_24_v1")
OUTPUT_DIR      = str(_HERE / "model_checkpoints" / "train_lora_unimatch_aug_exclude5053_188")
MEDSAM2_DIR     = str(_ROOT / "MedSAM2")
SAM2_CFG        = "configs/sam2.1_hiera_t512.yaml"
SAM2_CKPT       = str(_ROOT / "MedSAM2" / "checkpoints" / "MedSAM2_US_Heart.pt")
CONF_THRESH     = 0.95
EMA_DECAY       = 0.996


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


def _load_gray_arr(path: str, clip_pct: float = 0.5) -> np.ndarray:
    """grayscale PNG → 정규화된 float32 numpy [H, W]. 크기 조정 전 단계."""
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    lo  = np.percentile(arr, clip_pct)
    hi  = np.percentile(arr, 100 - clip_pct)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-6)


def _arr_to_tensor(arr: np.ndarray, img_size: int) -> torch.Tensor:
    """[H, W] float32 → normalize → [3, img_size, img_size]."""
    t    = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
    t    = F.interpolate(t.unsqueeze(0), size=(img_size, img_size),
                         mode="bilinear", align_corners=False).squeeze(0)
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return norm(t)


def _mask_arr_to_tensor(arr: np.ndarray, img_size: int) -> torch.Tensor:
    """[H, W] binary float32 → [1, img_size, img_size]."""
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(img_size, img_size), mode="nearest").squeeze(0)


def _random_crop(h: int, w: int,
                 ratio_min: float = 0.6,
                 ratio_max: float = 1.0) -> tuple[int, int, int, int]:
    """
    [H, W] 이미지에서 랜덤 crop 좌표 (y1, y2, x1, x2) 반환.
    ratio_min ~ ratio_max 비율로 높이/너비를 독립적으로 샘플링.
    frame과 mask에 동일하게 적용해 공간 일관성 유지.
    """
    crop_h = int(h * random.uniform(ratio_min, ratio_max))
    crop_w = int(w * random.uniform(ratio_min, ratio_max))
    crop_h = max(crop_h, 32)   # 너무 작은 crop 방지
    crop_w = max(crop_w, 32)
    y1 = random.randint(0, h - crop_h)
    x1 = random.randint(0, w - crop_w)
    return y1, y1 + crop_h, x1, x1 + crop_w


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
                 labeled_outputs: list, img_size: int = 512, augment: bool = True):
        self.img_size = img_size
        self.augment  = augment   # False이면 augmentation 없이 원본만 반환 (val용)
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
        mode = "train+aug" if augment else "val (no aug)"
        print(f"[Labeled/{mode}]  {len(self.pairs)}개  outputs={labeled_outputs}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frame_path, mask_path = self.pairs[idx]

        # 원본 크기로 numpy 로드 (crop 적용 전)
        frame_arr = _load_gray_arr(frame_path)
        mask_arr  = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.float32)

        h, w = frame_arr.shape

        if self.augment:
            # ── Random Crop (frame + mask 동일 영역) ──────────────────────────
            if random.random() < 0.8:
                y1, y2, x1, x2 = _random_crop(h, w, ratio_min=0.6, ratio_max=1.0)
                frame_arr = frame_arr[y1:y2, x1:x2]
                mask_arr  = mask_arr[y1:y2,  x1:x2]

        # numpy → tensor (img_size로 resize + normalize)
        img  = _arr_to_tensor(frame_arr, self.img_size)
        mask = _mask_arr_to_tensor(mask_arr, self.img_size)

        if self.augment:
            # ── Horizontal Flip ───────────────────────────────────────────────
            if random.random() < 0.5:
                img  = torch.flip(img,  dims=[-1])
                mask = torch.flip(mask, dims=[-1])

            # ── Brightness / Contrast Jitter ──────────────────────────────────
            if random.random() < 0.6:
                img = img * random.uniform(0.7, 1.3) + random.uniform(-0.1, 0.1)

        return img, mask


class SMASUnlabeledDataset(Dataset):
    """
    result_resolution/output{N}/frames/ 전체 프레임 중
    labeled_pairs 에 없는 프레임 = unlabeled 데이터.

    반환: (img_w, img_s)
        img_w : weak aug (flip only) → Teacher pseudo-label 생성용
        img_s : strong aug x1       → Student encode 입력용
                                      (채널 드롭아웃은 train loop에서 피처 레벨로 처리)
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

        # strong aug x1 (두 번째 뷰는 피처 레벨 Complementary Dropout으로 생성)
        img_s = strong_aug(img_w.clone())

        return img_w, img_s


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
# 3. Forward pass — 인코더 / 디코더 분리
# ──────────────────────────────────────────────────────────────────────────────

def encode(model, images: torch.Tensor):
    """
    이미지 → 인코더 피처.
    반환: (pix_feat, high_res_features, feat_sizes)
        pix_feat         : [B, C, H_f, W_f]  메모리 조건화된 메인 피처
        high_res_features: [fpn_0, fpn_1]     FPN 고해상도 피처 (디코더용)
        feat_sizes       : [(H0,W0), ...]     각 레벨 공간 크기
    """
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

    return pix_feat, high_res_features, feat_sizes


def decode(model, pix_feat: torch.Tensor,
           high_res_features: list, feat_sizes: list) -> torch.Tensor:
    """
    피처 → [B, 1, img_size, img_size] logits.
    pix_feat를 교체해서 호출하면 Complementary Dropout 뷰를 따로 디코딩할 수 있다.
    """
    B      = pix_feat.shape[0]
    device = pix_feat.device

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


def forward_single(model, images: torch.Tensor) -> torch.Tensor:
    """[B,3,512,512] → [B,1,512,512] logits. supervised 및 validate 전용."""
    pix_feat, high_res_features, feat_sizes = encode(model, images)
    return decode(model, pix_feat, high_res_features, feat_sizes)


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
                optimizer, criterion, device, global_step, scaler, epoch=0):
    student.train()
    teacher.eval()

    total_loss = total_sup = total_unsup = 0.0
    n_batches  = min(len(labeled_loader), len(unlabeled_loader))

    labeled_iter   = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx in range(n_batches):
        # ── 데이터 로드 ──
        img_x, mask_x = next(labeled_iter)
        img_u_w, img_u_s = next(unlabeled_iter)

        img_x   = img_x.to(device)
        mask_x  = mask_x.to(device)
        img_u_w = img_u_w.to(device)
        img_u_s = img_u_s.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):

            # ── Supervised loss ──
            pred_x   = forward_single(student, img_x)
            loss_sup = criterion(pred_x, mask_x)

            # ── Pseudo-label (EMA teacher, no grad) ──
            with torch.no_grad():
                pred_u_w   = forward_single(teacher, img_u_w)   # [B, 1, H, W]
                conf_u_w   = torch.sigmoid(pred_u_w)            # confidence
                pseudo_u_w = (conf_u_w > CONF_THRESH).float()   # pseudo-label

            # conf 통과 픽셀 모니터링
            valid_px = pseudo_u_w.sum().item()
            total_px = pseudo_u_w.numel()
            avg_conf = conf_u_w.mean().item()
            print(f"  [E{epoch} B{batch_idx+1}/{n_batches}] "
                  f"conf_avg={avg_conf:.4f} | "
                  f"threshold({CONF_THRESH}) 통과: {int(valid_px)}/{total_px} "
                  f"({100*valid_px/total_px:.2f}%)")

            # ── Student: 인코더 1회 통과 → 피처 획득 ──
            pix_feat, high_res_feats, feat_sizes = encode(student, img_u_s)

            # ── Complementary Dropout (UniMatch V2 핵심 로직) ──
            # 채널 차원으로 이진 마스크 M을 생성하고, 상보적 두 뷰를 만든다.
            # feat_s1 = feat * M * 2,  feat_s2 = feat * (1-M) * 2
            # → 두 뷰의 기댓값이 원본 feat와 동일하도록 ×2 스케일링
            C = pix_feat.shape[1]
            B_u = pix_feat.shape[0]
            mask_m = torch.distributions.binomial.Binomial(
                1, 0.5 * torch.ones(B_u, C, 1, 1, device=device)
            ).sample()
            feat_s1 = pix_feat * mask_m * 2.0
            feat_s2 = pix_feat * (1.0 - mask_m) * 2.0

            # ── 디코더: 두 뷰 각각 예측 ──
            pred_u_s1 = decode(student, feat_s1, high_res_feats, feat_sizes)
            pred_u_s2 = decode(student, feat_s2, high_res_feats, feat_sizes)

            # ── Confidence-masked BCE: 두 뷰 모두 동일한 pseudo-label과 비교 ──
            loss_u_s1  = _masked_bce(pred_u_s1, pseudo_u_w, conf_u_w)
            loss_u_s2  = _masked_bce(pred_u_s2, pseudo_u_w, conf_u_w)
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
    from lora_medsam2 import (apply_lora_to_model, add_default_prompt,
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
    # output1, output10 → val 전용 (영상 내 연속 프레임 leakage 방지)
    # output20, output30, output42, output44 → train 전용
    train_outputs = [o for o in TRAIN_OUTPUTS if o not in EXCLUDE_OUTPUTS]
    val_outputs   = [o for o in VAL_OUTPUTS   if o not in EXCLUDE_OUTPUTS]
    if EXCLUDE_OUTPUTS:
        print(f"[EXCLUDE] 제외된 outputs: {EXCLUDE_OUTPUTS}")

    train_labeled = SMASLabeledDataset(
        ann_root=args.ann_root,
        frames_root=args.frames_root,
        labeled_outputs=train_outputs,
        img_size=args.img_size,
        augment=True,
    )
    val_labeled = SMASLabeledDataset(
        ann_root=args.ann_root,
        frames_root=args.frames_root,
        labeled_outputs=val_outputs,
        img_size=args.img_size,
        augment=False,   # val은 augmentation 없이 원본 그대로 평가
    )

    # labeled 전체 경로 집합 (unlabeled에서 제외)
    labeled_paths = (
        {p for p, _ in train_labeled.pairs} |
        {p for p, _ in val_labeled.pairs}
    )

    unlabeled_ds = SMASUnlabeledDataset(
        frames_root=args.frames_root,
        labeled_pairs=labeled_paths,
        img_size=args.img_size,
    )

    print(f"[Train] train={len(train_labeled)}, val={len(val_labeled)}, "
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
            optimizer, criterion, device, global_step, scaler, epoch=epoch)

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
