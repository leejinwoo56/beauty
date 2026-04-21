"""
infer_lora.py
-------------
LoRA 체크포인트를 적용한 MedSAM2로 초음파 영상에서 SMAS 층 자동 탐지.

전략 (Hybrid):
  Frame 0   : LoRA + learnable default prompt (학습과 동일) → SMAS mask 자동 획득
  Frame 1~N : SAM2 video memory propagation (LoRA backbone 계속 적용)
              → frame 0의 mask가 memory에 저장되어 공간 정보로 활용

디렉토리 구조:
  beauty/
    analog/
      analog_result_24_v1/
        output1/  ← frames_root/video_name
          frame_00001.png
          frame_00002.png
          ...
        output2/
          ...

사용법:
    python infer_lora.py --video_name output1
    ../analog/analog_result_24_v1  폴더 안에 있는 output1 안에 있는 frame을 돌면서 finetuning된 medsam을 적용한다.
    # 여러 폴더 한번에
    python infer_lora.py --video_name output1 output2 output3

    # 경로 직접 지정
    python infer_lora.py \\
        --video_name output1 \\
        --frames_root ../analog/analog_result_24_v1 \\
        --output_root output_lora/lora_masks

    # 단일 프레임 추론 (절대 경로)
    python infer_lora.py \\
        --frame /abs/path/to/frame_00001.png \\
        --output_dir /abs/path/to/output_folder
"""

import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent      # medsam_finetune_lora/
_ROOT    = _HERE.parent                         # beauty/
_MEDSAM2 = _ROOT / "MedSAM2"

sys.path.insert(0, str(_MEDSAM2))

from sam2.build_sam import build_sam2_video_predictor, build_sam2

SAM2_CFG    = "configs/sam2.1_hiera_t512.yaml"
SAM2_CKPT   = str(_MEDSAM2 / "checkpoints" / "MedSAM2_US_Heart.pt")
LORA_CKPT = str(_HERE / "model_checkpoints" / "train_lora_unimatch_aug_exclude5053_188" / "best_lora_unimatch.pt")
FRAMES_ROOT = str(_ROOT / "analog" / "analog_result_24_v1")
OUTPUT_ROOT = str(_HERE / "output_lora" /"train_lora_unimatch_aug_exclude5053_188")


# ──────────────────────────────────────────────────────────────────────────────
# 전처리 (학습과 동일)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_frame(img_path: str, img_size: int = 512) -> torch.Tensor:
    """
    grayscale 초음파 프레임을 학습과 동일한 방식으로 전처리.
    Returns: [3, img_size, img_size] float32 tensor
    """
    img_arr = np.array(Image.open(img_path).convert("L"), dtype=np.float32)

    lo = np.percentile(img_arr, 0.5)
    hi = np.percentile(img_arr, 99.5)
    img_arr = np.clip(img_arr, lo, hi)
    if hi > lo:
        img_arr = (img_arr - lo) / (hi - lo)
    else:
        img_arr = np.zeros_like(img_arr)

    img_t = torch.from_numpy(img_arr).unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
    img_t = F.interpolate(
        img_t.unsqueeze(0), size=(img_size, img_size),
        mode="bilinear", align_corners=False
    ).squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = (img_t - mean) / std
    return img_t


# ──────────────────────────────────────────────────────────────────────────────
# Frame 0 전용: default prompt로 SMAS mask 예측
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_frame0(model, img_path: str, device: torch.device) -> np.ndarray:
    """
    학습 때와 동일한 forward pass로 frame 0의 SMAS mask 예측.

    Returns:
        binary_mask : bool numpy [H_orig, W_orig] — 원본 해상도로 리사이즈
    """
    # 학습과 동일한 전처리
    img_t = preprocess_frame(img_path).unsqueeze(0).to(device)  # [1, 3, 512, 512]

    # 이미지 인코딩
    backbone_out = model.forward_image(img_t)
    _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)

    # 메모리 없이 첫 프레임
    empty_out = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    pix_feat = model._prepare_memory_conditioned_features(
        frame_idx=0,
        is_init_cond_frame=True,
        current_vision_feats=vision_feats,
        current_vision_pos_embeds=vision_pos_embeds,
        feat_sizes=feat_sizes,
        output_dict=empty_out,
        num_frames=1,
    )

    high_res_features = [
        backbone_out["backbone_fpn"][0],
        backbone_out["backbone_fpn"][1],
    ]

    # SAMed default prompt → mask decoder 직접 호출
    B = 1
    H, W = feat_sizes[-1]
    sparse_embeddings = model.default_prompt_embedding.to(device)  # [1, n_tokens, 256]
    dense_embeddings = (
        model.sam_prompt_encoder.no_mask_embed.weight
        .reshape(1, -1, 1, 1)
        .expand(B, -1, H, W)
    )

    (low_res_masks, _, _, _) = model.sam_mask_decoder(
        image_embeddings=pix_feat,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    low_res_masks = low_res_masks.float()
    high_res_masks = F.interpolate(
        low_res_masks,
        size=(model.image_size, model.image_size),
        mode="bilinear",
        align_corners=False,
    )

    # binary mask [512, 512]
    binary = (torch.sigmoid(high_res_masks) > 0.5)[0, 0].cpu().numpy()

    # 원본 이미지 해상도로 리사이즈
    orig = np.array(Image.open(img_path))
    orig_h, orig_w = orig.shape[:2]
    if (orig_h, orig_w) != (512, 512):
        binary_pil = Image.fromarray(binary.astype(np.uint8) * 255).resize(
            (orig_w, orig_h), Image.NEAREST
        )
        binary = np.array(binary_pil) > 127

    return binary.astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
# 단일 프레임 추론
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_single_frame(
    model,
    frame_path: str,
    output_dir: str,
    device: torch.device,
    color: tuple = (0, 200, 0),
    alpha: float = 0.5,
):
    """
    단일 프레임을 추론하고 마스크 + 오버레이 이미지를 저장.

    frame_path : 추론할 프레임 절대 경로 (PNG / JPG)
    output_dir : 결과 저장 폴더 (없으면 자동 생성)

    저장 파일:
        output_dir/<stem>_mask.png    — 이진 마스크 (흰색=SMAS)
        output_dir/<stem>_overlay.png — 오버레이 결과
    """
    frame_path = Path(frame_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SingleFrame] 입력: {frame_path}")

    t_total_start = time.perf_counter()

    # ── 마스크 예측 (predict_frame0 재사용) ──
    t_infer_start = time.perf_counter()
    mask = predict_frame0(model, str(frame_path), device)   # bool [H_orig, W_orig]
    t_infer_end = time.perf_counter()

    mask_uint8 = mask.astype(np.uint8) * 255

    mask_save_path = output_dir / (frame_path.stem + "_mask.png")
    Image.fromarray(mask_uint8).save(str(mask_save_path))
    print(f"  마스크   → {mask_save_path}  (SMAS 픽셀: {mask.sum():,})")

    # ── 원본 프레임 로드 ──
    frame_bgr = cv2.imread(str(frame_path))
    if frame_bgr is None:
        frame_pil = Image.open(frame_path).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # mask 크기가 frame과 다를 경우 맞춤 (원본 해상도 보장)
    fh, fw = frame_bgr.shape[:2]
    if mask.shape != (fh, fw):
        mask_pil = Image.fromarray(mask_uint8).resize((fw, fh), Image.NEAREST)
        mask = np.array(mask_pil) > 127

    # ── 오버레이 생성 및 저장 ──
    overlay   = frame_bgr.copy()
    color_arr = np.array(color, dtype=np.float32)   # BGR
    overlay[mask] = (
        frame_bgr[mask].astype(np.float32) * (1 - alpha) + color_arr * alpha
    ).astype(np.uint8)

    overlay_save_path = output_dir / (frame_path.stem + "_overlay.png")
    cv2.imwrite(str(overlay_save_path), overlay)

    t_total_end = time.perf_counter()

    print(f"  오버레이 → {overlay_save_path}")

    # ── FPS 출력 (단일 프레임 = 1 frame) ──
    total_elapsed = t_total_end - t_total_start
    infer_elapsed = t_infer_end - t_infer_start
    print(f"\n[FPS] 전체 처리 시간  : {total_elapsed:.3f}s  →  전체 FPS  : {1.0 / total_elapsed:.2f} fps")
    print(f"[FPS] 모델 추론 시간  : {infer_elapsed:.3f}s  →  추론 FPS  : {1.0 / infer_elapsed:.2f} fps")


# ──────────────────────────────────────────────────────────────────────────────
# 영상 전체 추론
# ──────────────────────────────────────────────────────────────────────────────

def visualize(frames_dir: str, output_dir: str, fps: float = 30.0,
              color=(0, 200, 0), alpha: float = 0.5):
    """
    오버레이 프레임(PNG) + 영상(MP4)을 output_dir 안에 저장.

    저장 구조:
        output_dir/overlay/frame_XXXXX.png  — 오버레이된 개별 프레임
        output_dir/result.mp4               — 오버레이 영상
    """
    frame_files = sorted(
        f for ext in ("png", "jpg", "jpeg")
        for f in Path(frames_dir).glob(f"frame_*.{ext}")
    )
    mask_files  = sorted(Path(output_dir).glob("frame_*.png"))

    if not frame_files or not mask_files:
        print("  [Viz] 프레임 또는 마스크 없음, 시각화 건너뜀")
        return

    overlay_dir = Path(output_dir) / "overlay"
    overlay_dir.mkdir(exist_ok=True)
    video_path  = Path(output_dir) / "result.mp4"

    sample = cv2.imread(str(frame_files[0]))
    h, w   = sample.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    color_arr = np.array(color, dtype=np.float32)  # BGR

    print(f"  [Viz] 오버레이 생성 중...")
    for i, (fp, mp) in enumerate(zip(frame_files, mask_files)):
        frame = cv2.imread(str(fp))
        mask  = np.array(Image.open(mp).convert("L"))

        overlay = frame.copy()
        region  = mask > 127
        overlay[region] = (
            frame[region].astype(np.float32) * (1 - alpha) + color_arr * alpha
        ).astype(np.uint8)

        # 개별 오버레이 프레임 저장
        cv2.imwrite(str(overlay_dir / fp.name.replace(fp.suffix, ".png")), overlay)
        writer.write(overlay)
        print(f"    [{i+1}/{len(frame_files)}]", end="\r")

    writer.release()
    print(f"\n  [Viz] 오버레이 프레임 → {overlay_dir}")
    print(f"  [Viz] 영상           → {video_path}")


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def infer_video(
    model,
    predictor,
    frames_dir: str,
    output_dir: str,
    device: torch.device,
    fps: float = 30.0,
):
    """
    단일 영상에 대한 추론.

    model     : SAM2Base (LoRA + default prompt 적용)
    predictor : SAM2VideoPredictor (동일 LoRA + checkpoint 적용)
    frames_dir: frame_XXXXX.png 파일들이 있는 디렉토리
    output_dir: 결과 마스크 저장 디렉토리
    """
    frame_files = sorted(
        f for ext in ("png", "jpg", "jpeg")
        for f in Path(frames_dir).glob(f"frame_*.{ext}")
    )
    if not frame_files:
        print(f"[Error] 프레임 없음: {frames_dir}")
        return

    total = len(frame_files)
    print(f"[Infer] {total}개 프레임 처리: {frames_dir}")

    t_total_start = time.perf_counter()

    # ── Step 1: Frame 0 → LoRA default prompt으로 SMAS mask 예측 ──
    print("  [1/3] Frame 0 자동 탐지 중...")
    t_infer_start = time.perf_counter()
    mask0 = predict_frame0(model, str(frame_files[0]), device)
    smas_pixels = mask0.sum()
    print(f"         탐지 완료 — SMAS 픽셀: {smas_pixels:,}")

    if smas_pixels == 0:
        print("[Warning] Frame 0에서 SMAS 탐지 실패. 빈 마스크로 계속 진행.")

    # ── Step 2: Video predictor init + frame 0 mask 주입 ──
    print("  [2/3] Video tracking 초기화...")
    inference_state = predictor.init_state(
        video_path=str(frames_dir), async_loading_frames=False
    )
    predictor.reset_state(inference_state)

    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=mask0,
    )

    # ── Step 3: 전체 영상 propagation ──
    print("  [3/3] 전체 영상 propagation...")
    os.makedirs(output_dir, exist_ok=True)

    for out_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # mask_logits: [n_objects, 1, H, W]
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
        out_path = Path(output_dir) / (frame_files[out_idx].stem + ".png")
        Image.fromarray(mask).save(str(out_path))
        print(f"    [{out_idx + 1:4d}/{total}]", end="\r")

    t_infer_end = time.perf_counter()
    print(f"\n  완료 → {output_dir}")

    # ── Step 4: 오버레이 프레임 + 영상 저장 ──
    visualize(frames_dir, output_dir, fps=fps)

    t_total_end = time.perf_counter()

    # ── FPS 출력 ──
    total_elapsed = t_total_end - t_total_start
    infer_elapsed = t_infer_end - t_infer_start
    print(f"\n[FPS] 총 프레임       : {total}")
    print(f"[FPS] 전체 처리 시간  : {total_elapsed:.2f}s  →  전체 FPS  : {total / total_elapsed:.2f} fps")
    print(f"[FPS]   (모델 추론 {infer_elapsed:.2f}s + 시각화 {total_elapsed - infer_elapsed:.2f}s)")
    print(f"[FPS] 모델 추론 시간  : {infer_elapsed:.2f}s  →  추론 FPS  : {total / infer_elapsed:.2f} fps")


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedSAM2 LoRA SMAS 추론")

    # ── 단일 프레임 모드 ──
    parser.add_argument("--frame",        default=None,
                        help="단일 프레임 절대 경로 (지정 시 단일 프레임 모드로 동작)")
    parser.add_argument("--output_dir",   default=None,
                        help="단일 프레임 모드: 결과 저장 폴더 (절대 경로)")

    # ── 영상 모드 ──
    parser.add_argument("--video_name",   nargs="+", default=None,
                        help="처리할 영상 폴더명 (예: output1 output2)")
    parser.add_argument("--frames_root",  default=FRAMES_ROOT,
                        help="프레임 루트 (기본값: beauty/analog/analog_result_24_v1)")
    parser.add_argument("--output_root",  default=OUTPUT_ROOT,
                        help="영상 모드: 마스크 출력 루트 폴더")

    # ── 공통 ──
    parser.add_argument("--lora_ckpt",    default=LORA_CKPT,
                        help="LoRA 체크포인트 경로")
    parser.add_argument("--n_prompt_tokens", type=int, default=2,
                        help="학습 때와 동일한 프롬프트 토큰 수")
    parser.add_argument("--lora_r",       type=int, default=4,
                        help="LoRA rank (학습 때와 동일하게 설정, 기본값: 4)")
    parser.add_argument("--lora_alpha",   type=float, default=4.0,
                        help="LoRA alpha (기본값: 4.0)")
    parser.add_argument("--img_size",     type=int, default=512)
    args = parser.parse_args()

    # ── 인자 검증 ──
    if args.frame is None and args.video_name is None:
        parser.error("--frame 또는 --video_name 중 하나는 반드시 지정해야 합니다.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] 디바이스: {device}")

    from lora_medsam2 import apply_lora_to_model, add_default_prompt, load_lora_checkpoint

    # ── LoRA 모델 로드 (항상 필요) ──
    print("[Infer] LoRA 모델 로딩 중...")
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model = model.to(device)
    apply_lora_to_model(model, r=args.lora_r, lora_alpha=args.lora_alpha)
    add_default_prompt(model, n_tokens=args.n_prompt_tokens)
    load_lora_checkpoint(model, args.lora_ckpt)
    model.eval()

    # ── 단일 프레임 모드 ──
    if args.frame is not None:
        if args.output_dir is None:
            parser.error("--frame 사용 시 --output_dir 도 지정해야 합니다.")
        print("[Infer] 단일 프레임 모드\n")
        infer_single_frame(model, args.frame, args.output_dir, device)
        print("\n[완료]")
        return

    # ── 영상 모드: Video Predictor 추가 로드 ──
    print("[Infer] Video predictor 로딩 중...")
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    apply_lora_to_model(predictor, r=args.lora_r, lora_alpha=args.lora_alpha)
    add_default_prompt(predictor, n_tokens=args.n_prompt_tokens)
    load_lora_checkpoint(predictor, args.lora_ckpt)
    predictor.eval()
    print("[Infer] 로딩 완료\n")

    for video_name in args.video_name:
        frames_dir = Path(args.frames_root) / video_name / "frames"
        output_dir = Path(args.output_root) / video_name

        if not frames_dir.exists():
            print(f"[Warning] 프레임 폴더 없음: {frames_dir}")
            continue

        infer_video(model, predictor, str(frames_dir), str(output_dir), device)

    print("\n[완료]")


if __name__ == "__main__":
    main()
