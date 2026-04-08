"""
inference.py
-----------
infer_lora_video.py의 AI 추론 로직을 FastAPI용으로 분리한 모듈.

원본 infer_lora_video.py와의 차이점:
- tkinter (파일 다이얼로그) 의존성 완전 제거
- 파일 경로를 인자로 받아 처리
- GPU/CPU 자동 감지 (autocast를 조건부 적용)
- load_models() 함수로 모델 로딩을 서버 시작 시 한 번만 처리
"""

import os
import sys
import tempfile
import shutil
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from contextlib import contextmanager
from PIL import Image

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
# __file__ = beauty/webpage/backend/inference.py
_BACKEND  = Path(__file__).resolve().parent        # beauty/webpage/backend/
_WEBPAGE  = _BACKEND.parent                        # beauty/webpage/
_ROOT     = _WEBPAGE.parent                        # beauty/
_MEDSAM2  = _ROOT / "MedSAM2"
_LORA_DIR = _ROOT / "medsam_finetune_lora"

# MedSAM2와 LoRA 모듈을 Python 검색 경로에 추가
sys.path.insert(0, str(_MEDSAM2))
sys.path.insert(0, str(_LORA_DIR))

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from lora_medsam2 import apply_lora_to_model, add_default_prompt, load_lora_checkpoint

# ── 모델 경로 상수 ─────────────────────────────────────────────────────────────
SAM2_CFG  = "configs/sam2.1_hiera_t512.yaml"
SAM2_CKPT = str(_MEDSAM2 / "checkpoints" / "MedSAM2_US_Heart.pt")
LORA_CKPT = str(
    _LORA_DIR / "model_checkpoints"
    / "train_lora_unimatch_aug_usingall"
    / "best_lora_unimatch.pt"
)


# ── autocast 컨텍스트 (GPU가 없으면 비활성화) ──────────────────────────────────
@contextmanager
def maybe_autocast(device: torch.device):
    """CUDA가 있으면 bfloat16 autocast 사용, 없으면 그냥 실행."""
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


# ── 모델 로딩 ─────────────────────────────────────────────────────────────────

def load_models(
    device: torch.device,
    lora_r: int = 4,
    lora_alpha: float = 4.0,
    n_prompt_tokens: int = 2,
):
    """
    MedSAM2 + LoRA 모델 두 개를 로드.
    - model: 단일 프레임 예측용 (Frame 0 초기 마스크 생성)
    - predictor: 전체 영상 propagation용
    서버 시작 시 한 번만 호출하여 메모리에 유지.
    """
    print(f"[Model] SAM2 체크포인트: {SAM2_CKPT}")
    print(f"[Model] LoRA 체크포인트: {LORA_CKPT}")

    # 단일 프레임 모델
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model = model.to(device)
    apply_lora_to_model(model, r=lora_r, lora_alpha=lora_alpha)
    add_default_prompt(model, n_tokens=n_prompt_tokens)
    load_lora_checkpoint(model, LORA_CKPT)
    model.eval()

    # 비디오 propagation 모델
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    apply_lora_to_model(predictor, r=lora_r, lora_alpha=lora_alpha)
    add_default_prompt(predictor, n_tokens=n_prompt_tokens)
    load_lora_checkpoint(predictor, LORA_CKPT)
    predictor.eval()

    return model, predictor


# ── 프레임 전처리 ──────────────────────────────────────────────────────────────

def preprocess_frame(img_path: str, img_size: int = 512) -> torch.Tensor:
    """
    단일 이미지를 모델 입력 형식으로 변환.
    - 그레이스케일 변환
    - 0.5~99.5 백분위수 기반 정규화 (초음파 영상의 밝기 편차 대응)
    - ImageNet 평균/표준편차로 표준화
    - 512×512 리사이즈
    """
    img_arr = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
    lo = np.percentile(img_arr, 0.5)
    hi = np.percentile(img_arr, 99.5)
    img_arr = np.clip(img_arr, lo, hi)
    if hi > lo:
        img_arr = (img_arr - lo) / (hi - lo)
    else:
        img_arr = np.zeros_like(img_arr)

    img_t = torch.from_numpy(img_arr).unsqueeze(0).repeat(3, 1, 1)
    img_t = F.interpolate(
        img_t.unsqueeze(0), size=(img_size, img_size),
        mode="bilinear", align_corners=False
    ).squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_t - mean) / std


# ── Frame 0 마스크 예측 ────────────────────────────────────────────────────────

@torch.no_grad()
def predict_frame0(model, img_path: str, device: torch.device) -> np.ndarray:
    """
    첫 번째 프레임에서 SMAS 마스크를 자동 생성.
    이 마스크를 시작점으로 나머지 프레임에 propagation한다.
    """
    img_t = preprocess_frame(img_path).unsqueeze(0).to(device)

    backbone_out = model.forward_image(img_t)
    _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)

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

    B = 1
    H, W = feat_sizes[-1]
    sparse_embeddings = model.default_prompt_embedding.to(device)
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
        mode="bilinear", align_corners=False,
    )

    binary = (torch.sigmoid(high_res_masks) > 0.5)[0, 0].cpu().numpy()

    # 원본 해상도로 복원
    orig = np.array(Image.open(img_path))
    orig_h, orig_w = orig.shape[:2]
    if (orig_h, orig_w) != (512, 512):
        binary_pil = Image.fromarray(binary.astype(np.uint8) * 255).resize(
            (orig_w, orig_h), Image.NEAREST
        )
        binary = np.array(binary_pil) > 127

    return binary.astype(bool)


# ── 영상 → 프레임 추출 ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, frames_dir: str):
    """
    mp4 등 영상을 PNG 프레임 시퀀스로 분해.
    SAM2 video predictor는 폴더 내 PNG 파일을 순서대로 읽는 방식 사용.
    Returns: (총 프레임 수, fps, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(frames_dir, exist_ok=True)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(Path(frames_dir) / f"frame_{idx+1:05d}.png"), frame)
        idx += 1
    cap.release()
    return idx, fps, width, height


# ── 추론 + 오버레이 영상 생성 ──────────────────────────────────────────────────

def infer_and_render(
    model,
    predictor,
    frames_dir: str,
    output_video_path: str,
    device: torch.device,
    fps: float = 30.0,
    color: tuple = (0, 200, 0),   # BGR 기준 초록색
    alpha: float = 0.5,
):
    """
    프레임 폴더 → SMAS 마스크 오버레이 영상 생성.
    1. Frame 0에서 마스크 자동 생성
    2. SAM2 video predictor로 전체 영상에 propagation
    3. 각 프레임에 반투명 초록색 오버레이 합성 후 저장
    """
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frame_files:
        raise ValueError("프레임 파일이 없습니다.")

    total = len(frame_files)

    with torch.inference_mode(), maybe_autocast(device):
        # Step 1: 첫 프레임 마스크
        mask0 = predict_frame0(model, str(frame_files[0]), device)

        # Step 2: propagation 초기화
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

        # Step 3: 전체 영상 propagation
        masks: dict[int, np.ndarray] = {}
        for out_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
            masks[out_idx] = mask

    # Step 4: 오버레이 비디오 저장
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    color_arr = np.array(color, dtype=np.float32)

    for i, fp in enumerate(frame_files):
        frame = cv2.imread(str(fp))
        mask  = masks.get(i)

        if mask is not None and mask.any():
            if mask.shape != (h, w):
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
                ) > 127
            overlay = frame.copy()
            overlay[mask] = (
                frame[mask].astype(np.float32) * (1 - alpha) + color_arr * alpha
            ).astype(np.uint8)
            writer.write(overlay)
        else:
            writer.write(frame)

    writer.release()

    # ── ffmpeg로 H.264 재인코딩 ────────────────────────────────────────────────
    # OpenCV의 mp4v 코덱은 브라우저에서 재생 불가.
    # ffmpeg로 H.264(libx264)로 변환해야 video 태그에서 재생됨.
    _reencode_to_h264(output_video_path)


def _reencode_to_h264(video_path: str):
    """
    mp4v로 저장된 영상을 H.264로 재인코딩.
    브라우저 호환성을 위해 필수.
    임시 파일로 변환 후 원본 위치에 덮어씀.
    """
    tmp_path = video_path.replace(".mp4", "_tmp.mp4")
    cmd = [
        "ffmpeg", "-y",           # -y: 덮어쓰기 확인 없이 진행
        "-i", video_path,         # 입력: mp4v 파일
        "-vcodec", "libx264",     # 출력 코덱: H.264
        "-crf", "23",             # 화질 (낮을수록 고화질, 18~28 권장)
        "-preset", "fast",        # 인코딩 속도/압축 트레이드오프
        "-pix_fmt", "yuv420p",    # 브라우저 호환 픽셀 포맷
        tmp_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp_path, video_path)  # 임시 파일을 원본 위치로 이동


# ── 최상위 진입 함수 ───────────────────────────────────────────────────────────

def process_video(
    model,
    predictor,
    input_path: str,
    output_path: str,
    device: torch.device,
):
    """
    영상 파일 하나를 받아 SMAS 탐지 결과 영상으로 저장.
    임시 폴더에 프레임을 추출하고 추론 후 자동 삭제.
    """
    tmp_dir = tempfile.mkdtemp(prefix="smas_web_")
    try:
        _, fps, _, _ = extract_frames(input_path, tmp_dir)
        infer_and_render(model, predictor, tmp_dir, output_path, device, fps=fps)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)