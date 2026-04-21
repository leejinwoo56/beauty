"""
analog_detect_video.py
----------------------
지정한 번호의 영상을 읽어 analog 탐지를 수행하고 결과를 저장.

── 모드 1: 영상 직접 처리 (기본, --num) ─────────────────────────────────────
입력:  beauty/24/Image{N}.mp4
출력:  beauty/analog/analog_result_24_v1/output{N}/
          frames/frame_00001.png  ← ROI 크롭 원본 프레임
          masks/ frame_00001.png  ← SMAS 마스크 (이진 PNG)
          vis/  frame_00001.png   ← 오버레이 프레임
          output{N}.mp4           ← 최종 오버레이 영상

── 모드 2: 저장된 프레임 재처리 (--from_frames) ────────────────────────────
입력:  beauty/analog/analog_result_24_v1/output{N}/frames/*.png  (ROI 이미 적용됨)
출력:  beauty/analog/analog_result_24_v1/output{N}/
          dermis_masks/frame_00001.png  ← 진피 마스크 (이진 PNG)
       beauty/labeling_data/multiclass_masks/output{N}/
          frame_00001.png               ← class-index map (0=배경 1=진피 2=SMAS)
                                           SMAS 소스: propagated_masks 우선, 없으면 자동 탐지

사용법:
    # 모드 1: 영상 처리
    python analog_detect_video.py --num 45
    python analog_detect_video.py --num 1 2 55-57

    # 모드 2: 저장된 프레임으로 진피·multiclass GT 생성
    python analog_detect_video.py --from_frames --num 20 30 42 44
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

from analog_detect import (
    preprocess,
    find_epidermis_boundary_region,
    detect_all_peak_lines,
    select_L2_L3,
    check_fat_layer,
    build_layer_masks,
    render_overlay,
)

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent          # analog/
_ROOT        = _HERE.parent                             # beauty/
_INPUT_DIR   = _ROOT / "24"                             # beauty/24/
_OUTPUT_DIR  = _HERE / "analog_result_24_v1"            # beauty/analog/analog_result_24_v1/

# 학습용 GT 경로
_PROP_MASKS  = _ROOT / "labeling_data" / "propagated_masks"   # 기존 SMAS binary (수동 라벨)
_MULTI_MASKS = _ROOT / "labeling_data" / "multiclass_masks"   # 병합 class-index map (학습용)
# multiclass 픽셀 값: 0=배경, 1=진피, 2=SMAS

LAYER_ALPHA       = 0.23
SMAS_BRIGHT_ALPHA = 0.25
SMAS_THICK_ALPHA  = 0.69
DEFAULT_ROI       = (50, 70, 1100, 700)
#133, 70, 1015, 700

# ── 프레임 단위 처리 ───────────────────────────────────────────────────────────

def process_frame(frame_bgr, roi=DEFAULT_ROI):
    """
    프레임 하나를 처리.

    Args:
        frame_bgr : BGR 이미지
        roi       : (x1, y1, x2, y2) 크롭 영역. None이면 크롭 없이 전체 사용.

    Returns:
        crop        : (크롭된) 원본 프레임 (BGR)
        vis         : 오버레이 이미지 (BGR)
        smas_mask   : SMAS 마스크 (uint8, 0 or 255)
        dermis_mask : 진피 마스크 (uint8, 0 or 255)
    """
    h_full, w_full = frame_bgr.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_full, x2); y2 = min(h_full, y2)
        crop = frame_bgr[y1:y2, x1:x2]
    else:
        crop = frame_bgr

    h, w = crop.shape[:2]
    enh, smooth = preprocess(crop)
    boundary_mask, boundary_max_y = find_epidermis_boundary_region(enh, smooth)
    start_y = max(0, boundary_max_y + 1)
    all_lines = detect_all_peak_lines(enh, smooth, start_y)

    L1 = np.full(w, boundary_max_y, dtype=np.int32)
    if boundary_mask.any():
        bnd_rows, bnd_cols = np.where(boundary_mask)
        for x in range(w):
            col_rows = bnd_rows[bnd_cols == x]
            if len(col_rows) > 0:
                L1[x] = int(col_rows.max())
    L1_mean_y = int(L1.mean())

    L2_line, L3_line = select_L2_L3(all_lines, L1_mean_y, w)

    if L3_line is None:
        L2_line = np.full(w, h * 2 // 3, dtype=np.int32)
        L3_line = np.full(w, h * 3 // 4, dtype=np.int32)
    elif L2_line is None:
        L2_line = ((L1.astype(np.float32) + L3_line.astype(np.float32)) / 2).astype(np.int32)

    has_fat = check_fat_layer(enh, L1, L2_line, L3_line, bright_diff_thr=10000.0)
    layer_masks = build_layer_masks(enh, boundary_mask, [L1, L2_line, L3_line], has_fat)

    vis = render_overlay(crop, layer_masks,
                         alpha=LAYER_ALPHA,
                         smas_bright_alpha=SMAS_BRIGHT_ALPHA,
                         smas_thick_alpha=SMAS_THICK_ALPHA)

    smas_mask   = layer_masks["smas"].astype(np.uint8) * 255
    dermis_mask = layer_masks["epidermis"].astype(np.uint8) * 255

    return crop, vis, smas_mask, dermis_mask


# ── 모드 1: 영상 직접 처리 ────────────────────────────────────────────────────

def process_one(n: int):
    """영상 번호 n 하나를 처리 (기존 동작 유지)."""
    input_path  = _INPUT_DIR / f"Image{n}.mp4"
    output_root = _OUTPUT_DIR / f"output{n}"

    if not input_path.exists():
        print(f"[Error] 파일 없음: {input_path}")
        return

    frames_dir = output_root / "frames"
    masks_dir  = output_root / "masks"
    vis_dir    = output_root / "vis"
    for d in (frames_dir, masks_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    cap    = cv2.VideoCapture(str(input_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    x1, y1, x2, y2 = DEFAULT_ROI
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_full, x2); y2 = min(h_full, y2)
    crop_w, crop_h = x2 - x1, y2 - y1
    print(f"[Input]  {input_path.name}  원본 {w_full}x{h_full} → ROI {crop_w}x{crop_h} @ {fps:.1f}fps, {total}프레임")

    video_path = output_root / f"output{n}.wmv"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (crop_w, crop_h),
    )

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        fname = f"frame_{idx:05d}.png"
        crop, vis, smas_mask, _ = process_frame(frame)   # dermis는 모드 1에서 미사용
        cv2.imwrite(str(frames_dir / fname), crop)
        cv2.imwrite(str(masks_dir  / fname), smas_mask)
        cv2.imwrite(str(vis_dir    / fname), vis)
        writer.write(vis)
        print(f"[Frame] {idx}/{total}", end="\r")

    cap.release()
    writer.release()
    print(f"\n[완료] output{n} → {output_root}")
    print(f"  frames/: {idx}개  masks/: {idx}개  vis/: {idx}개  영상: {video_path.name}")


# ── 모드 2: 저장된 프레임으로 진피·multiclass GT 생성 ────────────────────────

def process_from_frames(n: int):
    """
    analog_result_24_v1/output{N}/frames/ 의 저장된 프레임을 이용해
    ROI 크롭 없이 재처리 → 진피 마스크 + multiclass GT 생성.

    SMAS 소스 우선순위:
        1. labeling_data/propagated_masks/output{N}/ (수동 라벨, 고품질)
        2. analog_detect 자동 탐지 결과 (fallback)

    저장:
        analog_result_24_v1/output{N}/dermis_masks/  ← 진피 binary (0/255)
        labeling_data/multiclass_masks/output{N}/    ← class-index map (0/1/2)
    """
    output_root  = _OUTPUT_DIR / f"output{n}"
    frames_dir   = output_root / "frames"

    if not frames_dir.exists():
        print(f"[Error] 프레임 폴더 없음: {frames_dir}")
        return

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        print(f"[Error] 프레임 파일 없음: {frames_dir}")
        return

    dermis_dir = output_root / "dermis_masks"
    multi_dir  = _MULTI_MASKS / f"output{n}"
    prop_dir   = _PROP_MASKS  / f"output{n}"
    dermis_dir.mkdir(parents=True, exist_ok=True)
    multi_dir.mkdir(parents=True, exist_ok=True)

    has_prop = prop_dir.exists() and any(prop_dir.glob("*.png"))
    smas_src = "propagated_masks" if has_prop else "auto-detect"
    print(f"[FromFrames] output{n}  프레임 {len(frame_files)}개  SMAS 소스: {smas_src}")

    for i, fp in enumerate(frame_files, 1):
        fname    = fp.name
        frame_bgr = cv2.imread(str(fp))
        if frame_bgr is None:
            print(f"  [Warning] 읽기 실패: {fp}")
            continue

        # ROI 없이 전체 프레임 처리
        _, _, smas_auto, dermis_mask = process_frame(frame_bgr, roi=None)

        # ── 진피 마스크 저장 ──
        cv2.imwrite(str(dermis_dir / fname), dermis_mask)

        # ── SMAS 결정: propagated 우선 ──
        prop_path = prop_dir / fname
        if has_prop and prop_path.exists():
            smas_mask = np.array(PILImage.open(prop_path).convert("L"))
        else:
            smas_mask = smas_auto

        # ── multiclass 병합: 0=배경, 1=진피, 2=SMAS (SMAS 우선순위 높음) ──
        h, w   = frame_bgr.shape[:2]
        merged = np.zeros((h, w), dtype=np.uint8)
        merged[dermis_mask > 127] = 1
        merged[smas_mask   > 127] = 2


        PILImage.fromarray(merged).save(str(multi_dir / fname))
        print(f"  [{i}/{len(frame_files)}]", end="\r")

    print(f"\n  dermis_masks/ → {dermis_dir}")
    print(f"  multiclass/   → {multi_dir}")


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def parse_num_list(tokens: list[str]) -> list[int]:
    """
    "55-57" → [55, 56, 57]
    "1"     → [1]
    혼합 가능: --num 1 2 55-57 → [1, 2, 55, 56, 57]
    """
    nums = []
    for token in tokens:
        if "-" in token:
            start, end = token.split("-", 1)
            nums.extend(range(int(start), int(end) + 1))
        else:
            nums.append(int(token))
    return nums


def main():
    parser = argparse.ArgumentParser(description="analog 탐지 영상 처리")
    parser.add_argument("--num", required=True, type=str, nargs="+",
                        help="처리할 영상 번호. 개별: --num 1 2 3 / 범위: --num 55-57 / 혼합: --num 1 55-57")
    parser.add_argument("--from_frames", action="store_true",
                        help="저장된 프레임을 이용해 진피 마스크 + multiclass GT 생성 (ROI 크롭 없음)")
    args = parser.parse_args()

    nums  = parse_num_list(args.num)
    total = len(nums)

    if args.from_frames:
        for i, n in enumerate(nums, 1):
            print(f"\n[{i}/{total}] output{n} 재처리 (from_frames)...")
            process_from_frames(n)
    else:
        for i, n in enumerate(nums, 1):
            print(f"\n[{i}/{total}] Image{n} 처리 중...")
            process_one(n)


if __name__ == "__main__":
    main()
