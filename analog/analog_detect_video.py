"""
analog_detect_video.py
----------------------
지정한 번호의 영상을 읽어 analog 탐지를 수행하고 결과를 저장.

입력:  beauty/24/Image{N}.mp4
출력:  beauty/analog/analog_result_24_v1/output{N}/
          frames/frame_00001.png  ← 원본 프레임
          masks/ frame_00001.png  ← SMAS 마스크 (이진 PNG)
          vis/  frame_00001.png   ← 오버레이 프레임
          output{N}.mp4           ← 최종 오버레이 영상

사용법:
    python analog_detect_video.py --num 45
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
_HERE       = Path(__file__).resolve().parent          # analog/
_ROOT       = _HERE.parent                             # beauty/
_INPUT_DIR  = _ROOT / "24"                             # beauty/24/
_OUTPUT_DIR = _HERE / "analog_result_24_v1"            # beauty/analog/analog_result_24_v1/

LAYER_ALPHA       = 0.23
SMAS_BRIGHT_ALPHA = 0.25
SMAS_THICK_ALPHA  = 0.69
DEFAULT_ROI       = (50, 70, 1100, 700)


# ── 프레임 단위 처리 ───────────────────────────────────────────────────────────

def process_frame(frame_bgr, roi=DEFAULT_ROI):
    """
    프레임 하나를 ROI 크롭 후 처리.

    Returns:
        crop    : ROI 크롭된 원본 프레임 (BGR)
        vis     : ROI 크롭된 오버레이 이미지 (BGR)
        mask    : ROI 크롭된 SMAS 마스크 (uint8, 0 or 255)
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

    mask = layer_masks["smas"].astype(np.uint8) * 255

    return crop, vis, mask


# ── 메인 ───────────────────────────────────────────────────────────────────────

def process_one(n: int):
    """영상 번호 n 하나를 처리."""
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

    video_path = output_root / f"output{n}.mp4"
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
        crop, vis, mask = process_frame(frame)
        cv2.imwrite(str(frames_dir / fname), crop)
        cv2.imwrite(str(masks_dir  / fname), mask)
        cv2.imwrite(str(vis_dir    / fname), vis)
        writer.write(vis)
        print(f"[Frame] {idx}/{total}", end="\r")

    cap.release()
    writer.release()
    print(f"\n[완료] output{n} → {output_root}")
    print(f"  frames/: {idx}개  masks/: {idx}개  vis/: {idx}개  영상: {video_path.name}")


def main():
    parser = argparse.ArgumentParser(description="analog 탐지 영상 처리")
    parser.add_argument("--num", required=True, type=int, nargs="+",
                        help="처리할 영상 번호 (여러 개 가능: --num 1 2 3)")
    args = parser.parse_args()

    total = len(args.num)
    for i, n in enumerate(args.num, 1):
        print(f"\n[{i}/{total}] Image{n} 처리 중...")
        process_one(n)


if __name__ == "__main__":
    main()
