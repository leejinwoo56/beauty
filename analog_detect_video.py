import argparse
import cv2
from analog_detect import (
    preprocess,
    find_epidermis_boundary_region,
    detect_all_peak_lines,
    select_L2_L3,
    check_fat_layer,
    build_layer_masks,
    render_overlay,
)
import numpy as np

LAYER_ALPHA       = 0.23
SMAS_BRIGHT_ALPHA = 0.25
SMAS_THICK_ALPHA  = 0.69


def process_frame(frame_bgr, roi=None):
    h_full, w_full = frame_bgr.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_full, x2); y2 = min(h_full, y2)
        crop = frame_bgr[y1:y2, x1:x2]
    else:
        x1, y1 = 0, 0
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

    key_lines = [L1, L2_line, L3_line]

    has_fat = check_fat_layer(enh, L1, L2_line, L3_line, bright_diff_thr=0.0)

    layer_masks = build_layer_masks(enh, boundary_mask, key_lines, has_fat)

    vis = render_overlay(crop, layer_masks,
                         alpha=LAYER_ALPHA,
                         smas_bright_alpha=SMAS_BRIGHT_ALPHA,
                         smas_thick_alpha=SMAS_THICK_ALPHA)

    result = frame_bgr.copy()
    if roi is not None:
        result[y1:y2, x1:x2] = vis
    else:
        result = vis

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()

    roi = tuple(args.roi) if args.roi else None

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[Error] 영상을 열 수 없습니다: {args.input}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    from pathlib import Path
    frame_dir = Path(args.output).parent / "frame"
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Video] {width}×{height} @ {fps:.1f}fps, {total} frames")
    if roi:
        print(f"[ROI] x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}")
    print(f"[Frames] 저장 경로: {frame_dir}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        print(f"[Frame] {idx}/{total}", end="\r")
        out_frame = process_frame(frame, roi=roi)
        writer.write(out_frame)
        cv2.imwrite(str(frame_dir / f"{idx}.png"), out_frame)

    cap.release()
    writer.release()
    print(f"\n[Done] 저장: {args.output}")


if __name__ == "__main__":
    main()
