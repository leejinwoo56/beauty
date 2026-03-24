import argparse
import json
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


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    bool/uint8 2D 마스크를 Fortran-order RLE로 인코딩.
    pycocotools 없이 동작하는 경량 구현.
    반환: {"size": [H, W], "counts": "..."}
    """
    h, w = mask.shape
    flat = mask.astype(np.uint8).flatten(order="F")  # column-major
    counts = []
    current = 0
    run = 0
    for v in flat:
        if v == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = v
    counts.append(run)
    # RLE는 항상 0(배경)부터 시작
    if flat[0] == 1:
        counts.insert(0, 0)
    return {"size": [h, w], "counts": counts}


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    """마스크에서 bounding box [x, y, w, h] 계산."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def build_smas_annotation(layer_masks: dict, crop_shape: tuple, file_name: str) -> dict:
    """
    smas_thick 마스크만 사용해 SA-1B 스타일 JSON annotation dict를 반환.
    """
    h, w = crop_shape[:2]

    smas_mask = layer_masks.get("smas_thick", np.zeros((h, w), dtype=bool)).astype(bool)

    area = int(smas_mask.sum())
    bbox = mask_to_bbox(smas_mask)
    rle  = mask_to_rle(smas_mask)

    return {
        "image": {
            "file_name": file_name,
            "height": h,
            "width":  w,
        },
        "annotations": [
            {
                "category": "smas",
                "segmentation": rle,
                "bbox": bbox,
                "area": area,
            }
        ],
    }


def process_frame(frame_bgr, roi=None):
    """
    반환: (overlay_full, crop_original, layer_masks)
      - overlay_full   : 원본 해상도 위에 overlay 합성된 시각화 이미지
      - crop_original  : ROI 크롭된 원본 이미지 (overlay 없음, SAM 학습 입력)
      - layer_masks    : build_layer_masks() 반환 dict
    """
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

    has_fat = check_fat_layer(enh, L1, L2_line, L3_line, bright_diff_thr=10000.0)

    layer_masks = build_layer_masks(enh, boundary_mask, key_lines, has_fat)

    vis = render_overlay(crop, layer_masks,
                         alpha=LAYER_ALPHA,
                         smas_bright_alpha=SMAS_BRIGHT_ALPHA,
                         smas_thick_alpha=SMAS_THICK_ALPHA)

    overlay_full = frame_bgr.copy()
    if roi is not None:
        overlay_full[y1:y2, x1:x2] = vis
    else:
        overlay_full = vis

    crop_original = crop.copy()

    return overlay_full, crop_original, layer_masks


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
    base_dir  = Path(args.output).parent
    vis_dir   = base_dir / "vis"       # 선별용 시각화
    frame_dir = base_dir / "frames"    # SAM 학습 입력 (원본 크롭)
    mask_dir  = base_dir / "masks"     # SAM 학습 레이블 (JSON)

    for d in (vis_dir, frame_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Video] {width}×{height} @ {fps:.1f}fps, {total} frames")
    if roi:
        print(f"[ROI] x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}")
    print(f"[vis]    → {vis_dir}")
    print(f"[frames] → {frame_dir}")
    print(f"[masks]  → {mask_dir}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        print(f"[Frame] {idx}/{total}", end="\r")

        name = f"frame_{idx:05d}"

        overlay_full, crop_original, layer_masks = process_frame(frame, roi=roi)

        # 동영상 출력
        writer.write(overlay_full)

        # 선별용 시각화 이미지
        cv2.imwrite(str(vis_dir / f"{name}.png"), overlay_full)

        # SAM 학습 입력: 원본 크롭 이미지
        cv2.imwrite(str(frame_dir / f"{name}.png"), crop_original)

        # SAM 학습 레이블: SMAS 마스크 JSON
        annotation = build_smas_annotation(layer_masks, crop_original.shape, f"{name}.png")
        with open(mask_dir / f"{name}.json", "w") as f:
            json.dump(annotation, f)

    cap.release()
    writer.release()
    print(f"\n[Done] 저장: {args.output}")
    print(f"       총 {idx}개 프레임 → vis / frames / masks 폴더 확인")


if __name__ == "__main__":
    main()
