"""
regen_masks.py

선별된 frames/ PNG를 읽어 detecting 후 smas_thick 마스크만
masks/ JSON으로 재생성. vis/, frames/ 는 건드리지 않음.

사용법:
    python regen_masks.py --data_root output_segmentation_video
    python regen_masks.py --data_root output_segmentation_video --dry-run
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analog_detect import (
    preprocess,
    find_epidermis_boundary_region,
    detect_all_peak_lines,
    select_L2_L3,
    check_fat_layer,
    build_layer_masks,
)
from analog_detect_video import mask_to_rle, mask_to_bbox

EXCLUDE_RESULTS = {6, 7, 9, 16, 17, 19}


def detect_smas_thick(frame_bgr: np.ndarray) -> np.ndarray:
    """이미지에서 smas_thick 마스크(bool) 반환."""
    h, w = frame_bgr.shape[:2]
    enh, smooth = preprocess(frame_bgr)

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

    return layer_masks.get("smas_thick", np.zeros((h, w), dtype=bool)).astype(bool)


def build_annotation(mask: np.ndarray, file_name: str) -> dict:
    h, w = mask.shape
    return {
        "image": {
            "file_name": file_name,
            "height": h,
            "width":  w,
        },
        "annotations": [
            {
                "category": "smas_thick",
                "segmentation": mask_to_rle(mask),
                "bbox": mask_to_bbox(mask),
                "area": int(mask.sum()),
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="output_segmentation_video")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 저장 없이 처리 대상만 출력")
    args = parser.parse_args()

    base = Path(args.data_root)
    result_dirs = sorted(base.glob("result*"), key=lambda p: int(p.name.replace("result", "")))

    total_ok = 0
    total_empty = 0

    for result_dir in result_dirs:
        try:
            num = int(result_dir.name.replace("result", ""))
        except ValueError:
            continue
        if num in EXCLUDE_RESULTS:
            continue

        frame_dir = result_dir / "frames"
        mask_dir  = result_dir / "masks"

        if not frame_dir.exists():
            continue

        mask_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(frame_dir.glob("*.png"))
        print(f"[{result_dir.name}] {len(frames)}개 프레임 처리 중...")

        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"  [warn] 읽기 실패: {frame_path.name}")
                continue

            mask = detect_smas_thick(img)
            ann  = build_annotation(mask, frame_path.name)

            json_path = mask_dir / (frame_path.stem + ".json")

            if args.dry_run:
                status = "empty" if mask.sum() == 0 else "ok"
                print(f"  [dry] {frame_path.name} → {json_path.name} ({status})")
            else:
                with open(json_path, "w") as f:
                    json.dump(ann, f)

            if mask.sum() == 0:
                total_empty += 1
            else:
                total_ok += 1

        print(f"  → 완료 (검출 성공: {total_ok}, 빈 마스크: {total_empty})")
        total_ok = 0
        total_empty = 0

    print(f"\n{'[dry-run 완료]' if args.dry_run else '[전체 완료]'}")


if __name__ == "__main__":
    main()