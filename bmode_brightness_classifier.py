import argparse
from pathlib import Path

import cv2
import numpy as np


CLASS_BG = 0
CLASS_1 = 1
CLASS_2 = 2
CLASS_3 = 3
CLASS_4 = 4
CLASS_5 = 5

CLASS_NAMES = {
    CLASS_1: "Class1",
    CLASS_2: "Class2",
    CLASS_3: "Class3",
    CLASS_4: "Class4",
    CLASS_5: "Class5",
}

CLASS_COLORS_BGR = {
    CLASS_1: (22, 166, 245),   # orange
    CLASS_2: (0, 195, 255),    # yellow
    CLASS_3: (167, 102, 255),  # pink
    CLASS_4: (190, 220, 255),  # light peach
    CLASS_5: (120, 210, 230),  # yellow-green
}

LINE_COLORS_BGR = [
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 180, 255),    # orange-ish
    (120, 255, 120),  # green
    (255, 140, 80),   # peach
]


def estimate_ultrasound_roi(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    roi = np.zeros_like(gray, dtype=np.uint8)

    y_start, y_end = 140, h - 180
    x_start, x_end = 0, w - 110

    y_start = int(np.clip(y_start, 0, h))
    y_end = int(np.clip(y_end, 0, h))
    x_start = int(np.clip(x_start, 0, w))
    x_end = int(np.clip(x_end, 0, w))
    if y_end > y_start and x_end > x_start:
        roi[y_start:y_end, x_start:x_end] = 1
    return roi


def smooth_curve(curve: np.ndarray, ksize: int = 31) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    pad = ksize // 2
    padded = np.pad(curve.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(ksize, dtype=np.float32) / float(ksize)
    smoothed = np.convolve(padded, kernel, mode="same")[pad:-pad]
    return smoothed


def _extract_line_candidates(
    column: np.ndarray,
    y_offset: int,
    min_line_distance: int,
    bright_quantile: float,
) -> list[int]:
    if column.size < 5:
        return []

    col = column.astype(np.float32)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    col_smooth = np.convolve(np.pad(col, (2, 2), mode="edge"), kernel, mode="valid")

    grad = np.abs(np.diff(col_smooth, prepend=col_smooth[0]))
    peaks = np.where((col_smooth[1:-1] >= col_smooth[:-2]) & (col_smooth[1:-1] >= col_smooth[2:]))[0] + 1
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(col_smooth))], dtype=np.int32)

    intensity_thr = float(np.percentile(col_smooth, np.clip(bright_quantile, 0.0, 100.0)))
    scored: list[tuple[float, int]] = []
    for p in peaks:
        if col_smooth[p] < intensity_thr:
            continue
        y = y_offset + int(p)
        score = float(col_smooth[p] + 0.35 * grad[p])
        scored.append((score, y))

    if not scored:
        order = np.argsort(col_smooth)[::-1]
        for p in order:
            y = y_offset + int(p)
            score = float(col_smooth[p] + 0.35 * grad[p])
            scored.append((score, y))

    scored.sort(key=lambda t: t[0], reverse=True)
    chosen: list[int] = []
    min_dist = max(1, int(min_line_distance))
    for _, y in scored:
        if all(abs(y - prev) >= min_dist for prev in chosen):
            chosen.append(int(y))
    return sorted(chosen)


def detect_line(
    gray: np.ndarray,
    roi: np.ndarray,
    min_line_distance: int = 10,
    max_jump_px: int = 12,
    smooth_ksize: int = 41,
    bright_quantile: float = 70.0,
) -> np.ndarray:
    h, w = gray.shape
    y_roi = np.where(roi > 0)[0]
    if y_roi.size == 0:
        return np.zeros((0, w), dtype=np.int32)

    y_min = int(y_roi.min())
    y_max = int(y_roi.max())
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates_per_col: list[list[int]] = []
    for x in range(w):
        ys = np.where(roi[:, x] > 0)[0]
        if ys.size == 0:
            candidates_per_col.append([])
            continue
        y0 = int(ys.min())
        y1 = int(ys.max())
        col = gray_blur[y0 : y1 + 1, x]
        candidates = _extract_line_candidates(
            col,
            y_offset=y0,
            min_line_distance=min_line_distance,
            bright_quantile=bright_quantile,
        )
        candidates_per_col.append(candidates)

    counts = np.array([len(cands) for cands in candidates_per_col if len(cands) > 0], dtype=np.int32)
    if counts.size == 0:
        default_y = int(np.median(y_roi))
        return np.full((1, w), default_y, dtype=np.int32)

    # Track every line candidate that appears in columns (no manual max cap).
    n_lines = max(1, int(counts.max()))

    tracked = np.full((n_lines, w), np.nan, dtype=np.float32)
    prev = np.full((n_lines,), np.nan, dtype=np.float32)
    jump_thr = float(max(1, int(max_jump_px)))

    for x in range(w):
        cands = np.array(candidates_per_col[x], dtype=np.float32)
        if cands.size == 0:
            tracked[:, x] = prev
            continue

        used = np.zeros(cands.size, dtype=bool)
        curr = np.full((n_lines,), np.nan, dtype=np.float32)

        for i in range(n_lines):
            if not np.isfinite(prev[i]):
                continue
            diffs = np.abs(cands - prev[i])
            diffs[used] = np.inf
            j = int(np.argmin(diffs))
            if np.isfinite(diffs[j]) and diffs[j] <= jump_thr:
                curr[i] = cands[j]
                used[j] = True
            else:
                # Reject abrupt jump and keep previous stable height.
                curr[i] = prev[i]

        remaining = np.sort(cands[~used])
        rem_idx = 0
        for i in range(n_lines):
            if np.isfinite(curr[i]):
                continue
            if np.isfinite(prev[i]):
                curr[i] = prev[i]
                continue
            if rem_idx < remaining.size:
                curr[i] = remaining[rem_idx]
                rem_idx += 1

        for i in range(1, n_lines):
            if np.isfinite(curr[i - 1]) and np.isfinite(curr[i]):
                if curr[i] <= curr[i - 1]:
                    curr[i] = curr[i - 1] + 1.0

        tracked[:, x] = curr
        prev = curr

    out = np.zeros((n_lines, w), dtype=np.int32)
    for i in range(n_lines):
        curve = tracked[i]
        finite = np.isfinite(curve)
        if not finite.any():
            fallback = int(y_min + (i + 1) * (y_max - y_min) / (n_lines + 1))
            out[i] = fallback
            continue

        valid_idx = np.where(finite)[0]
        filled = curve.copy()
        filled[: valid_idx[0]] = curve[valid_idx[0]]
        filled[valid_idx[-1] + 1 :] = curve[valid_idx[-1]]

        miss = np.where(~np.isfinite(filled))[0]
        if miss.size > 0:
            filled[miss] = np.interp(miss, valid_idx, curve[valid_idx])

        smoothed = smooth_curve(filled, ksize=smooth_ksize)
        for x in range(1, w):
            if abs(smoothed[x] - smoothed[x - 1]) > jump_thr:
                smoothed[x] = smoothed[x - 1]

        out[i] = np.clip(np.round(smoothed), y_min, y_max).astype(np.int32)
    return out


def build_class_map(
    gray: np.ndarray,
    min_line_distance: int,
    max_jump_px: int,
    line_smooth_ksize: int,
    line_bright_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    roi = estimate_ultrasound_roi(gray)
    if roi.sum() == 0:
        raise ValueError("ROI detection failed: no ultrasound area found.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    detected_lines = detect_line(
        clahe,
        roi,
        min_line_distance=min_line_distance,
        max_jump_px=max_jump_px,
        smooth_ksize=line_smooth_ksize,
        bright_quantile=line_bright_quantile,
    )

    ys, xs = np.indices(gray.shape)
    _, w = gray.shape

    valid_cols = np.where(roi.sum(axis=0) > 0)[0]
    x_min = int(valid_cols.min())
    x_max = int(valid_cols.max())

    y_roi = np.where(roi > 0)[0]
    y_min = int(y_roi.min())
    y_max = int(y_roi.max())
    depth = max(1, y_max - y_min)

    if detected_lines.shape[0] == 0:
        line_skin = np.full((w,), y_min + int(0.05 * depth), dtype=np.float32)
        line_bone = np.full((w,), y_min + int(0.75 * depth), dtype=np.float32)
    else:
        line_skin = detected_lines[0].astype(np.float32)
        if detected_lines.shape[0] >= 2:
            line_bone = detected_lines[-1].astype(np.float32)
        else:
            line_bone = line_skin + float(0.65 * depth)

    line_skin = np.clip(line_skin, y_min, y_max - 1)
    line_bone = np.clip(line_bone, y_min + int(0.4 * depth), y_max)
    line_bone = np.maximum(line_bone, line_skin + int(0.35 * depth))
    line_bone = np.clip(line_bone, y_min + 1, y_max)

    x_to_skin = line_skin[np.clip(xs, 0, w - 1)]
    x_to_bone = line_bone[np.clip(xs, 0, w - 1)]

    class_map = np.zeros_like(gray, dtype=np.uint8)
    roi_mask = roi > 0
    core_cols = (xs >= x_min) & (xs <= x_max)
    valid = roi_mask & core_cols

    column_span = np.maximum(1.0, x_to_bone - x_to_skin)
    rel_depth = (ys - x_to_skin) / column_span

    class_map[valid & (rel_depth < 0.0)] = CLASS_1
    class_map[valid & (rel_depth >= 0.0) & (rel_depth < 0.15)] = CLASS_1
    class_map[valid & (rel_depth >= 0.15) & (rel_depth < 0.35)] = CLASS_2
    class_map[valid & (rel_depth >= 0.35) & (rel_depth < 0.60)] = CLASS_3
    class_map[valid & (rel_depth >= 0.60) & (rel_depth < 1.00)] = CLASS_4
    class_map[valid & (rel_depth >= 1.00)] = CLASS_5

    return (
        class_map,
        roi,
        line_skin.astype(np.int32),
        line_bone.astype(np.int32),
        detected_lines.astype(np.int32),
    )


def render_overlay(image_bgr: np.ndarray, class_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_bgr.copy().astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    for class_id, color in CLASS_COLORS_BGR.items():
        mask = class_map == class_id
        if np.any(mask):
            overlay[mask] = (1.0 - a) * overlay[mask] + a * np.array(color, dtype=np.float32)
    return overlay.astype(np.uint8)


def render_color_mask(class_map: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS_BGR.items():
        color_mask[class_map == class_id] = color
    return color_mask


def put_class_labels(image: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    out = image.copy()
    for class_id, name in CLASS_NAMES.items():
        ys, xs = np.where(class_map == class_id)
        if xs.size == 0:
            continue
        cx = int(np.median(xs))
        cy = int(np.median(ys))
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x1, y1 = cx - tw // 2 - 8, cy - th - 10
        x2, y2 = cx + tw // 2 + 8, cy + 8
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(out.shape[1] - 1, x2)
        y2 = min(out.shape[0] - 1, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 60, 60), -1)
        cv2.putText(out, name, (x1 + 8, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def render_detected_lines(
    image_bgr: np.ndarray,
    detected_lines: np.ndarray,
    roi: np.ndarray,
    skin_line: np.ndarray,
    bone_line: np.ndarray,
) -> np.ndarray:
    vis = image_bgr.copy()
    valid_cols = np.where(roi.sum(axis=0) > 0)[0]
    if valid_cols.size == 0:
        return vis

    for i in range(detected_lines.shape[0]):
        color = LINE_COLORS_BGR[i % len(LINE_COLORS_BGR)]
        pts = np.stack((valid_cols, detected_lines[i, valid_cols]), axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, color, 1, cv2.LINE_AA)
        label_pos = (int(valid_cols[0]) + 6, int(detected_lines[i, valid_cols[0]]) - 4)
        cv2.putText(vis, f"L{i + 1}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    pts_skin = np.stack((valid_cols, skin_line[valid_cols]), axis=1).astype(np.int32).reshape(-1, 1, 2)
    pts_bone = np.stack((valid_cols, bone_line[valid_cols]), axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts_skin], False, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(vis, [pts_bone], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Top boundary", (int(valid_cols[0]) + 6, int(skin_line[valid_cols[0]]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "Bottom boundary", (int(valid_cols[0]) + 6, int(bone_line[valid_cols[0]]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brightness-filter based B-mode class segmentation with multi-line detection.")
    parser.add_argument("--input", type=str, required=True, help="Path to B-mode image.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save results.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha (0~1).")
    parser.add_argument("--min-line-distance", type=int, default=10, help="Minimum vertical distance between detected line candidates.")
    parser.add_argument("--max-line-jump", type=int, default=12, help="Reject candidate if height jump from previous column is larger than this value.")
    parser.add_argument("--line-smooth-ksize", type=int, default=41, help="Smoothing kernel size for detected lines.")
    parser.add_argument("--line-bright-quantile", type=float, default=70.0, help="Brightness quantile threshold for selecting line candidates.")
    parser.add_argument("--show", action="store_true", help="Show preview windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    class_map, roi, skin_line, bone_line, detected_lines = build_class_map(
        gray,
        min_line_distance=max(1, args.min_line_distance),
        max_jump_px=max(1, args.max_line_jump),
        line_smooth_ksize=max(3, args.line_smooth_ksize),
        line_bright_quantile=float(args.line_bright_quantile),
    )

    overlay = render_overlay(image_bgr, class_map, alpha=args.alpha)
    overlay_labeled = put_class_labels(overlay, class_map)
    mask_color = render_color_mask(class_map)
    debug_line = render_detected_lines(image_bgr, detected_lines, roi, skin_line, bone_line)

    base = input_path.stem
    p_overlay = outdir / f"{base}_overlay.png"
    p_overlay_label = outdir / f"{base}_overlay_labeled.png"
    p_mask = outdir / f"{base}_class_mask.png"
    p_roi = outdir / f"{base}_roi.png"
    p_debug = outdir / f"{base}_lines.png"
    p_ids = outdir / f"{base}_class_ids.npy"
    p_lines = outdir / f"{base}_detected_lines.npy"

    cv2.imwrite(str(p_overlay), overlay)
    cv2.imwrite(str(p_overlay_label), overlay_labeled)
    cv2.imwrite(str(p_mask), mask_color)
    cv2.imwrite(str(p_roi), roi * 255)
    cv2.imwrite(str(p_debug), debug_line)
    np.save(str(p_ids), class_map)
    np.save(str(p_lines), detected_lines)

    print(f"[Saved] {p_overlay}")
    print(f"[Saved] {p_overlay_label}")
    print(f"[Saved] {p_mask}")
    print(f"[Saved] {p_roi}")
    print(f"[Saved] {p_debug}")
    print(f"[Saved] {p_ids}")
    print(f"[Saved] {p_lines}")
    print("\nDetected lines:")
    for i in range(detected_lines.shape[0]):
        print(f"  Line {i + 1}")
    print("\nClass IDs:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name}")

    if args.show:
        cv2.imshow("Input", image_bgr)
        cv2.imshow("Class Mask", mask_color)
        cv2.imshow("Overlay", overlay_labeled)
        cv2.imshow("Detected Lines", debug_line)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
