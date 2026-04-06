"""
MedSAM 자동 층 분리 스크립트.

라인 탐지(표피/진피 빨간선 + peak 흰선) → 각 층에 슬라이딩 window box prompt
로 MedSAM 추론 → 마스크 합성 → 컬러 overlay 저장.

box prompt 규격
---------------
  width  : 이미지 너비 / n_windows  (--n-windows, 기본 15)
  stride : width / 2               (50 % 겹침)
  height : 해당 구간의 위/아래 라인 y좌표에 맞춰 동적으로 결정

실행 예시
---------
python medsam_layer_segment.py \\
    --input path/to/image.png \\
    --checkpoint medsam_vit_b.pth \\
    --n-windows 15 \\
    --min-dist-px 30 \\
    --prominence 5.0 \\
    --show
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from segment_anything import SamPredictor, sam_model_registry

INPUT_SIZE = 1024

# BGR 색상 팔레트 (층 순서대로)
LAYER_COLORS_BGR = [
    (60,   60,  200),   # 표피        빨강 계열
    (60,  180,  255),   # 진피        노랑 계열
    (160, 120,  255),   # layer3      분홍 계열
    (190, 220,  255),   # layer4      연살구
    (130, 210,  160),   # layer5      연두
    ( 50, 200,  255),   # layer6
    (255, 160,   60),   # layer7
    (200,  80,  200),   # bottom      보라
]


# ─────────────────────────────────────────────────────────────────────────────
# MedSAM 전처리 / 좌표 변환
# ─────────────────────────────────────────────────────────────────────────────

def medsam_preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """BGR → RGB min-max 정규화 → 1024x1024 (bicubic)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    lo, hi = image_rgb.min(), image_rgb.max()
    if hi > lo:
        image_rgb = (image_rgb - lo) / (hi - lo) * 255.0
    else:
        image_rgb = np.zeros_like(image_rgb)
    return cv2.resize(image_rgb.astype(np.uint8), (INPUT_SIZE, INPUT_SIZE),
                      interpolation=cv2.INTER_CUBIC)


def scale_box(box: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """표시 좌표 box → 1024x1024 기준 box."""
    x1, y1, x2, y2 = box
    sx = INPUT_SIZE / orig_w
    sy = INPUT_SIZE / orig_h
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 라인 탐지
# ─────────────────────────────────────────────────────────────────────────────

def _col_argmax(col: np.ndarray, center: float, half: int,
                y_min: int, y_max: int) -> int:
    y0 = max(y_min, int(center) - half)
    y1 = min(y_max, int(center) + half)
    if y1 <= y0:
        return int(center)
    return y0 + int(np.argmax(col[y0:y1 + 1].astype(np.float32)))


def _propagate_line(enh_smooth: np.ndarray, pk_y: int,
                    w: int, h: int, mid_x: int,
                    max_step: int, init_band: int, win: int) -> np.ndarray:
    """단일 peak y 좌우 propagation → (W,) int32."""
    raw = np.full(w, np.nan, dtype=np.float32)
    raw[mid_x] = _col_argmax(enh_smooth[:, mid_x], float(pk_y),
                              init_band, 0, h - 1)
    prev = raw[mid_x]
    for x in range(mid_x + 1, w):
        prev = _col_argmax(enh_smooth[:, x], prev, max_step, 0, h - 1)
        raw[x] = prev
    prev = raw[mid_x]
    for x in range(mid_x - 1, -1, -1):
        prev = _col_argmax(enh_smooth[:, x], prev, max_step, 0, h - 1)
        raw[x] = prev
    try:
        smoothed = savgol_filter(raw, window_length=win, polyorder=3)
    except ValueError:
        smoothed = raw
    return np.clip(np.round(smoothed), 0, h - 1).astype(np.int32)


def detect_skin_layer_lines(
    image_bgr: np.ndarray,
    center_half_width: int = 100,
    max_step: int = 6,
    init_band: int = 40,
    smooth_win_frac: float = 0.15,
) -> tuple[list[np.ndarray], int]:
    """표피/진피 경계선 탐지 (중앙 ±center_half_width px 평균 프로파일)."""
    h, w = image_bgr.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enh   = clahe.apply(gray)

    cx      = w // 2
    x_s     = max(0, cx - center_half_width)
    x_e     = min(w, cx + center_half_width)
    profile  = enh[:, x_s:x_e].astype(np.float32).mean(axis=1)
    smoothed = gaussian_filter1d(profile, sigma=max(1.5, h * 0.012))

    gradient  = np.gradient(smoothed)
    neg_grad  = gaussian_filter1d(-gradient, sigma=max(1.5, h * 0.008))
    thr       = max(0.5, float(np.std(neg_grad)) * 0.8)
    drops, _  = find_peaks(neg_grad, prominence=thr, distance=max(5, h // 30))
    print(f"[Skin] drop rows: {drops.tolist()}")

    enh_smooth = cv2.GaussianBlur(enh, (1, 5), 0)
    mid_x = w // 2
    win   = min(max(5, int(smooth_win_frac * w)) | 1, w - 2) | 1

    skin_lines: list[np.ndarray] = []
    skin_end_y = 0

    # drops[0] = 표피 하단 → 스킵, drops[1]부터 사용 (진피 하단부터)
    for dy in drops[1:3]:
        line = _propagate_line(enh_smooth, int(dy), w, h,
                               mid_x, max_step, init_band, win)
        skin_lines.append(line)
        skin_end_y = max(skin_end_y, int(dy))
        print(f"[Skin] 진피 경계 y={dy}")

    # 진피 내 추가 전환점 (최대 1개) — drops[1] 이후부터 탐색
    if len(drops) >= 2:
        ds    = int(drops[1]) + 5
        sub_p = smoothed[ds:]
        if len(sub_p) > 10:
            sub_ng  = -np.gradient(gaussian_filter1d(sub_p, sigma=max(1.5, h * 0.008)))
            sub_thr = max(0.5, float(np.std(sub_ng)) * 0.8)
            sub_d, _ = find_peaks(sub_ng, prominence=sub_thr,
                                   distance=max(5, h // 30))
            for sd in sub_d[:1]:
                abs_y = ds + int(sd)
                if abs_y > skin_end_y + 5:
                    line = _propagate_line(enh_smooth, abs_y, w, h,
                                           mid_x, max_step, init_band, win)
                    skin_lines.append(line)
                    skin_end_y = max(skin_end_y, abs_y)
                    print(f"[Skin] 진피 내 전환 y={abs_y}")

    return skin_lines, skin_end_y


def detect_all_peak_lines(
    image_bgr: np.ndarray,
    start_y: int = 0,
    min_dist_px: int = 10,
    max_step: int = 6,
    init_band: int = 40,
    smooth_win_frac: float = 0.15,
    prominence: float = 0.0,
) -> list[np.ndarray]:
    """start_y 아래의 모든 brightness peak 라인."""
    h, w = image_bgr.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enh   = clahe.apply(gray)

    profile     = enh.astype(np.float32).mean(axis=1)
    smoothed    = gaussian_filter1d(profile, sigma=max(1.5, h * 0.01))
    sub         = smoothed[start_y:]
    auto_prom   = max(1.0, float(np.std(sub)) * 0.15)
    use_prom    = prominence if prominence > 0.0 else auto_prom
    peaks, _    = find_peaks(sub, distance=min_dist_px, prominence=use_prom)
    peaks       = peaks + start_y
    print(f"[Peaks] prominence={use_prom:.2f}  {len(peaks)} peaks: {peaks.tolist()}")

    enh_smooth = cv2.GaussianBlur(enh, (1, 5), 0)
    mid_x = w // 2
    win   = min(max(5, int(smooth_win_frac * w)) | 1, w - 2) | 1

    return [
        _propagate_line(enh_smooth, int(pk_y), w, h, mid_x, max_step, init_band, win)
        for pk_y in peaks
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 경계 구성
# ─────────────────────────────────────────────────────────────────────────────

def build_boundaries(
    skin_lines: list[np.ndarray],
    peak_lines: list[np.ndarray],
    h: int,
    w: int,
) -> list[np.ndarray]:
    """
    이미지 상단(y=0) + skin_lines + peak_lines + 이미지 하단(y=h-1) 순서로 반환.
    인접한 두 경계 사이가 한 층(layer).
    """
    top    = np.zeros(w, dtype=np.int32)
    bottom = np.full(w, h - 1, dtype=np.int32)
    return [top] + list(skin_lines) + list(peak_lines) + [bottom]


# ─────────────────────────────────────────────────────────────────────────────
# 층 분리 (슬라이딩 window box prompt)
# ─────────────────────────────────────────────────────────────────────────────

def _make_region_mask(upper_line: np.ndarray, lower_line: np.ndarray,
                      h: int, w: int) -> np.ndarray:
    """upper_line[x] ≤ row ≤ lower_line[x] 인 픽셀을 True로."""
    rows  = np.arange(h, dtype=np.int32)[:, np.newaxis]   # (h, 1)
    upper = upper_line.astype(np.int32)[np.newaxis, :]     # (1, w)
    lower = lower_line.astype(np.int32)[np.newaxis, :]     # (1, w)
    return (rows >= upper) & (rows <= lower)


def segment_layer(
    predictor: SamPredictor,
    orig_h: int,
    orig_w: int,
    upper_line: np.ndarray,
    lower_line: np.ndarray,
    n_windows: int = 15,
    close_px: int = 5,
    layer_idx: int = 0,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    슬라이딩 window box prompt 로 한 층 segmentation.

    - box width  = orig_w // n_windows
    - stride     = box_width // 2  (50 % 겹침)
    - box height = 해당 x 범위의 upper/lower y 값에 맞춰 동적 결정

    반환: ((orig_h, orig_w) bool mask, 사용된 box 좌표 리스트 [(x1,y1,x2,y2), ...])
    """
    box_width = max(10, orig_w // n_windows)
    stride    = max(5,  box_width // 2)

    union   = np.zeros((orig_h, orig_w), dtype=np.uint8)
    n_boxes = 0
    used_boxes: list[tuple[int, int, int, int]] = []

    x = 0
    while True:
        x2 = min(x + box_width, orig_w)

        # 이 x 범위에서 위/아래 라인의 y 범위
        y1 = int(upper_line[x:x2].min())
        y2 = int(lower_line[x:x2].max())
        y1 = max(0, y1)
        y2 = min(orig_h - 1, y2)

        if y2 - y1 >= 2 and x2 - x >= 2:
            box_disp = np.array([x, y1, x2, y2], dtype=np.float32)
            box_1024 = scale_box(box_disp, orig_h, orig_w)

            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_1024,
                multimask_output=False,
            )
            mask_resized = cv2.resize(
                masks[0].astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            union |= mask_resized
            used_boxes.append((x, y1, x2, y2))
            n_boxes += 1
            print(f"    box {n_boxes:>2d}  x=[{x:>4d},{x2:>4d}]  "
                  f"y=[{y1:>4d},{y2:>4d}]  score={scores[0]:.3f}")

        if x2 >= orig_w:
            break
        x += stride

    print(f"  → {n_boxes} boxes  union={union.sum()} px")

    # ── closing: close_px=0 이면 스킵 (sharp), 클수록 넓게 연결 ─────────────
    if close_px > 0:
        kw       = close_px * 2 + 1
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        closed   = cv2.morphologyEx(union, cv2.MORPH_CLOSE, h_kernel)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        closed   = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, v_kernel)
    else:
        closed = union  # closing 없이 raw union 사용

    # ── 경계선(upper/lower) 바깥 클리핑 ─────────────────────────────────────
    region  = _make_region_mask(upper_line, lower_line, orig_h, orig_w)
    clipped = (closed.astype(bool) & region).astype(np.uint8)

    # ── 가장 큰 connected component 만 남기기 ────────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clipped, connectivity=8
    )
    if n_labels > 1:   # label 0 은 배경
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clipped = (labels == largest).astype(np.uint8)
        print(f"  largest component: {stats[largest, cv2.CC_STAT_AREA]} px "
              f"(of {n_labels - 1} components)")

    return clipped.astype(bool), used_boxes


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def draw_boxes_on(
    image_bgr: np.ndarray,
    all_layer_boxes: list[list[tuple[int, int, int, int]]],
) -> np.ndarray:
    """각 층의 box prompt를 층별 다른 색으로 그려서 반환."""
    # 층별 색상 (BGR, 반투명 느낌으로 밝은 색 사용)
    BOX_COLORS = [
        (100, 100, 255),   # 빨강 계열
        (100, 230, 255),   # 노랑 계열
        (230, 150, 255),   # 분홍 계열
        (230, 255, 200),   # 연두 계열
        (255, 200, 100),   # 하늘 계열
        (200, 100, 255),   # 보라 계열
        (100, 255, 200),   # 민트 계열
        (255, 150, 200),   # 핑크 계열
    ]
    out = image_bgr.copy()
    for layer_idx, boxes in enumerate(all_layer_boxes):
        color = BOX_COLORS[layer_idx % len(BOX_COLORS)]
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            # 층 번호 표시 (첫 번째 box에만)
        if boxes:
            bx1, by1, _, _ = boxes[0]
            cv2.putText(out, f"L{layer_idx + 1}", (bx1 + 2, by1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def draw_lines_on(image_bgr: np.ndarray,
                  skin_lines: list[np.ndarray],
                  peak_lines: list[np.ndarray]) -> np.ndarray:
    out = image_bgr.copy()
    w   = out.shape[1]
    xs  = np.arange(w, dtype=np.int32)
    for line in peak_lines:
        pts = np.stack((xs, line), axis=1).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
    for line in skin_lines:
        pts = np.stack((xs, line), axis=1).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def render_overlay(image_bgr: np.ndarray,
                   layer_masks: list[np.ndarray],
                   alpha: float = 0.5) -> np.ndarray:
    overlay = image_bgr.copy().astype(np.float32)
    for i, mask in enumerate(layer_masks):
        color = np.array(LAYER_COLORS_BGR[i % len(LAYER_COLORS_BGR)], dtype=np.float32)
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    return overlay.astype(np.uint8)


def render_class_map(layer_masks: list[np.ndarray],
                     h: int, w: int) -> np.ndarray:
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for i, mask in enumerate(layer_masks):
        color_map[mask] = LAYER_COLORS_BGR[i % len(LAYER_COLORS_BGR)]
    return color_map


# ─────────────────────────────────────────────────────────────────────────────
# 솔리드 색상 & 패턴 오버레이
# ─────────────────────────────────────────────────────────────────────────────

# 솔리드 색상 (BGR): layer1=빨강, layer2=파랑(SMAS), layer3=노랑(지방), layer4=흰색(뼈)
_SOLID_COLORS = [
    (0,   0, 255),    # layer1 — 빨강
    (200, 50,  50),   # layer2 — 파랑 (SMAS)
    (0, 255, 255),    # layer3 — 노랑 (지방)
    (255, 255, 255),  # layer4 — 흰색 (뼈)
]


def _make_stripe_pattern(h: int, w: int, stripe_px: int = 10) -> np.ndarray:
    """45도 대각선 파란색/흰색 줄무늬 패턴 (H×W×3, BGR)"""
    yy, xx = np.mgrid[:h, :w]
    idx = ((xx + yy) // stripe_px) % 2
    blue  = np.array([200, 50, 50],    dtype=np.uint8)
    white = np.array([255, 255, 255],  dtype=np.uint8)
    pattern = np.where(idx[:, :, None] == 0, blue, white).astype(np.uint8)
    return pattern


def _make_circle_pattern(h: int, w: int,
                         radius: int = 14, spacing: int = 18) -> np.ndarray:
    """주황색 원 격자 패턴 — 원끼리 겹쳐서 지방 질감 (H×W×3, BGR)"""
    pattern = np.zeros((h, w, 3), dtype=np.uint8)
    for cy in range(spacing // 2, h, spacing):
        for cx in range(spacing // 2, w, spacing):
            cv2.circle(pattern, (cx, cy), radius, (0, 140, 255), -1)  # 주황 BGR
    return pattern


def _make_bone_stripe_pattern(h: int, w: int,
                               stripe_px: int = 10) -> np.ndarray:
    """연회색 배경 + 밝은 수평 줄무늬 — 뼈 질감 (H×W×3, BGR)"""
    yy = np.arange(h, dtype=np.int32)[:, None] * np.ones((1, w), dtype=np.int32)
    # stripe_px 간격으로 밝은 줄, 나머지는 연회색
    on_stripe = (yy % (stripe_px * 2)) < stripe_px
    pattern = np.full((h, w, 3), 185, dtype=np.uint8)   # 연회색 베이스
    pattern[on_stripe] = (230, 230, 230)                  # 더 밝은 흰줄
    return pattern


def _sort_masks_by_y(masks: list[np.ndarray]) -> list[np.ndarray]:
    """비어있지 않은 마스크를 y 중심값 기준 오름차순(위→아래)으로 정렬."""
    pairs = []
    for m in masks:
        if m.any():
            pairs.append((float(np.where(m)[0].mean()), m))
    pairs.sort(key=lambda x: x[0])
    return [m for _, m in pairs]


_EPIDERMIS_MAX_BOTTOM_Y = 110   # 이 y 이하여야 표피로 인정


def _has_epidermis(ordered: list[np.ndarray]) -> bool:
    """정렬된 마스크 리스트에서 최상단 층이 표피인지 확인.
    최상단 층의 하단 y <= _EPIDERMIS_MAX_BOTTOM_Y 이면 표피."""
    if not ordered:
        return False
    ys = np.where(ordered[0])[0]
    return len(ys) > 0 and int(ys.max()) <= _EPIDERMIS_MAX_BOTTOM_Y


def render_solid_overlay(image_bgr: np.ndarray,
                         masks: list[np.ndarray],
                         alpha: float = 0.5) -> np.ndarray:
    """
    y축 위치(위→아래) + 표피 여부 기준으로 색상 배정:
      표피 있음: 표피(빨강) / SMAS(파랑) / [지방(노랑)] / 뼈(연회색+줄)
      표피 없음: SMAS(파랑) / [지방(노랑)] / 뼈(연회색+줄)
    """
    h, w = image_bgr.shape[:2]
    bone_pat = _make_bone_stripe_pattern(h, w)
    out = image_bgr.astype(np.float32)

    ordered = _sort_masks_by_y(masks)
    n = len(ordered)
    epid = _has_epidermis(ordered)

    for rank, m in enumerate(ordered):
        if rank == 0 and epid:                     # 표피 (빨강)
            c = np.array(_SOLID_COLORS[0], np.float32)
        elif rank == n - 1:                        # 최하단 = 뼈
            out[m] = (1 - alpha) * out[m] + alpha * bone_pat[m].astype(np.float32)
            continue
        elif epid and n == 4 and rank == 2:        # 표피+4층: 세 번째 = 지방 (노랑)
            c = np.array(_SOLID_COLORS[2], np.float32)
        else:                                      # SMAS (파랑)
            c = np.array(_SOLID_COLORS[1], np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * c
    return out.astype(np.uint8)


def render_pattern_overlay(image_bgr: np.ndarray,
                           masks: list[np.ndarray],
                           alpha: float = 0.5) -> np.ndarray:
    """
    y축 위치(위→아래) + 표피 여부 기준으로 패턴 배정:
      표피 있음: 표피(빨강) / SMAS(대각선줄) / [지방(원)] / 뼈(수평줄)
      표피 없음: SMAS(대각선줄) / [지방(원)] / 뼈(수평줄)
    """
    h, w = image_bgr.shape[:2]
    stripe   = _make_stripe_pattern(h, w)
    circles  = _make_circle_pattern(h, w)
    bone_pat = _make_bone_stripe_pattern(h, w)
    out = image_bgr.astype(np.float32)

    ordered = _sort_masks_by_y(masks)
    n = len(ordered)
    epid = _has_epidermis(ordered)

    for rank, m in enumerate(ordered):
        if rank == 0 and epid:                     # 표피 (빨강 solid)
            pat = None
            c   = np.array(_SOLID_COLORS[0], np.float32)
        elif rank == n - 1:                        # 최하단 = 뼈
            pat = bone_pat
        elif epid and n == 4 and rank == 2:        # 표피+4층: 세 번째 = 지방 (원)
            pat = circles
        else:                                      # SMAS (대각선 줄무늬)
            pat = stripe
            c   = None  # pat이 있으므로 c 불필요

        if pat is None:
            out[m] = (1 - alpha) * out[m] + alpha * c
        else:
            out[m] = (1 - alpha) * out[m] + alpha * pat[m].astype(np.float32)
    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 후처리: 잘못 분류된 영역 재할당 + 마스크 다듬기
# ─────────────────────────────────────────────────────────────────────────────

def _closest_layer(new_masks: list[np.ndarray], src: int,
                   y_center: int) -> int:
    """y_center 기준으로 src를 제외한 가장 가까운 층 인덱스 반환."""
    best_j, best_dist = -1, float('inf')
    for j, other in enumerate(new_masks):
        if j == src or not other.any():
            continue
        rows = np.where(other.any(axis=1))[0]
        y_min, y_max = int(rows[0]), int(rows[-1])
        dist = (0 if y_min <= y_center <= y_max
                else min(abs(y_center - y_min), abs(y_center - y_max)))
        if dist < best_dist:
            best_dist, best_j = dist, j
    return best_j


def _reassign_isolated_regions(masks: list[np.ndarray],
                                gap_ratio: float = 0.15,
                                min_gap_rows: int = 4) -> list[np.ndarray]:
    """
    전체 층 합산 행별 픽셀 수(all_counts)를 기반으로 gap 탐지.

    gap 기준을 '해당 층 픽셀 수'가 아닌 '모든 층의 총 픽셀 수'로 잡는 이유:
      - 대각선으로 내려가는 층은 자기 픽셀 수가 줄어도, 다른 층 픽셀이 그
        행을 채우고 있으므로 all_counts는 유지 → false positive 방지
      - 진짜 gap 구간은 어떤 층도 해당 행을 채우지 않아 all_counts 자체가
        급감 → 정확한 탐지 가능

    알고리즘:
      1. all_counts[y] = 현재 masks 전체 합산의 행별 픽셀 수
      2. peak_all = 해당 층의 y 범위 내 all_counts 최대값
      3. all_counts[y] < peak_all * gap_ratio → "thin 행"
      4. thin 행이 min_gap_rows 이상 연속 → gap 구간 확정
      5. gap 구간 픽셀 제거 + 아래 부분을 가장 가까운 층으로 재할당
    """
    new_masks = [m.copy().astype(bool) for m in masks]
    n_layers  = len(new_masks)

    for _pass in range(n_layers):
        # 현재 상태의 전체 합산 픽셀 수 (패스마다 갱신)
        all_counts = sum(m.astype(np.float32).sum(axis=1)
                         for m in new_masks)   # shape (H,)

        any_changed = False
        for i in range(n_layers):
            m = new_masks[i]
            if not m.any():
                continue

            # 이 층이 실제로 존재하는 y 범위
            layer_counts = m.sum(axis=1)
            active = np.where(layer_counts > 0)[0]
            if len(active) < 2:
                continue

            y_start, y_end = int(active[0]), int(active[-1])

            # gap 기준: 전체 층 합산 픽셀 수의 최댓값
            peak_all = float(all_counts[y_start:y_end + 1].max())
            if peak_all == 0:
                continue

            thin_thresh = peak_all * gap_ratio

            # top→bottom 으로 첫 번째 충분히 긴 thin 구간 탐색
            gap_start = None
            gap_end_y = None
            y = y_start
            while y <= y_end:
                if all_counts[y] <= thin_thresh:
                    if gap_start is None:
                        gap_start = y
                else:
                    if gap_start is not None:
                        run_len = y - gap_start
                        if run_len >= min_gap_rows:
                            gap_end_y = y
                            break
                        else:
                            gap_start = None   # 짧은 thin → 무시
                y += 1

            if gap_start is None or gap_end_y is None:
                continue

            # gap 구간 픽셀 제거
            new_masks[i][gap_start:gap_end_y] = False

            # gap 아래 부분 분리 후 재할당
            below_pixels = new_masks[i].copy()
            below_pixels[:gap_end_y] = False
            new_masks[i][gap_end_y:] = False

            if below_pixels.any():
                below_y_center = int(np.where(below_pixels.any(axis=1))[0].mean())
                best_j = _closest_layer(new_masks, i, below_y_center)
                if best_j >= 0:
                    new_masks[best_j] |= below_pixels
                    print(f"  [Reassign] layer{i+1} "
                          f"gap y=[{gap_start},{gap_end_y})  "
                          f"→ layer{best_j+1}  (below y≈{below_y_center})")

            any_changed = True

        if not any_changed:
            break

    return new_masks


def _smooth_layers(masks: list[np.ndarray],
                   close_h: int = 25,
                   close_v: int = 8,
                   open_px: int = 3) -> list[np.ndarray]:
    """
    층 마스크 다듬기:
      1. 수평 closing → 가로 틈새 연결 (층은 가로로 이어져야 함)
      2. 전방향 closing → 내부 구멍·세로 틈새 채우기
      3. opening → 작은 돌기·외부 잡음 제거
    """
    result = []
    for m in masks:
        u8 = m.astype(np.uint8)
        if not u8.any():
            result.append(m)
            continue

        # 수평 closing
        h_kern = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_h * 2 + 1, 3))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, h_kern)

        # 전방향 closing
        e_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_v * 2 + 1, close_v * 2 + 1))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, e_kern)

        # opening
        if open_px > 0:
            o_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (open_px * 2 + 1, open_px * 2 + 1))
            u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, o_kern)

        result.append(u8.astype(bool))
    return result


def _remove_small_blobs(masks: list[np.ndarray],
                        image_area: int,
                        min_ratio: float = 0.01) -> list[np.ndarray]:
    """
    이미지 전체 크기(image_area) 대비 min_ratio 미만인 층/컴포넌트 제거.

    - 층 전체 픽셀 수 < image_area * min_ratio → 층 통째로 소멸 (빈 마스크)
    - 층은 유지되지만 개별 컴포넌트가 작으면 해당 컴포넌트만 제거
    """
    layer_thresh = int(image_area * min_ratio)
    result = []
    for m in masks:
        u8 = m.astype(np.uint8)
        total = int(u8.sum())

        # 층 전체가 기준 미달 → 통째로 소멸
        if total < layer_thresh:
            print(f"    [blob] 층 소멸: {total} px < {layer_thresh} px ({min_ratio*100:.1f}% 미만)")
            result.append(np.zeros_like(m))
            continue

        # 층은 유지, 작은 컴포넌트만 제거
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            u8, connectivity=8)

        keep = np.zeros_like(u8)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= layer_thresh:
                keep[labels == lbl] = 1

        # 모두 제거되면 가장 큰 컴포넌트는 남김 (층 소멸은 위에서만)
        if keep.sum() == 0 and n_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep[labels == largest] = 1

        result.append(keep.astype(bool))
    return result


def _fill_and_extend_layers(masks: list[np.ndarray]) -> list[np.ndarray]:
    """
    층이 존재하는 행(row)을 이미지 가장자리까지 수평 연장.
      - 해당 층에 픽셀이 하나라도 있는 행 → 전체 행을 이 층으로 채움
      - 단, 다른 층이 이미 점유한 픽셀은 침범하지 않음
      - 결과적으로 각 층은 '가로 띠(band)' 형태에 가까워짐
    """
    result = [m.copy().astype(np.uint8) for m in masks]
    h, w = masks[0].shape

    for i in range(len(result)):
        if not result[i].any():
            continue

        # 다른 층 합산 점유 맵
        others = np.zeros((h, w), dtype=bool)
        for j in range(len(result)):
            if j != i:
                others |= result[j].astype(bool)

        # 이 층에 픽셀이 있는 행만 전체 채우기
        row_has = result[i].any(axis=1)          # (H,) bool
        filled = np.zeros((h, w), dtype=np.uint8)
        filled[row_has] = 1                       # 해당 행 전체를 1
        filled[others]  = 0                       # 다른 층 점유 제외

        result[i] = filled

    return [r.astype(bool) for r in result]


def _check_epidermis_boundary(masks: list[np.ndarray],
                               max_bottom_y: int = 110) -> list[np.ndarray]:
    """
    표피(y축 최상단 층)의 하단 y > max_bottom_y 이면 표피 미탐지로 판단.
    해당 마스크를 다음 층(SMAS)에 합산하고 표피는 빈 마스크로 소멸.
    """
    non_empty = sorted(
        [(i, m) for i, m in enumerate(masks) if m.any()],
        key=lambda x: float(np.where(x[1])[0].mean())
    )
    if not non_empty:
        return masks

    top_idx, top_mask = non_empty[0]
    bottom_y = int(np.where(top_mask)[0].max())

    if bottom_y <= max_bottom_y:
        return masks  # 정상 표피

    print(f"  [Epidermis] 하단 y={bottom_y} > {max_bottom_y} "
          f"→ 표피 미탐지, SMAS로 재분류")
    new_masks = [m.copy() for m in masks]
    new_masks[top_idx][:] = False

    if len(non_empty) >= 2:
        smas_idx, _ = non_empty[1]
        new_masks[smas_idx] |= top_mask

    return new_masks


def postprocess_layers(masks: list[np.ndarray],
                       orig_h: int, orig_w: int) -> list[np.ndarray]:
    """
    층 분류 후처리 파이프라인:
      1. isolated 컴포넌트 → y 기준 가까운 층으로 재할당
      2. 작은 blob/층 제거 (이미지 크기 대비 min_ratio 미만 → 소멸)
      3. 표피 하단 y > 110 → 표피 미탐지, SMAS로 재분류
      4. 수평 채우기/가장자리 연장 (행 단위로 이미지 끝까지)
      5. 마스크 smoothing (closing + opening)
    """
    image_area = orig_h * orig_w

    print("\n[PostProc] isolated 영역 재할당 중 ...")
    masks = _reassign_isolated_regions(masks)

    print("[PostProc] 작은 blob/층 제거 중 ...")
    masks = _remove_small_blobs(masks, image_area)

    print("[PostProc] 표피 경계 확인 중 ...")
    masks = _check_epidermis_boundary(masks)

    print("[PostProc] 수평 채우기 / 가장자리 연장 중 ...")
    masks = _fill_and_extend_layers(masks)

    print("[PostProc] 층 마스크 smoothing 중 ...")
    masks = _smooth_layers(masks)

    for i, m in enumerate(masks):
        state = f"{m.sum()} px" if m.any() else "소멸됨"
        print(f"  → layer{i + 1}: {state}")
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# CLI & main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedSAM 슬라이딩 window 층 분리")
    p.add_argument("--input",       required=True,  help="입력 이미지 경로")
    p.add_argument("--checkpoint",  required=True,  help="medsam_vit_b.pth")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--outdir",      default="outputs_layer")
    p.add_argument("--n-windows",   type=int,   default=15,
                   help="가로 방향 슬라이딩 window 수 (stride=width/2, default: 15)")
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="overlay 투명도 (default: 0.5)")
    p.add_argument("--min-dist-px", type=int,   default=10,
                   help="peak 간 최소 거리 px (default: 10)")
    p.add_argument("--max-step",    type=int,   default=6,
                   help="propagation 최대 y 이동 px (default: 6)")
    p.add_argument("--init-band",   type=int,   default=40,
                   help="중앙 초기 탐색 반경 px (default: 40)")
    p.add_argument("--prominence",  type=float, default=0.0,
                   help="peak prominence 최솟값 (0=자동, default: 0)")
    p.add_argument("--close-px",   type=int,   default=5,
                   help="마스크 연결 closing kernel 반경 px "
                        "(0=closing 없이 sharp, 클수록 넓게 연결, default: 5)")
    p.add_argument("--show",        action="store_true",
                   help="결과 이미지 화면 표시")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[Device] {device}")

    # ── MedSAM 로드 ──────────────────────────────────────────────────────────
    print(f"[MedSAM] Loading vit_b from {args.checkpoint} ...")
    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    # ── 이미지 로드 ──────────────────────────────────────────────────────────
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지 없음: {args.input}")
    orig_h, orig_w = image_bgr.shape[:2]
    print(f"[Image] {orig_w}x{orig_h}")

    # ── MedSAM 임베딩 (1회) ──────────────────────────────────────────────────
    image_1024_rgb = medsam_preprocess(image_bgr)
    print("[MedSAM] Computing embedding ...")
    predictor.set_image(image_1024_rgb)

    # ── 라인 탐지 ────────────────────────────────────────────────────────────
    print("\n[Skin] Detecting epidermis/dermis lines ...")
    skin_lines, skin_end_y = detect_skin_layer_lines(
        image_bgr,
        max_step=args.max_step,
        init_band=args.init_band,
    )
    print(f"[Skin] {len(skin_lines)} skin lines  end_y={skin_end_y}")

    print("\n[Peaks] Detecting peak lines ...")
    peak_lines = detect_all_peak_lines(
        image_bgr,
        start_y=max(0, skin_end_y + 5),
        min_dist_px=args.min_dist_px,
        max_step=args.max_step,
        init_band=args.init_band,
        prominence=args.prominence,
    )
    print(f"[Peaks] {len(peak_lines)} peak lines")

    # ── 경계 구성 ────────────────────────────────────────────────────────────
    boundaries = build_boundaries(skin_lines, peak_lines, orig_h, orig_w)
    n_layers   = len(boundaries) - 1
    print(f"\n[Seg] {n_layers} layers  (boundaries: {n_layers + 1})")
    for i, b in enumerate(boundaries):
        label = {0: "image top", len(boundaries) - 1: "image bottom"}.get(
            i, f"skin[{i-1}]" if i <= len(skin_lines) else f"peak[{i-1-len(skin_lines)}]"
        )
        print(f"  boundary {i}: {label}  mean_y={int(b.mean())}")

    # ── 맨 위 층 segmentation: y=0 ~ 첫 번째 라인 ───────────────────────────
    # 첫 번째 skin line이 없으면 peak_lines[0] 사용, 그것도 없으면 h//3 fallback
    if skin_lines:
        first_line = skin_lines[0]
    elif peak_lines:
        first_line = peak_lines[0]
    else:
        first_line = np.full(orig_w, orig_h // 3, dtype=np.int32)
        print("[Seg] 라인 미탐지 → fallback lower_line y=h//3")

    upper_line = np.zeros(orig_w, dtype=np.int32)  # y=0 고정
    lower_line = first_line

    print(f"\n[Seg] 맨 위 층  y=[0, mean={int(lower_line.mean())}]  "
          f"n_windows={args.n_windows}  close_px={args.close_px}")
    mask, boxes = segment_layer(
        predictor, orig_h, orig_w, upper_line, lower_line,
        n_windows=args.n_windows,
        close_px=args.close_px,
        layer_idx=1,
    )
    print(f"  final mask: {mask.sum()} px")

    all_layer_boxes = [boxes]

    # ── 라인 정리 ─────────────────────────────────────────────────────────────
    # all_lines: 위에서 아래 순서 (skin_lines + peak_lines)
    all_lines   = list(skin_lines) + list(peak_lines)
    second_line = all_lines[1] if len(all_lines) >= 2 else first_line
    last_line   = all_lines[-1] if len(all_lines) >= 2 else first_line
    bottom_line = np.full(orig_w, orig_h - 1, dtype=np.int32)

    # ── 층 2/3/4 region mask (SAM 없이) ──────────────────────────────────────
    # layer2 (흰색): first_line ~ second_line
    # layer3 (노랑): second_line ~ last_line  (중간 라인 무시, 통째로 한 층)
    # layer4 (연회색): last_line ~ 이미지 하단
    mask_layer2 = _make_region_mask(first_line,  second_line, orig_h, orig_w)
    mask_layer3 = _make_region_mask(second_line, last_line,   orig_h, orig_w)
    mask_layer4 = _make_region_mask(last_line,   bottom_line, orig_h, orig_w)

    print(f"\n[Lines]  first  mean_y={int(first_line.mean())}")
    print(f"[Lines]  second mean_y={int(second_line.mean())}")
    print(f"[Lines]  last   mean_y={int(last_line.mean())}")
    print(f"[Layers] layer1(SAM)={mask.sum()} px")
    print(f"[Layers] layer2     ={mask_layer2.sum()} px  (y {int(first_line.mean())} ~ {int(second_line.mean())})")
    print(f"[Layers] layer3     ={mask_layer3.sum()} px  (y {int(second_line.mean())} ~ {int(last_line.mean())})")
    print(f"[Layers] layer4     ={mask_layer4.sum()} px  (y {int(last_line.mean())} ~ {orig_h})")

    # ── 후처리 ───────────────────────────────────────────────────────────────
    mask, mask_layer2, mask_layer3, mask_layer4 = postprocess_layers(
        [mask, mask_layer2, mask_layer3, mask_layer4], orig_h, orig_w)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base   = Path(args.input).stem

    # 라인 이미지: first/second/last 만 표시 (중간 라인 표시 안 함)
    line_img = image_bgr.copy()
    xs = np.arange(orig_w, dtype=np.int32)
    for line, color, thickness in [
        (first_line,  (0,   0, 255), 2),   # 빨강
        (second_line, (255, 255, 255), 1),  # 흰색
        (last_line,   (255, 255, 255), 1),  # 흰색
    ]:
        pts = np.stack((xs, line), axis=1).reshape(-1, 1, 2)
        cv2.polylines(line_img, [pts], False, color, thickness, cv2.LINE_AA)

    # overlay: 솔리드 기본값으로 저장용 이미지 생성
    _layer_masks = [mask, mask_layer2, mask_layer3, mask_layer4]
    overlay = render_solid_overlay(image_bgr, _layer_masks, args.alpha)

    class_map = render_class_map([mask, mask_layer2, mask_layer3, mask_layer4],
                                 orig_h, orig_w)
    box_img   = draw_boxes_on(image_bgr, all_layer_boxes)

    cv2.imwrite(str(outdir / f"{base}_lines.png"),     line_img)
    cv2.imwrite(str(outdir / f"{base}_overlay.png"),   overlay)
    cv2.imwrite(str(outdir / f"{base}_class_map.png"), class_map)
    cv2.imwrite(str(outdir / f"{base}_boxes.png"),     box_img)

    print(f"\n[Saved] {outdir}/{base}_lines.png")
    print(f"[Saved] {outdir}/{base}_overlay.png")
    print(f"[Saved] {outdir}/{base}_class_map.png")
    print(f"[Saved] {outdir}/{base}_boxes.png")

    # 층별 통계
    print("\n[Layers]")
    for label, m in [("layer1(빨강/SAM)", mask),
                     ("layer2(파랑/SMAS)", mask_layer2),
                     ("layer3(노랑/지방)", mask_layer3),
                     ("layer4(흰색/뼈)",   mask_layer4)]:
        pct = 100.0 * m.sum() / (orig_h * orig_w)
        print(f"  {label:<20s}: {m.sum():>8d} px  ({pct:5.1f} %)")

    if args.show:
        WIN = "Layer Overlay  [P] Pattern ON/OFF  [Q/ESC] 종료"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Alpha %", WIN, int(args.alpha * 100), 100, lambda _: None)

        pattern_mode = [False]

        def _draw_y_ruler(img: np.ndarray, step: int = 100) -> np.ndarray:
            out = img.copy()
            h, w = out.shape[:2]
            for y in range(0, h, step):
                # 눈금선 (반투명 흰색)
                cv2.line(out, (0, y), (30, y), (255, 255, 255), 1, cv2.LINE_AA)
                # y값 텍스트
                cv2.putText(out, str(y), (2, max(y + 12, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 255, 255), 1, cv2.LINE_AA)
            return out

        while True:
            alpha_val = cv2.getTrackbarPos("Alpha %", WIN) / 100.0
            if pattern_mode[0]:
                disp = render_pattern_overlay(image_bgr, _layer_masks, alpha_val)
            else:
                disp = render_solid_overlay(image_bgr, _layer_masks, alpha_val)
            disp = _draw_y_ruler(disp)
            cv2.imshow(WIN, disp)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):   # Q 또는 ESC
                break
            elif key in (ord('p'), ord('P')):
                pattern_mode[0] = not pattern_mode[0]
                mode_str = "ON (패턴)" if pattern_mode[0] else "OFF (솔리드)"
                print(f"[Pattern] {mode_str}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
