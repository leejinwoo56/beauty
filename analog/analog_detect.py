"""
analog_detect.py

MedSAM 없이 순수 아날로그(이미지 처리) 방법으로 초음파 피부층 분리.

층 구성 (위→아래):
  1층: 표피 (epidermis)         — 0 ~ L1
  2층: 진피/SMAS                — L1 ~ L2
  3층: 지방 (fat, 선택적)        — L2 ~ L3  (SMAS와 밝기 비슷하면 SMAS에 합산)
  4층: 뼈 (bone)                — L3 ~ 이미지 하단

경계선 (L1, L2, L3) 은 detect_all_lines 로 찾은 peak lines 중
  top / middle / bottom 3개를 선별한 것.

실행 예시
---------
  python analog_detect.py --input image.png --show
  python analog_detect.py --input image.png --roi 50 40 420 310 --show
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

# ─────────────────────────────────────────────────────────────────────────────
# 색상 팔레트 (BGR)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "epidermis":   (0,   0, 255),    # 빨강
    "smas":        (200, 50,  50),   # 파랑 (반투명 기본)
    "smas_bright": (0,  200, 255),   # 주황-노랑 (SMAS 내 밝은 줄기)
    "smas_thick":  (0,  200,   0),   # 초록 (SMAS 내 가장 굵은 줄기)
    "fat":         (180,  80, 255),  # 핑크
    "bone":        (200, 200, 200),  # 연회색
}
OVERLAY_ALPHA      = 0.45
SMAS_BRIGHT_ALPHA  = 0.70   # 밝은 줄기는 더 진하게
SMAS_THICK_ALPHA   = 0.85   # 굵은 줄기는 가장 진하게


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CLAHE 강조 + 가우시안 블러. (enh, smooth) 반환."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    smooth = cv2.GaussianBlur(enh, (1, 5), 0)
    return enh, smooth


def make_zone_mask(line_top: np.ndarray, line_bot: np.ndarray,
                   h: int, w: int) -> np.ndarray:
    """line_top[x] ≤ row < line_bot[x] 인 픽셀을 True 로."""
    rows = np.arange(h, dtype=np.int32)[:, None]          # (H, 1)
    lt = line_top.astype(np.int32)[None, :]                # (1, W)
    lb = line_bot.astype(np.int32)[None, :]
    return (rows >= lt) & (rows < lb)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 표피/진피 경계 - Region 으로 탐지
# ─────────────────────────────────────────────────────────────────────────────

def find_epidermis_boundary_region(
    enh: np.ndarray,
    smooth: np.ndarray,
    center_half_w: int = 100,
    max_y_extent: int = 25,
    bright_thr: float = 10.0,
) -> tuple[np.ndarray, int]:
    """
    중앙 ±center_half_w 열의 평균 밝기 프로파일에서 상승 지점을 탐지하고,
    해당 y 좌표를 seed 로 각 열에서 밝기가 비슷한 영역을 region grow 하여
    '경계 영역(boundary_mask)' 을 반환.

    표피/SMAS 경계는 초음파상 hyperechoic band 로 나타나므로
    밝기가 올라가는(양의 gradient) 첫 번째 지점을 사용한다.

    반환
    ----
    boundary_mask : (H, W) bool  — 경계 영역 픽셀
    boundary_max_y : int         — 경계 영역의 최하단 y 좌표
    """
    h, w = enh.shape
    cx = w // 2
    x0 = max(0, cx - center_half_w)
    x1 = min(w, cx + center_half_w)

    profile = enh[:, x0:x1].astype(np.float32).mean(axis=1)
    smoothed = gaussian_filter1d(profile, sigma=max(1.5, h * 0.012))

    # 밝기 양의 gradient (= 상승 지점) — 표피/SMAS 경계는 밝기가 올라가는 곳
    grad = np.gradient(smoothed)
    pos_grad = gaussian_filter1d(grad, sigma=max(1.5, h * 0.008))
    thr = max(0.5, float(np.std(pos_grad)) * 0.8)
    rises, _ = find_peaks(pos_grad, prominence=thr, distance=max(5, h // 30))

    if len(rises) == 0:
        return np.zeros((h, w), dtype=bool), 0

    # 첫 번째 상승 지점 = 표피/진피 경계 (밝기가 올라오는 hyperechoic band 시작)
    seed_y = int(rises[0])

    # ── 열별 독립 region growing ──────────────────────────────────────────────
    # seed_y 에서의 각 열 밝기를 기준으로 위/아래 방향으로
    # 밝기 차이 < bright_thr 인 구간 확장 (max_y_extent 이상은 확장 금지)
    boundary_mask = np.zeros((h, w), dtype=bool)

    for x in range(w):
        col = smooth[:, x].astype(np.float32)
        ref = float(col[min(seed_y, h - 1)])

        # 위 방향
        y_top = seed_y
        for y in range(seed_y, max(-1, seed_y - max_y_extent) - 1, -1):
            if 0 <= y < h and abs(col[y] - ref) <= bright_thr:
                y_top = y
            else:
                break

        # 아래 방향
        y_bot = seed_y
        for y in range(seed_y, min(h, seed_y + max_y_extent)):
            if abs(col[y] - ref) <= bright_thr:
                y_bot = y
            else:
                break

        # y 축 최대 높이 제한
        if y_bot - y_top > max_y_extent:
            y_bot = y_top + max_y_extent

        boundary_mask[y_top:y_bot + 1, x] = True

    # 형태학적 정리 (수평 연결)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    u8 = cv2.morphologyEx(boundary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kern)
    boundary_mask = u8.astype(bool)

    rows = np.where(boundary_mask.any(axis=1))[0]
    boundary_max_y = int(rows[-1]) if len(rows) > 0 else seed_y

    return boundary_mask, boundary_max_y


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: boundary 아래 모든 peak line 탐지 + propagation
# ─────────────────────────────────────────────────────────────────────────────

def _col_argmax(col: np.ndarray, center: float, half: int,
                y_min: int, y_max: int) -> int:
    y0 = max(y_min, int(center) - half)
    y1 = min(y_max, int(center) + half)
    if y1 <= y0:
        return int(center)
    return y0 + int(np.argmax(col[y0:y1 + 1]))


def _propagate_line(smooth: np.ndarray, pk_y: int,
                    w: int, h: int, mid_x: int,
                    max_step: int = 6,
                    init_band: int = 40,
                    smooth_frac: float = 0.15) -> np.ndarray:
    """단일 peak y 에서 좌우 propagation → (W,) int32."""
    raw = np.full(w, np.nan, dtype=np.float32)
    raw[mid_x] = _col_argmax(smooth[:, mid_x], float(pk_y), init_band, 0, h - 1)

    prev = raw[mid_x]
    for x in range(mid_x + 1, w):
        prev = _col_argmax(smooth[:, x], prev, max_step, 0, h - 1)
        raw[x] = prev

    prev = raw[mid_x]
    for x in range(mid_x - 1, -1, -1):
        prev = _col_argmax(smooth[:, x], prev, max_step, 0, h - 1)
        raw[x] = prev

    win = min(max(5, int(smooth_frac * w)) | 1, w - 2) | 1
    try:
        smoothed = savgol_filter(raw, window_length=win, polyorder=3)
    except ValueError:
        smoothed = raw

    return np.clip(np.round(smoothed), 0, h - 1).astype(np.int32)


def detect_all_peak_lines(
    enh: np.ndarray,
    smooth: np.ndarray,
    start_y: int,
    min_dist: int = 10,
    max_step: int = 6,
    init_band: int = 40,
    prominence: float = 0.0,
    smooth_frac: float = 0.15,
) -> list[np.ndarray]:
    """start_y 아래의 모든 brightness peak 을 탐지하고 라인으로 propagate."""
    h, w = enh.shape

    # 전체 열 평균 밝기 프로파일
    profile = enh.astype(np.float32).mean(axis=1)
    sm = gaussian_filter1d(profile, sigma=max(1.5, h * 0.01))
    sub = sm[start_y:]

    auto_prom = max(1.0, float(np.std(sub)) * 0.15)
    use_prom = prominence if prominence > 0.0 else auto_prom

    peaks, _ = find_peaks(sub, distance=min_dist, prominence=use_prom)
    peaks = peaks + start_y

    mid_x = w // 2
    return [
        _propagate_line(smooth, int(pk), w, h, mid_x, max_step, init_band, smooth_frac)
        for pk in peaks
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: L2, L3 선별 (L1 은 boundary_max_y 로 고정)
# ─────────────────────────────────────────────────────────────────────────────

def select_L2_L3(
    lines: list[np.ndarray],
    L1_y: int,
    w: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    detect_all_peak_lines 결과에서 L2, L3 를 선별.

    L1 은 boundary_max_y 로 이미 확정된 상태.
    - L3 = 탐지된 라인 중 y 평균이 가장 아래인 것
    - L2 = L1_y 와 L3_y 의 중간값에 가장 가까운 라인

    반환: (L2 line array or None, L3 line array or None)
    """
    if len(lines) == 0:
        return None, None

    sorted_lines = sorted(lines, key=lambda l: float(l.mean()))

    # L3: 가장 아래 라인
    L3 = sorted_lines[-1]
    L3_y = float(L3.mean())

    if len(sorted_lines) == 1:
        # 라인이 1개뿐이면 L3만 존재, L2 없음
        return None, L3

    # L2: L1_y ~ L3_y 중간에 가장 가까운 라인
    mid_target = (L1_y + L3_y) / 2.0
    # L3 제외한 나머지 후보
    candidates = sorted_lines[:-1]
    L2 = min(candidates, key=lambda l: abs(float(l.mean()) - mid_target))

    return L2, L3


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: 지방층 존재 여부 판별
# ─────────────────────────────────────────────────────────────────────────────

def check_fat_layer(
    enh: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
    L3: np.ndarray,
    bright_diff_thr: float = 15.0,
) -> bool:
    """
    Zone(L1~L2) = 2층, Zone(L2~L3) = 3층.
    평균 밝기 차이 ≥ bright_diff_thr → 지방층 존재.
    차이 < bright_diff_thr → 둘 다 SMAS (지방 없음).
    """
    h, w = enh.shape

    def zone_mean(la: np.ndarray, lb: np.ndarray) -> float:
        mask = make_zone_mask(la, lb, h, w)
        return float(enh[mask].mean()) if mask.any() else 0.0

    b2 = zone_mean(L1, L2)
    b3 = zone_mean(L2, L3)
    diff = abs(b2 - b3)
    has_fat = diff >= bright_diff_thr
    return has_fat


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: 층 마스크 생성 — 외곽층은 region growing, 중간층은 zone 직접 할당
# ─────────────────────────────────────────────────────────────────────────────

def _region_grow_in_zone(
    enh: np.ndarray,
    zone_mask: np.ndarray,
    seed_mask: np.ndarray,
    bright_thr: float = 12.0,
    max_iter: int = 20,
) -> np.ndarray:
    """
    seed_mask 에서 시작해 밝기가 비슷한 픽셀로 확장.
    zone_mask 경계를 벗어나지 않는다.
    """
    if not seed_mask.any():
        return zone_mask.copy()

    mask = (seed_mask & zone_mask).astype(bool)
    if not mask.any():
        return zone_mask.copy()

    ref = float(enh[mask].mean())
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_iter):
        dilated = cv2.dilate(mask.astype(np.uint8), kern).astype(bool) & zone_mask
        cands = dilated & ~mask
        new_px = cands & (np.abs(enh.astype(np.float32) - ref) <= bright_thr)
        if not new_px.any():
            break
        mask |= new_px
        ref = float(enh[mask].mean())

    return mask


def _extract_smas_bright(
    enh: np.ndarray,
    smas_zone: np.ndarray,
    bright_factor: float = 0.8,
    close_h: int = 15,   # 줄일수록 가로 연결 범위 좁아짐 → 더 sharp
    close_v: int = 2,    # 줄일수록 세로 번짐 없어짐 → 더 sharp
) -> np.ndarray:
    """
    SMAS zone 내에서 상대적으로 밝은 픽셀(줄기)만 추출.

    1. zone 내 픽셀의 평균 + bright_factor * 표준편차 를 임계값으로 설정
    2. 임계값 이상 픽셀 선택
    3. 수평 closing → 가로 방향 줄기 연결
    4. 전방향 closing → 세로 틈 메우기
    """
    if not smas_zone.any():
        return np.zeros(enh.shape, dtype=bool)

    pixels = enh[smas_zone].astype(np.float32)
    thr = float(pixels.mean()) + bright_factor * float(pixels.std())

    bright = (enh.astype(np.float32) >= thr) & smas_zone

    # 수평 closing — 가로 줄기 이어붙이기
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (close_h * 2 + 1, 3))
    u8 = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_CLOSE, h_kern)

    # 전방향 closing — 세로 틈 메우기
    e_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (close_v * 2 + 1, close_v * 2 + 1))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, e_kern)

    # zone 경계 클리핑
    return u8.astype(bool) & smas_zone


def _detect_smas_thick_stripe(
    enh: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
    h: int,
    w: int,
    half_win: int = 25,
    l1_margin: int = 60,
    bright_factor: float = 0.5,
) -> np.ndarray:
    """
    enh 원본 밝기 합으로 sliding window 탐색 → 가장 밝은 구간을 thick stripe 로 반환.

    L2 평균 y 에서 L1 평균 y 방향으로 1px 씩 올라가며
    가로 전체 × ±half_win px 윈도우 내 enh 픽셀 합이 가장 큰 y_center 선택.
    L1 에 붙어 있는 위치(y_top ≤ L1_y + l1_margin) 는 제외.

    결과: 선택된 window 내에서 enh > (window 평균 + bright_factor * std) 인 픽셀.
    (smas_bright_mask 여부 무관하게 밝은 픽셀 전부 포함)
    """
    L1_y = int(L1.mean())
    L2_y = int(L2.mean())
    enh_f = enh.astype(np.float32)

    # SMAS zone 자체가 없는 경우
    if L2_y <= L1_y:
        return np.zeros((h, w), dtype=bool)

    # 모든 후보 y_center 벡터 생성
    y_centers = np.arange(L1_y, L2_y + 1, dtype=np.int32)
    y_tops = np.maximum(0,     y_centers - half_win)
    y_bots = np.minimum(h - 1, y_centers + half_win)

    # l1_margin 필터
    valid = y_tops > L1_y + l1_margin

    if not valid.any():
        return np.zeros((h, w), dtype=bool)

    y_centers = y_centers[valid]
    y_tops    = y_tops[valid]
    y_bots    = y_bots[valid]

    # 행별 합 cumsum으로 sliding window 합 O(h*w) → O(h+w)
    row_sums = enh_f.sum(axis=1)                          # (h,)
    cumsum   = np.concatenate([[0.0], np.cumsum(row_sums)])  # (h+1,)
    win_sums = cumsum[y_bots + 1] - cumsum[y_tops]        # (n_valid,)

    best_idx = int(np.argmax(win_sums))
    best_y   = int(y_centers[best_idx])
    best_sum = float(win_sums[best_idx])

    y_top = max(0, best_y - half_win)
    y_bot = min(h - 1, best_y + half_win)

    # window 내 enh 밝기로 직접 threshold (smas_bright 무관)
    win_patch = enh_f[y_top:y_bot + 1, :]
    thr = float(win_patch.mean() + bright_factor * win_patch.std())

    result = np.zeros((h, w), dtype=bool)
    result[y_top:y_bot + 1, :] = win_patch >= thr

    # 가로 방향으로 closing → 끊긴 밝은 픽셀 연결
    ker_h = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, ker_h).astype(bool)
    # 세로 방향으로 closing → 줄기 매끄럽게
    ker_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, ker_v).astype(bool)

    return result


def build_layer_masks(
    enh: np.ndarray,
    boundary_mask: np.ndarray,
    key_lines: list[np.ndarray],
    has_fat: bool,
    grow_thr: float = 12.0,
) -> dict[str, np.ndarray]:
    """
    최종 층 마스크 딕셔너리 반환.

    key_lines = [L1, L2, L3]
    ┌─────────────┐  y = 0
    │  표피 Zone  │  0 ~ L1
    ├─────────────┤  L1
    │  Zone 1     │  L1 ~ L2   → SMAS
    ├─────────────┤  L2
    │  Zone 2     │  L2 ~ L3   → SMAS (has_fat=False) or 지방 (has_fat=True)
    ├─────────────┤  L3
    │  뼈 Zone    │  L3 ~ h-1
    └─────────────┘

    표피(1층)와 뼈(4층 or 3층): region growing 으로 다듬기
    중간층: line 사이 전체를 층으로 직접 할당
    """
    h, w = enh.shape

    if len(key_lines) < 3:
        # 라인 탐지 실패 fallback: 이미지를 3등분
        L1 = np.full(w, h // 3, dtype=np.int32)
        L2 = np.full(w, h * 2 // 3, dtype=np.int32)
        L3 = np.full(w, h - 1, dtype=np.int32)
    else:
        L1, L2, L3 = key_lines[0], key_lines[1], key_lines[2]

    top0  = np.zeros(w, dtype=np.int32)
    L1p1  = np.clip(L1 + 1, 0, h - 1)             # L1 바로 아래 행 (SMAS 시작)
    botH  = np.full(w, h, dtype=np.int32)          # h 포함 (make_zone_mask 는 < 이므로)

    zone_epid = make_zone_mask(top0, L1,   h, w)  # 0 ~ L1-1  (L1 행 미포함)
    zone1     = make_zone_mask(L1p1, L2,   h, w)  # L1+1 ~ L2 (SMAS, boundary 겹침 없음)
    zone2     = make_zone_mask(L2,   L3,   h, w)  # L2 ~ L3
    zone_bone = make_zone_mask(L3,   botH, h, w)  # L3 ~ 이미지 끝

    # ── 1층: 표피 — L1 위 전체 영역 직접 할당 ───────────────────────────────
    epid_mask = zone_epid.astype(bool)

    # ── 중간층 — line 사이 전체 직접 할당 ────────────────────────────────────
    if has_fat:
        smas_mask = zone1.copy()
        fat_mask  = zone2.copy()
    else:
        smas_mask = (zone1 | zone2).astype(bool)
        fat_mask  = np.zeros((h, w), dtype=bool)

    # ── SMAS 내 밝은 줄기 추출 ───────────────────────────────────────────────
    # has_fat=True  → smas = zone1 (L1~L2) 만 → zone1 에서 탐지
    # has_fat=False → smas = zone1+zone2 (L1~L3) 전체 → smas_mask 전체에서 탐지
    smas_bright_mask = _extract_smas_bright(enh, smas_mask.astype(bool),
                                             bright_factor=0.4)

    # ── SMAS 내 굵은 줄기 탐지 (L2 기준 박스 ROI) ───────────────────────────
    # L2 평균 y 에서 L1 방향으로 sliding window → 가장 빽빽한 구간 선택
    smas_thick_mask = _detect_smas_thick_stripe(
        enh, L1, L2, h, w)

    # ── 4층 (or 3층): 뼈 — L3 아래 전체 (L1, L2, L3 외 다른 선 없음) ──────────
    # region growing 으로 제한하지 않고 zone_bone 전체를 뼈로 직접 할당
    bone_mask = zone_bone.astype(bool)

    result: dict[str, np.ndarray] = {
        "epidermis":   epid_mask,
        "smas":        smas_mask,
        "smas_bright": smas_bright_mask,
        "smas_thick":  smas_thick_mask,
        "bone":        bone_mask,
    }
    if has_fat:
        result["fat"] = fat_mask

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def draw_lines_on(image_bgr: np.ndarray,
                  key_lines: list[np.ndarray]) -> np.ndarray:
    """L1, L2, L3 세 경계선만 그린다 (L1=흰, L2=노랑, L3=초록)."""
    out = image_bgr.copy()
    w = out.shape[1]
    xs = np.arange(w, dtype=np.int32)

    line_colors = [(255, 255, 255), (0, 220, 220), (100, 255, 100)]
    labels      = ["L1", "L2", "L3"]
    for i, line in enumerate(key_lines):
        pts = np.stack((xs, line), axis=1).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, line_colors[i], 1, cv2.LINE_AA)
        # 왼쪽 끝에 라벨 표시
        cv2.putText(out, labels[i], (4, int(line[0]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_colors[i], 1, cv2.LINE_AA)

    return out


def render_overlay(image_bgr: np.ndarray,
                   layer_masks: dict[str, np.ndarray],
                   alpha: float = OVERLAY_ALPHA,
                   smas_bright_alpha: float = SMAS_BRIGHT_ALPHA,
                   smas_thick_alpha: float = SMAS_THICK_ALPHA) -> np.ndarray:
    """층 마스크를 컬러 반투명 overlay 로 합성."""
    out = image_bgr.astype(np.float32)

    for name in ["epidermis", "smas", "fat", "bone"]:
        if name not in layer_masks:
            continue
        mask = layer_masks[name]
        if not np.any(mask):
            continue
        color = np.array(COLORS[name], dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color

    if "smas_bright" in layer_masks and np.any(layer_masks["smas_bright"]):
        mask = layer_masks["smas_bright"]
        color = np.array(COLORS["smas_bright"], dtype=np.float32)
        out[mask] = (1.0 - smas_bright_alpha) * out[mask] + smas_bright_alpha * color

    if "smas_thick" in layer_masks and np.any(layer_masks["smas_thick"]):
        mask = layer_masks["smas_thick"]
        color = np.array(COLORS["smas_thick"], dtype=np.float32)
        out[mask] = (1.0 - smas_thick_alpha) * out[mask] + smas_thick_alpha * color

    return out.astype(np.uint8)


def draw_labels(image_bgr: np.ndarray,
                layer_masks: dict[str, np.ndarray]) -> np.ndarray:
    out = image_bgr.copy()
    label_map = {
        "epidermis": "Epidermis",
        "smas":      "SMAS",
        "fat":       "Fat",
        "bone":      "Bone",
    }
    for name, mask in layer_masks.items():
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.putText(out, label_map.get(name, name),
                    (cx - 25, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLORS.get(name, (255, 255, 255)), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def run(
    image_bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    # epidermis boundary
    center_half_w: int   = 100,
    max_y_extent: int    = 25,
    epid_bright_thr: float = 10.0,
    # peak line detection
    peak_min_dist: int   = 10,
    peak_prominence: float = 0.0,
    # fat detection
    fat_diff_thr: float  = 0.0,
    # region growing
    grow_thr: float      = 12.0,
    # output
    show: bool           = False,
    output_path: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    파이프라인 실행.

    Parameters
    ----------
    image_bgr  : 원본 이미지 (BGR)
    roi        : (x1, y1, x2, y2) ROI 좌표. None 이면 전체 이미지 처리.

    Returns
    -------
    result     : overlay 합성된 원본 이미지 (크기 동일)
    layer_masks: {'epidermis', 'smas', 'fat'(선택), 'bone'} → ROI 기준 bool mask
    """
    h_full, w_full = image_bgr.shape[:2]

    # ── ROI 추출 ──────────────────────────────────────────────────────────────
    if roi is not None:
        x1r, y1r, x2r, y2r = roi
        x1r = max(0, x1r);  y1r = max(0, y1r)
        x2r = min(w_full, x2r);  y2r = min(h_full, y2r)
        roi_img = image_bgr[y1r:y2r, x1r:x2r]
    else:
        x1r, y1r = 0, 0
        roi_img = image_bgr

    h, w = roi_img.shape[:2]
    print(f"\n[Run] ROI 크기: {w}×{h}")

    # ── 전처리 ────────────────────────────────────────────────────────────────
    enh, smooth = preprocess(roi_img)

    # ── Step 3: 표피/진피 경계 영역 탐지 ─────────────────────────────────────
    boundary_mask, boundary_max_y = find_epidermis_boundary_region(
        enh, smooth, center_half_w, max_y_extent, epid_bright_thr)

    # ── Step 4: 경계 아래 모든 peak line 탐지 ─────────────────────────────────
    start_y = max(0, boundary_max_y + 1)
    all_lines = detect_all_peak_lines(
        enh, smooth, start_y,
        min_dist=peak_min_dist,
        prominence=peak_prominence,
    )

    # ── Step 5: L1 = boundary_mask 열별 하단 y (실제 경계선), L2/L3 선별 ────────
    # 각 열에서 boundary_mask 의 가장 아래 픽셀 y → 울퉁불퉁한 실제 경계선
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
        # peak line을 전혀 탐지 못한 경우 — 이미지 하단 2/3, 4/4 지점 fallback
        L2_line = np.full(w, h * 2 // 3, dtype=np.int32)
        L3_line = np.full(w, h * 3 // 4, dtype=np.int32)
        print("[Lines] fallback L2/L3 적용")
    elif L2_line is None:
        # 라인 1개뿐 → L3만 존재, L2를 L1~L3 중간으로 설정
        L2_line = ((L1.astype(np.float32) + L3_line.astype(np.float32)) / 2).astype(np.int32)
        print("[Lines] L2 중간값 fallback 적용")

    key_lines = [L1, L2_line, L3_line]

    # ── Step 6: 지방층 존재 여부 판별 ─────────────────────────────────────────
    has_fat = check_fat_layer(enh, L1, L2_line, L3_line, bright_diff_thr=fat_diff_thr)

    # ── Step 7: 층 마스크 생성 (region growing + zone fill) ───────────────────
    layer_masks = build_layer_masks(
        enh, boundary_mask, key_lines, has_fat, grow_thr=grow_thr)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    vis = draw_lines_on(roi_img, key_lines)
    vis = render_overlay(vis, layer_masks)

    # 원본 이미지에 ROI overlay 합성
    result = image_bgr.copy()
    if roi is not None:
        result[y1r:y2r, x1r:x2r] = vis
    else:
        result = vis

    if output_path is not None:
        cv2.imwrite(str(output_path), result)
        print(f"[Output] 저장: {output_path}")

    if show:
        WIN = "Analog Detect — Overlay"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        # 슬라이드바
        cv2.createTrackbar("Layer alpha %",       WIN, int(OVERLAY_ALPHA     * 100), 100, lambda _: None)
        cv2.createTrackbar("SMAS bright alpha %", WIN, int(SMAS_BRIGHT_ALPHA * 100), 100, lambda _: None)
        cv2.createTrackbar("SMAS thick alpha %",  WIN, int(SMAS_THICK_ALPHA  * 100), 100, lambda _: None)
        cv2.createTrackbar("Show lines (0/1)",    WIN, 1, 1, lambda _: None)

        print("[Show] ESC 또는 q 로 종료 / 슬라이드바로 투명도·라인 표시 조절")

        while True:
            a_layer    = cv2.getTrackbarPos("Layer alpha %",       WIN) / 100.0
            a_bright   = cv2.getTrackbarPos("SMAS bright alpha %", WIN) / 100.0
            a_thick    = cv2.getTrackbarPos("SMAS thick alpha %",  WIN) / 100.0
            show_lines = cv2.getTrackbarPos("Show lines (0/1)",    WIN)

            # 라인 표시 여부에 따라 base 이미지 선택
            base = draw_lines_on(roi_img, key_lines) if show_lines else roi_img.copy()

            vis_cur = render_overlay(base, layer_masks,
                                     alpha=a_layer, smas_bright_alpha=a_bright,
                                     smas_thick_alpha=a_thick)

            frame = image_bgr.copy()
            if roi is not None:
                frame[y1r:y2r, x1r:x2r] = vis_cur
            else:
                frame = vis_cur

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord('q'):   # ESC or q
                break

        cv2.destroyAllWindows()

    return result, layer_masks


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="아날로그 방식 피부층 분리 (MedSAM 불필요)")
    parser.add_argument("--input",  required=True,
                        help="입력 이미지 경로")
    parser.add_argument("--roi",    nargs=4, type=int,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="ROI 좌표 (생략 시 전체 이미지)")
    parser.add_argument("--output", default=None,
                        help="출력 이미지 경로 (기본: <stem>_analog.<ext>)")
    parser.add_argument("--show",   action="store_true",
                        help="결과 창 표시")

    # 알고리즘 파라미터
    parser.add_argument("--center-half-w",   type=int,   default=100,
                        help="표피 경계 탐지 시 중앙 열 범위 (기본 100)")
    parser.add_argument("--max-y-extent",    type=int,   default=25,
                        help="경계 영역 최대 y 높이 (기본 25)")
    parser.add_argument("--epid-bright-thr", type=float, default=10.0,
                        help="표피 경계 region growing 밝기 임계값 (기본 10)")
    parser.add_argument("--peak-min-dist",   type=int,   default=10,
                        help="peak 탐지 최소 거리 (기본 10)")
    parser.add_argument("--prominence",      type=float, default=0.0,
                        help="peak prominence (0 = 자동)")
    parser.add_argument("--fat-diff-thr",    type=float, default=0.0,
                        help="지방층 판별 밝기 차이 임계값 (기본 15)")
    parser.add_argument("--grow-thr",        type=float, default=12.0,
                        help="외곽층 region growing 밝기 임계값 (기본 12)")

    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"[Error] 이미지를 읽을 수 없습니다: {args.input}")
        return

    roi = tuple(args.roi) if args.roi else None

    if args.output is None:
        p = Path(args.input)
        out_path = str(p.parent / (p.stem + "_analog" + p.suffix))
    else:
        out_path = args.output

    run(
        img,
        roi=roi,
        center_half_w=args.center_half_w,
        max_y_extent=args.max_y_extent,
        epid_bright_thr=args.epid_bright_thr,
        peak_min_dist=args.peak_min_dist,
        peak_prominence=args.prominence,
        fat_diff_thr=args.fat_diff_thr,
        grow_thr=args.grow_thr,
        show=args.show,
        output_path=out_path,
    )


if __name__ == "__main__":
    main()
