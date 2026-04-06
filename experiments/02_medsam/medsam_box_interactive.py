"""
MedSAM 실시간 인터랙티브 box segmentation + 밝기 peak 라인 표시.

논문 스펙 (Nature Communications, 2024)
----------------------------------------
- 모델    : SAM ViT-B (MedSAM fine-tuned)
- 입력    : 1024 x 1024
- 프롬프트: Bounding box (논문 권장)
- 출력    : multimask_output=False (단일 최적 마스크)
- 전처리  : min-max 정규화 → [0,255] uint8 → 1024x1024 리사이즈

체크포인트 다운로드
-------------------
https://drive.google.com/file/d/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_/view
파일명: medsam_vit_b.pth

조작법
------
  마우스 드래그 : box 그리기 → 손 떼면 즉시 segmentation
  R             : 마스크 초기화 (peak 라인은 유지)
  Q / ESC       : 종료
"""

import argparse

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from segment_anything import SamPredictor, sam_model_registry

INPUT_SIZE = 1024

MASK_COLORS = [
    (0,   255, 255),
    (255,   0, 255),
    (0,   255,   0),
    (255, 128,   0),
    (128,   0, 255),
]


# ---------------------------------------------------------------------------
# MedSAM 전처리 (논문 공식 방법)
# ---------------------------------------------------------------------------

def medsam_preprocess(image_bgr: np.ndarray, apply_clahe: float = 0.0) -> np.ndarray:
    """
    MedSAM 논문 공식 전처리:
      1. BGR → RGB
      2. min-max 정규화 → [0, 255] uint8
      3. (선택) CLAHE 적용 — 채널별로 각각 적용
      4. 1024x1024 리사이즈 (bicubic)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_min, img_max = image_rgb.min(), image_rgb.max()
    if img_max > img_min:
        image_rgb = (image_rgb - img_min) / (img_max - img_min) * 255.0
    else:
        image_rgb = np.zeros_like(image_rgb)
    image_u8 = image_rgb.astype(np.uint8)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=apply_clahe, tileGridSize=(8, 8))
        image_u8 = np.stack([clahe.apply(image_u8[:, :, c]) for c in range(3)], axis=2)

    image_1024 = cv2.resize(image_u8, (INPUT_SIZE, INPUT_SIZE),
                            interpolation=cv2.INTER_CUBIC)
    return image_1024   # RGB uint8, 1024x1024


def scale_box(box: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """표시 좌표 box → 1024x1024 기준 좌표로 변환."""
    x1, y1, x2, y2 = box
    sx = INPUT_SIZE / orig_w
    sy = INPUT_SIZE / orig_h
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


# ---------------------------------------------------------------------------
# 밝기 peak 감지 + 좌우 propagation (모든 peak)
# ---------------------------------------------------------------------------

def _col_argmax(col: np.ndarray, center: float, half: int,
                y_min: int, y_max: int) -> int:
    y0 = max(y_min, int(center) - half)
    y1 = min(y_max, int(center) + half)
    if y1 <= y0:
        return int(center)
    return y0 + int(np.argmax(col[y0:y1 + 1].astype(np.float32)))


def _propagate_line(
    enh_smooth: np.ndarray,
    pk_y: int,
    w: int,
    h: int,
    mid_x: int,
    max_step: int,
    init_band: int,
    win: int,
) -> np.ndarray:
    """단일 peak y 로부터 좌우 propagation → (W,) int32 라인."""
    raw = np.full(w, np.nan, dtype=np.float32)
    raw[mid_x] = _col_argmax(enh_smooth[:, mid_x], float(pk_y), init_band, 0, h - 1)
    prev = raw[mid_x]
    for x in range(mid_x + 1, w):
        prev = _col_argmax(enh_smooth[:, x], prev, max_step, 0, h - 1)
        raw[x] = prev
    prev = raw[mid_x]
    for x in range(mid_x - 1, -1, -1):
        prev = _col_argmax(enh_smooth[:, x], prev, max_step, 0, h - 1)
        raw[x] = prev
    try:
        smoothed_line = savgol_filter(raw, window_length=win, polyorder=3)
    except ValueError:
        smoothed_line = raw
    return np.clip(np.round(smoothed_line), 0, h - 1).astype(np.int32)


def detect_skin_layer_lines(
    image_bgr: np.ndarray,
    center_half_width: int = 100,
    max_step: int = 6,
    init_band: int = 40,
    smooth_win_frac: float = 0.15,
) -> tuple[list[np.ndarray], int]:
    """
    표피 / 진피 경계선 탐지 (중앙 ±center_half_width px 평균 프로파일 사용).

    1. 맨 위 밝은 영역 → 어두워지는 지점 = 표피 하단 경계
    2. 그 아래 다음 유의미한 밝기 전환 = 진피 하단 경계

    반환
    ----
    skin_lines : 표피/진피 경계선 리스트 (최대 2개), 각각 (W,) int32
    skin_end_y : 피부층 탐지 종료 y좌표 (이 아래부터 일반 peak 탐지)
    """
    h, w = image_bgr.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enh   = clahe.apply(gray)

    # 중앙 ±100px 열만 평균
    cx      = w // 2
    x_start = max(0, cx - center_half_width)
    x_end   = min(w, cx + center_half_width)
    profile  = enh[:, x_start:x_end].astype(np.float32).mean(axis=1)
    smoothed = gaussian_filter1d(profile, sigma=max(1.5, h * 0.012))

    # 밝기 감소(drop) 지점 탐지: 음의 gradient 에서 peak 찾기
    gradient = np.gradient(smoothed)
    neg_grad = -gradient
    neg_grad = gaussian_filter1d(neg_grad, sigma=max(1.5, h * 0.008))

    drop_threshold = max(0.5, float(np.std(neg_grad)) * 0.8)
    drops, _ = find_peaks(neg_grad, prominence=drop_threshold, distance=max(5, h // 30))

    print(f"[Skin] drop points at rows: {drops.tolist()}")

    enh_smooth = cv2.GaussianBlur(enh, (1, 5), 0)
    mid_x      = w // 2
    win        = max(5, int(smooth_win_frac * w)) | 1
    win        = min(win, w - 2) | 1

    skin_lines: list[np.ndarray] = []
    skin_end_y = 0

    # 앞에서 최대 2개 drop (표피 하단, 진피 하단)
    for i, dy in enumerate(drops[:2]):
        line = _propagate_line(enh_smooth, int(dy), w, h, mid_x,
                               max_step, init_band, win)
        skin_lines.append(line)
        skin_end_y = max(skin_end_y, int(dy))
        label = "표피 하단" if i == 0 else "진피 하단"
        print(f"[Skin] {label} line: y={dy}")

    # 진피 이후에도 밝기 변화가 있으면 추가 탐지 (진피 내 전환점)
    if len(drops) > 0:
        dermis_start = int(drops[0]) + 5
        sub_profile  = smoothed[dermis_start:]
        if len(sub_profile) > 10:
            sub_neg = -np.gradient(gaussian_filter1d(sub_profile, sigma=max(1.5, h * 0.008)))
            sub_thr = max(0.5, float(np.std(sub_neg)) * 0.8)
            sub_drops, _ = find_peaks(sub_neg, prominence=sub_thr,
                                      distance=max(5, h // 30))
            for sd in sub_drops[:1]:   # 추가 1개만
                abs_y = dermis_start + int(sd)
                if abs_y > skin_end_y + 5:
                    line = _propagate_line(enh_smooth, abs_y, w, h, mid_x,
                                           max_step, init_band, win)
                    skin_lines.append(line)
                    skin_end_y = max(skin_end_y, abs_y)
                    print(f"[Skin] 진피 내 전환 line: y={abs_y}")

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
    """start_y 아래에서 모든 밝기 peak 를 좌우 propagation."""
    h, w = image_bgr.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enh   = clahe.apply(gray)

    profile  = enh.astype(np.float32).mean(axis=1)
    smoothed = gaussian_filter1d(profile, sigma=max(1.5, h * 0.01))

    # start_y 아래 구간만 탐색
    sub_smoothed = smoothed[start_y:]
    auto_prominence = max(1.0, float(np.std(sub_smoothed)) * 0.15)
    use_prominence  = prominence if prominence > 0.0 else auto_prominence
    peaks, _ = find_peaks(sub_smoothed, distance=min_dist_px, prominence=use_prominence)
    peaks = peaks + start_y   # 이미지 좌표로 변환

    print(f"[Peaks] prominence={use_prominence:.2f}  "
          f"{'(manual)' if prominence > 0.0 else '(auto)'}")
    if len(peaks) == 0:
        return []
    print(f"[Peaks] {len(peaks)} peaks at rows: {peaks.tolist()}")

    enh_smooth = cv2.GaussianBlur(enh, (1, 5), 0)
    mid_x      = w // 2
    win        = max(5, int(smooth_win_frac * w)) | 1
    win        = min(win, w - 2) | 1

    lines = []
    for pk_y in peaks:
        line = _propagate_line(enh_smooth, int(pk_y), w, h, mid_x,
                               max_step, init_band, win)
        lines.append(line)
    return lines


def draw_lines(
    image_bgr: np.ndarray,
    skin_lines: list[np.ndarray],
    peak_lines: list[np.ndarray],
) -> np.ndarray:
    """표피/진피 라인은 빨간색, 나머지 peak 라인은 흰색으로 그림."""
    out = image_bgr.copy()
    xs  = np.arange(out.shape[1], dtype=np.int32)

    for line in peak_lines:
        pts = np.stack((xs, line), axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)

    for line in skin_lines:
        pts = np.stack((xs, line), axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 0, 255), 2, cv2.LINE_AA)   # 빨간색, 굵게

    return out


# ---------------------------------------------------------------------------
# 상태 관리
# ---------------------------------------------------------------------------

class State:
    def __init__(self, base: np.ndarray):
        self.base    = base.copy()   # peak 라인 포함 기본 이미지
        self.canvas  = base.copy()   # SAM 마스크 누적 overlay
        self.drawing = False
        self.x0 = self.y0 = 0
        self.x1 = self.y1 = 0
        self.mask_count = 0


def draw_box(state: State) -> np.ndarray:
    frame = state.canvas.copy()
    cv2.rectangle(frame, (state.x0, state.y0), (state.x1, state.y1),
                  (0, 200, 255), 2)
    return frame


def apply_mask(state: State, mask: np.ndarray, color: tuple) -> None:
    overlay = state.canvas.astype(np.float32)
    overlay[mask] = overlay[mask] * 0.45 + np.array(color, np.float32) * 0.55
    state.canvas = overlay.astype(np.uint8)
    seg_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(state.canvas, contours, -1, color, 2)
    state.mask_count += 1


# ---------------------------------------------------------------------------
# 마우스 콜백
# ---------------------------------------------------------------------------

def make_callback(state: State, predictor: SamPredictor,
                  orig_h: int, orig_w: int, window: str):
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.x0, state.y0 = x, y
            state.x1, state.y1 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
            state.x1, state.y1 = x, y
            cv2.imshow(window, draw_box(state))

        elif event == cv2.EVENT_LBUTTONUP:
            state.drawing = False
            state.x1, state.y1 = x, y

            x1 = min(state.x0, state.x1)
            y1 = min(state.y0, state.y1)
            x2 = max(state.x0, state.x1)
            y2 = max(state.y0, state.y1)

            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                return

            # 표시 좌표 → 1024x1024 기준으로 변환
            box_1024 = scale_box(
                np.array([x1, y1, x2, y2], dtype=np.float32),
                orig_h=state.base.shape[0],
                orig_w=state.base.shape[1],
            )
            print(f"[Box] 표시 ({x1},{y1})→({x2},{y2})  "
                  f"1024기준 ({box_1024[0]:.0f},{box_1024[1]:.0f})→"
                  f"({box_1024[2]:.0f},{box_1024[3]:.0f})")

            # MedSAM 추론 (multimask_output=False — 논문 권장)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_1024,
                multimask_output=False,
            )
            mask = masks[0]
            print(f"  IOU score: {scores[0]:.3f}  pixels: {mask.sum()}")

            # 마스크를 표시 해상도로 리사이즈
            mask_display = cv2.resize(
                mask.astype(np.uint8),
                (state.base.shape[1], state.base.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            color = MASK_COLORS[state.mask_count % len(MASK_COLORS)]
            apply_mask(state, mask_display, color)
            cv2.imshow(window, state.canvas)

    return callback


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedSAM box segmentation + peak 라인")
    p.add_argument("--input",       required=True,
                   help="입력 이미지 경로")
    p.add_argument("--checkpoint",  required=True,
                   help="MedSAM checkpoint 경로 (medsam_vit_b.pth)")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--min-dist-px", type=int, default=10,
                   help="peak 간 최소 거리 px (default: 10)")
    p.add_argument("--max-step",    type=int, default=6,
                   help="좌우 propagation 최대 y 이동 px (default: 6)")
    p.add_argument("--init-band",   type=int,   default=40,
                   help="중앙 초기 탐색 반경 px (default: 40)")
    p.add_argument("--prominence",  type=float, default=0.0,
                   help="peak 채택 prominence 최솟값 (default: 0 → 자동). "
                        "높일수록 약한 peak 제거됨")
    p.add_argument("--clahe",       type=float, default=0.0,
                   help="CLAHE clipLimit (0=비활성, 클수록 강함, 예: 3.0 / 8.0 / 20.0)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[Device] {device}")

    # ── MedSAM 로드 (항상 vit_b) ───────────────────────────────────────────────
    print(f"[MedSAM] Loading vit_b from {args.checkpoint} ...")
    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    # ── 이미지 로드 ────────────────────────────────────────────────────────────
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {args.input}")

    orig_h, orig_w = image_bgr.shape[:2]
    print(f"[Image] 원본 크기: {orig_w}x{orig_h}")

    # ── MedSAM 전처리 & 임베딩 ────────────────────────────────────────────────
    image_1024_rgb = medsam_preprocess(image_bgr, apply_clahe=args.clahe)
    print(f"[Preprocess] CLAHE {'clipLimit=' + str(args.clahe) if args.clahe > 0 else '미적용'}")
    print("[MedSAM] Computing image embedding (1024x1024) ...")
    predictor.set_image(image_1024_rgb)

    # ── 표피/진피 경계선 탐지 (빨간색) ───────────────────────────────────────
    print("[Skin] Detecting epidermis / dermis lines ...")
    skin_lines, skin_end_y = detect_skin_layer_lines(
        image_bgr,
        max_step=args.max_step,
        init_band=args.init_band,
    )
    print(f"[Skin] {len(skin_lines)} skin lines, skin_end_y={skin_end_y}")

    # ── 진피 아래 peak 라인 탐지 (흰색) ──────────────────────────────────────
    print("[Peaks] Detecting peak lines below dermis ...")
    peak_lines = detect_all_peak_lines(
        image_bgr,
        start_y=max(0, skin_end_y + 5),
        min_dist_px=args.min_dist_px,
        max_step=args.max_step,
        init_band=args.init_band,
        prominence=args.prominence,
    )
    print(f"[Peaks] {len(peak_lines)} peak lines.")

    display_with_lines = draw_lines(image_bgr, skin_lines, peak_lines)
    print(f"[Lines] total {len(skin_lines) + len(peak_lines)} lines drawn.")
    print("[MedSAM] Ready. 마우스로 box 를 드래그하세요.\n")

    # ── 윈도우 & 콜백 ─────────────────────────────────────────────────────────
    WINDOW = "MedSAM Box + Peak Lines  |  R: 마스크 초기화  |  Q/ESC: 종료"
    state  = State(display_with_lines)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(
        WINDOW,
        make_callback(state, predictor, orig_h, orig_w, WINDOW),
    )
    cv2.imshow(WINDOW, state.canvas)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            state.canvas     = display_with_lines.copy()
            state.mask_count = 0
            cv2.imshow(WINDOW, state.canvas)
            print("[Reset] 마스크 초기화")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
