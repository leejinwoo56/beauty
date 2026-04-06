"""
annotation_tool.py

OpenCV 기반 인터랙티브 마스크 주석 도구.
논문 Step 2.1~2.2 (첫 프레임 주석) 및 Step 2.4 (중간 프레임 정제)에 사용.

조작법:
    마우스:
        왼쪽 드래그  : BBox 그리기 → SAM2 자동 예측
        오른쪽 드래그: 마스크 지우기 (브러시)
        가운데 드래그: 마스크 칠하기 (브러시)
    키보드:
        s : 현재 마스크 저장 후 다음 프레임
        p : 이전 프레임으로
        r : 현재 프레임 초기화 (마스크 삭제)
        b : BBox 모드로 초기화 (새 BBox 그리기)
        +/- : 브러시 크기 조절
        q : 종료

사용법:
    # 첫 프레임 주석 (Step 2.1~2.2)
    python annotation_tool.py --video_dir result_resolution/output1/frames --mask_dir annotation/output1

    # 중간 프레임 정제 (Step 2.4, pipeline_review.py 실행 후)
    python annotation_tool.py --video_dir result_resolution/output1/frames --mask_dir annotation/output1 --keyframes_only
"""

import argparse
import sys
import os
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "MedSAM2"))

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── 설정 ──────────────────────────────────────────────────────────────────────
SAM2_CFG   = "configs/sam2.1_hiera_t512.yaml"
SAM2_CKPT  = "MedSAM2/checkpoints/MedSAM2_US_Heart.pt"
OVERLAY_ALPHA = 0.25
MASK_COLOR    = (0, 200, 0)   # BGR green
TOP_BAR_H     = 40            # draw_ui 상단 바 높이 (px) — 마우스 오프셋 보정용

# ── 전역 상태 ──────────────────────────────────────────────────────────────────
state = {
    "mode": "bbox",       # "bbox" | "paint" | "erase"
    "drawing": False,
    "bbox_start": None,
    "bbox_end": None,
    "brush_size": 15,
    "mask": None,
    "image": None,
    "predictor": None,
    "cursor_pos": None,   # 브러시 커서 표시용
}


def load_predictor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    return SAM2ImagePredictor(model)


def apply_overlay(image, mask):
    vis = image.copy()
    if mask is not None and mask.any():
        color = np.array(MASK_COLOR, dtype=np.float32)
        region = mask > 0
        vis[region] = (
            image[region].astype(np.float32) * (1 - OVERLAY_ALPHA) + color * OVERLAY_ALPHA
        ).astype(np.uint8)
        # 윤곽선 그리기
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    return vis


def predict_from_bbox(predictor, image_rgb, bbox):
    """bbox [x1,y1,x2,y2] → binary mask"""
    predictor.set_image(image_rgb)
    box = np.array(bbox)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    best = np.argmax(scores)
    return masks[best].astype(np.uint8)


def mouse_callback(event, x, y, flags, param):
    s = state
    h, w = s["image"].shape[:2]

    # 상단 UI 바 높이만큼 오프셋 보정
    y = y - TOP_BAR_H
    x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)

    # 커서 위치 항상 업데이트 (브러시 표시용)
    s["cursor_pos"] = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        s["drawing"] = True
        s["mode"] = "bbox"
        s["bbox_start"] = (x, y)
        s["bbox_end"] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if not s["drawing"]:
            return
        if s["mode"] == "bbox":
            s["bbox_end"] = (x, y)
        elif s["mode"] == "erase":
            cv2.circle(s["mask"], (x, y), s["brush_size"], 0, -1)
        elif s["mode"] == "paint":
            cv2.circle(s["mask"], (x, y), s["brush_size"], 1, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        if s["mode"] == "bbox" and s["bbox_start"] and s["bbox_end"]:
            x1, y1 = s["bbox_start"]
            x2, y2 = s["bbox_end"]
            if abs(x2-x1) > 5 and abs(y2-y1) > 5:
                bbox = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
                image_rgb = cv2.cvtColor(s["image"], cv2.COLOR_BGR2RGB)
                print("  [SAM2] 예측 중...", end="\r")
                new_mask = predict_from_bbox(s["predictor"], image_rgb, bbox)
                s["mask"] = np.clip(s["mask"] + new_mask, 0, 1).astype(np.uint8)  # 기존 마스크와 합치기(OR)
                print("  [SAM2] 예측 완료   ")
        s["drawing"] = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        s["drawing"] = True
        s["mode"] = "erase"
        cv2.circle(s["mask"], (x, y), s["brush_size"], 0, -1)

    elif event == cv2.EVENT_RBUTTONUP:
        s["drawing"] = False
        s["mode"] = "bbox"

    elif event == cv2.EVENT_MBUTTONDOWN:
        s["drawing"] = True
        s["mode"] = "paint"
        cv2.circle(s["mask"], (x, y), s["brush_size"], 1, -1)

    elif event == cv2.EVENT_MBUTTONUP:
        s["drawing"] = False
        s["mode"] = "bbox"

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            s["brush_size"] = min(s["brush_size"] + 2, 60)
        else:
            s["brush_size"] = max(s["brush_size"] - 2, 3)


def draw_ui(vis, mask, brush_size, frame_idx, total, frame_name, saved):
    h, w = vis.shape[:2]
    # 상단 상태바
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    status = f"Frame {frame_idx+1}/{total}  [{frame_name}]  Brush:{brush_size}px"
    if saved:
        status += "  [SAVED]"
    fg_pixels = int(mask.sum()) if mask is not None else 0
    status += f"  FG:{fg_pixels}px"
    cv2.putText(bar, status, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # 하단 도움말
    help_bar = np.zeros((30, w, 3), dtype=np.uint8)
    help_txt = "LClick:BBox  RClick:Erase  MClick:Paint  s:Save+Next  p:Prev  r:Reset  b:NewBBox  +/-:Brush  q:Quit"
    cv2.putText(help_bar, help_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return np.vstack([bar, vis, help_bar])


def run_annotation(frame_files, mask_dir, predictor, start_idx=0):
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    total = len(frame_files)

    state["predictor"] = predictor

    cv2.namedWindow("Annotation Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotation Tool", 1200, 700)
    cv2.setMouseCallback("Annotation Tool", mouse_callback)

    idx = start_idx
    saved_flag = False

    while 0 <= idx < total:
        frame_path = frame_files[idx]
        frame_name = frame_path.stem
        image = cv2.imread(str(frame_path))
        h, w = image.shape[:2]

        state["image"] = image
        state["bbox_start"] = None
        state["bbox_end"] = None

        # 마스크 로드: 수정 저장본 → 전파 결과 → 빈 마스크 순으로 우선순위
        mask_path = mask_dir / f"{frame_name}.png"
        prop_path = state.get("propagated_dir") and Path(state["propagated_dir"]) / f"{frame_name}.png"

        if mask_path.exists():
            state["mask"] = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            status = "(수정본 로드)"
        elif prop_path and prop_path.exists():
            state["mask"] = (cv2.imread(str(prop_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            status = "(전파 결과 로드)"
        else:
            state["mask"] = np.zeros((h, w), dtype=np.uint8)
            status = "(새 프레임)"

        saved_flag = mask_path.exists()
        print(f"\n[Frame {idx+1}/{total}] {frame_name}  {status}")

        while True:
            vis = apply_overlay(state["image"], state["mask"])

            # bbox 드래그 중이면 사각형 표시
            if state["mode"] == "bbox" and state["drawing"] and state["bbox_start"] and state["bbox_end"]:
                cv2.rectangle(vis, state["bbox_start"], state["bbox_end"], (0, 100, 255), 2)

            # 브러시 커서 표시 (erase=빨강, paint=파랑)
            cp = state["cursor_pos"]
            if cp is not None and state["mode"] in ("erase", "paint"):
                color = (0, 0, 220) if state["mode"] == "erase" else (220, 100, 0)
                cv2.circle(vis, cp, state["brush_size"], color, 2)
                cv2.circle(vis, cp, 2, color, -1)

            display = draw_ui(vis, state["mask"], state["brush_size"], idx, total, frame_name, saved_flag)
            cv2.imshow("Annotation Tool", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                print("[종료]")
                return

            elif key == ord('s'):
                # 저장 후 다음 프레임
                out_mask = (state["mask"] * 255).astype(np.uint8)
                cv2.imwrite(str(mask_path), out_mask)
                saved_flag = True
                print(f"  [저장] {mask_path}")
                idx += 1
                break

            elif key == ord('p'):
                idx -= 1
                break

            elif key == ord('r'):
                state["mask"] = np.zeros((h, w), dtype=np.uint8)
                saved_flag = False
                print("  [초기화]")

            elif key == ord('b'):
                state["mode"] = "bbox"
                state["bbox_start"] = None
                state["bbox_end"] = None
                print("  [BBox 모드]")

            elif key == ord('+') or key == ord('='):
                state["brush_size"] = min(state["brush_size"] + 2, 60)

            elif key == ord('-'):
                state["brush_size"] = max(state["brush_size"] - 2, 3)

    cv2.destroyAllWindows()
    print("\n[완료] 모든 프레임 처리됨")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",     required=True, help="프레임 이미지 폴더")
    parser.add_argument("--mask_dir",      required=True, help="마스크 저장 폴더")
    parser.add_argument("--keyframes_only",  action="store_true", help="keyframes.txt에 있는 프레임만 표시")
    parser.add_argument("--propagated_dir",  default=None, help="전파 결과 마스크 폴더 (Step 2.4 정제 시 오버레이 초기값)")
    parser.add_argument("--start",           type=int, default=0, help="시작 프레임 인덱스")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    all_frames = sorted(video_dir.glob("*.png")) + sorted(video_dir.glob("*.jpg"))

    if args.keyframes_only:
        kf_path = Path(args.mask_dir) / "keyframes.txt"
        if not kf_path.exists():
            print(f"[Error] keyframes.txt 없음: {kf_path}")
            print("  pipeline_review.py 를 먼저 실행하세요.")
            return
        with open(kf_path) as f:
            keyframe_names = set(line.strip() for line in f)
        all_frames = [f for f in all_frames if f.stem in keyframe_names]
        print(f"[Info] 키프레임 {len(all_frames)}개 로드")

    if not all_frames:
        print(f"[Error] 프레임 없음: {video_dir}")
        return

    print(f"[Info] 총 {len(all_frames)}개 프레임")
    if args.propagated_dir:
        state["propagated_dir"] = args.propagated_dir
        print(f"[Info] 전파 결과 로드: {args.propagated_dir}")
    print("[Info] SAM2 모델 로딩 중...")
    predictor = load_predictor()
    print("[Info] 모델 로딩 완료. 창이 열립니다.\n")

    run_annotation(all_frames, args.mask_dir, predictor, start_idx=args.start)


if __name__ == "__main__":
    main()
