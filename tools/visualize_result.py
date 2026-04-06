"""
visualize_results.py

MedSAM2 output masks를 result_resolution 원본 프레임에 오버레이해서 영상으로 저장.

사용법:
    python visualize_results.py --output_name output1
    python visualize_results.py --output_name output1 --fps 10
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name", required=True, help="예: output1")
    parser.add_argument("--frames_dir",  default="medsam_input_frame")
    parser.add_argument("--masks_dir",   default="medsam_output_mask")
    parser.add_argument("--output",      default=None, help="출력 mp4 경로 (기본: output_medsam2_{output_name}.mp4)")
    parser.add_argument("--fps",         type=float, default=10.0)
    parser.add_argument("--color",       nargs=3, type=int, default=[0, 200, 0], metavar=("B", "G", "R"))
    parser.add_argument("--alpha",       type=float, default=0.5)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    masks_dir  = Path(args.masks_dir)  / args.output_name
    output_path = Path(args.output or f"medsam2_output_video/output_medsam2_{args.output_name}.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.png" or "*.png"))
    masks  = sorted(masks_dir.glob("*.png"))

    if not frames:
        print(f"[Error] 프레임 없음: {frames_dir}")
        return
    if not masks:
        print(f"[Error] 마스크 없음: {masks_dir}")
        return

    print(f"[Info] 프레임 {len(frames)}개 / 마스크 {len(masks)}개")

    sample = cv2.imread(str(frames[0]))
    h, w   = sample.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    color = np.array(args.color, dtype=np.float32)

    for i, (frame_path, mask_path) in enumerate(zip(frames, masks)):
        frame = cv2.imread(str(frame_path))
        mask  = np.array(Image.open(mask_path))

        overlay = frame.copy()
        region  = mask > 0
        overlay[region] = (
            frame[region].astype(np.float32) * (1 - args.alpha) + color * args.alpha
        ).astype(np.uint8)

        writer.write(overlay)
        print(f"  [{i+1}/{len(frames)}]", end="\r")

    writer.release()
    print(f"\n[Done] → {output_path}")


if __name__ == "__main__":
    main()
