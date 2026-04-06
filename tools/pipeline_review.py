"""
pipeline_review.py

Step 2.4 전처리: 전파된 마스크에서 균일 간격 키프레임 추출.
추출된 키프레임 목록을 keyframes.txt에 저장하고
annotation_tool.py --keyframes_only 로 정제 가능.

사용법:
    python pipeline_review.py --video_name output1 --num_keyframes 5
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name",    required=True, help="예: output1")
    parser.add_argument("--video_dir",     default="result_resolution")
    parser.add_argument("--mask_dir",      default="annotation")
    parser.add_argument("--num_keyframes", type=int, default=5, help="추출할 키프레임 수 (3~10 권장)")
    args = parser.parse_args()

    frames_dir = Path(args.video_dir) / args.video_name / "frames"
    mask_dir   = Path(args.mask_dir)  / args.video_name

    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        print(f"[Error] 프레임 없음: {frames_dir}")
        return

    total = len(frame_files)
    n = min(args.num_keyframes, total)

    # 이미 annotation이 있는 프레임 제외
    already_annotated = {f.stem for f in mask_dir.glob("*.png") if f.stem != "keyframes"}
    unannotated = [f for f in frame_files if f.stem not in already_annotated]

    if not unannotated:
        print("[Info] 모든 프레임에 수정본이 있습니다. 추가 키프레임 없음.")
        return

    # 미수정 프레임 중에서 균일 간격 추출
    total_un = len(unannotated)
    n = min(n, total_un)
    indices = [int(total_un * i / (n + 1)) for i in range(1, n + 1)]
    keyframes = [unannotated[i] for i in indices]

    print(f"[Info] 전체 {total}프레임 중 수정 완료 {len(already_annotated)}개, 미수정 {total_un}개")

    # keyframes.txt 저장
    mask_dir.mkdir(parents=True, exist_ok=True)
    kf_path = mask_dir / "keyframes.txt"
    with open(kf_path, "w") as f:
        for kf in keyframes:
            f.write(kf.stem + "\n")

    print(f"[Info] 총 {total}프레임에서 {n}개 키프레임 추출:")
    for i, kf in zip(indices, keyframes):
        mask_path = mask_dir / f"{kf.stem}.png"
        status = "마스크 있음" if mask_path.exists() else "마스크 없음"
        print(f"  [{i:4d}] {kf.stem}  ({status})")

    print(f"\n[Done] → {kf_path}")
    print(f"\n다음 명령어로 키프레임 정제:")
    print(f"  python annotation_tool.py \\")
    print(f"      --video_dir {frames_dir} \\")
    print(f"      --mask_dir {mask_dir} \\")
    print(f"      --keyframes_only")


if __name__ == "__main__":
    main()
