"""
pipeline_propagate.py

Step 2.3 / 2.5: annotation_tool.py로 주석한 키프레임 마스크를
MedSAM2로 전체 영상에 전파.

주석된 프레임이 많을수록 (Step 2.5) 결과가 더 정확합니다.

사용법:
    # Step 2.3: 첫 프레임만 전파
    python pipeline_propagate.py --video_name output1

    # Step 2.5: 키프레임 포함 재전파
    python pipeline_propagate.py --video_name output1 --use_all_masks
"""

import argparse
import sys
import os
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "MedSAM2"))

import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

SAM2_CFG  = "configs/sam2.1_hiera_t512.yaml"
SAM2_CKPT = "MedSAM2/checkpoints/MedSAM2_latest.pt"


def load_mask_png(path):
    mask = np.array(Image.open(path))
    return (mask > 0).astype(bool)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def propagate(predictor, video_dir, annotation_dir, output_dir, use_all_masks):
    frame_files = sorted(
        [p for p in Path(video_dir).iterdir()
         if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    frame_names = [f.stem for f in frame_files]
    total = len(frame_names)

    inference_state = predictor.init_state(
        video_path=str(video_dir), async_loading_frames=False
    )

    # 주석 마스크 로드
    ann_dir = Path(annotation_dir)
    annotated = {
        f.stem: f for f in ann_dir.glob("*.png")
        if f.stem in frame_names
    }

    if not annotated:
        print(f"[Error] 주석 마스크 없음: {ann_dir}")
        return

    # 어떤 프레임을 프롬프트로 쓸지 결정
    if use_all_masks:
        prompt_frames = sorted(annotated.keys())
    else:
        prompt_frames = [sorted(annotated.keys())[0]]  # 첫 주석 프레임만

    print(f"[Info] 전체 {total}프레임 / 프롬프트 {len(prompt_frames)}개 프레임 사용")

    for frame_name in prompt_frames:
        frame_idx = frame_names.index(frame_name)
        mask = load_mask_png(annotated[frame_name])
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            mask=mask,
        )
        print(f"  프롬프트 추가: [{frame_idx:4d}] {frame_name}")

    # 전파
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Propagate] 시작...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
        out_path = output_dir / f"{frame_names[out_frame_idx]}.png"
        Image.fromarray(mask).save(str(out_path))
        print(f"  [{out_frame_idx+1:4d}/{total}]", end="\r")

    print(f"\n[Done] 마스크 저장 → {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name",   required=True, help="예: output1")
    parser.add_argument("--video_dir",    default="result_resolution")
    parser.add_argument("--ann_dir",      default="annotation")
    parser.add_argument("--output_dir",   default="propagated_masks")
    parser.add_argument("--use_all_masks",action="store_true", help="Step 2.5: 모든 키프레임 마스크 사용")
    args = parser.parse_args()

    video_frames_dir = Path(args.video_dir) / args.video_name / "frames"
    annotation_dir   = Path(args.ann_dir)   / args.video_name
    output_dir       = Path(args.output_dir) / args.video_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Info] SAM2 모델 로딩 중...")
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    print("[Info] 완료\n")

    propagate(predictor, video_frames_dir, annotation_dir, output_dir, args.use_all_masks)

    print(f"\n결과 영상 생성:")
    print(f"  python visualize_results.py \\")
    print(f"      --output_name {args.video_name} \\")
    print(f"      --frames_dir {args.video_dir}/{args.video_name}/frames \\")
    print(f"      --masks_dir {args.output_dir}")


if __name__ == "__main__":
    main()
