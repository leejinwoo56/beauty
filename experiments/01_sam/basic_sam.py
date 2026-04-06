"""
기본 SAM 추론 스크립트.

SamAutomaticMaskGenerator 로 전체 마스크를 생성하고
모두 하나의 이미지에 overlay 해서 저장.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="기본 SAM 추론 — 전체 마스크 overlay")
    p.add_argument("--input",      required=True,  help="입력 이미지 경로")
    p.add_argument("--checkpoint", required=True,  help="SAM checkpoint 경로 (.pth)")
    p.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--device",     default="cuda")
    p.add_argument("--outdir",     default="outputs_auto")
    p.add_argument("--points-per-side", type=int, default=128)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[Device] {device}")

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    print(f"[SAM] Loading '{args.model_type}' from {args.checkpoint} ...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()

    # ── 이미지 로드 ────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    image_bgr  = cv2.imread(str(input_path))
    image_bgr = cv2.resize(image_bgr,(1024,1024))

    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {input_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── 전체 마스크 자동 생성 ──────────────────────────────────────────────────
    print(f"[SAM] Generating all masks (points_per_side={args.points_per_side}) ...")
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
    )
    with torch.no_grad():
        masks = generator.generate(image_rgb)

    print(f"[SAM] {len(masks)} masks detected.")

    # 면적 큰 순 정렬 (큰 영역이 먼저 칠해지고 작은 영역이 위에 덮임)
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    # ── 전체 마스크를 하나의 이미지에 overlay ─────────────────────────────────
    overlay = image_bgr.copy().astype(np.float32)
    np.random.seed(42)
    for m in masks:
        seg   = m["segmentation"]
        color = np.random.randint(50, 255, 3).astype(np.float32)
        overlay[seg] = overlay[seg] * 0.5 + color * 0.5

    overlay = overlay.astype(np.uint8)
    import matplotlib.pyplot as plt
    plt.imshow(overlay)
    plt.show()
    # 경계선 추가
    contour_img = overlay.copy()
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = input_path.stem

    cv2.imwrite(str(outdir / f"{base}_overlay.png"),  overlay)
    cv2.imwrite(str(outdir / f"{base}_contours.png"), contour_img)
    print(f"[Saved] {outdir / f'{base}_overlay.png'}")
    print(f"[Saved] {outdir / f'{base}_contours.png'}")

    # 상위 10개 마스크 정보 출력
    print("\n[Masks] 면적 큰 순 상위 10개:")
    for i, m in enumerate(masks[:10]):
        print(f"  {i+1}: area={m['area']:>7d}  iou={m['predicted_iou']:.3f}  stability={m['stability_score']:.3f}")

    if args.show:
        cv2.imshow("Overlay",  overlay)
        cv2.imshow("Contours", contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
