import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np
import torch


def add_repo_to_path(repo_path: Path) -> None:
    repo_str = str(repo_path.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess(image_gray: np.ndarray, input_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = image_gray.shape[:2]
    resized = cv2.resize(image_gray, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return x, (h, w)


def extract_prediction_tensor(model_out: Any) -> torch.Tensor:
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (list, tuple)):
        for v in model_out:
            if torch.is_tensor(v):
                return v
    if isinstance(model_out, dict):
        for k in ["masks", "mask", "pred", "logits", "output"]:
            v = model_out.get(k)
            if torch.is_tensor(v):
                return v
        for v in model_out.values():
            if torch.is_tensor(v):
                return v
    raise RuntimeError("Could not find prediction tensor from model output.")


def run_model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        # AutoSAMUS is expected to be auto-prompted, but keep fallbacks for API differences.
        try:
            out = model(x)
        except TypeError:
            try:
                out = model(x, None)
            except TypeError:
                # Fallback with a dummy point prompt if needed by SAMUS-style signatures.
                dummy_points = (
                    torch.tensor([[[x.shape[-1] // 2, x.shape[-2] // 2]]], dtype=torch.float32, device=x.device),
                    torch.tensor([[1]], dtype=torch.float32, device=x.device),
                )
                out = model(x, dummy_points)
    return extract_prediction_tensor(out)


def logits_to_mask(pred: torch.Tensor) -> np.ndarray:
    # Normalize to [B,C,H,W]
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
    elif pred.ndim != 4:
        raise RuntimeError(f"Unexpected prediction shape: {tuple(pred.shape)}")

    pred = pred.float()
    if pred.shape[1] == 1:
        prob = torch.sigmoid(pred[:, 0])
        mask = (prob > 0.5).cpu().numpy().astype(np.uint8)[0]
    else:
        cls = torch.argmax(pred, dim=1)
        mask = (cls > 0).cpu().numpy().astype(np.uint8)[0]
    return mask


def save_overlay(image_gray: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    color = np.array([80, 220, 255], dtype=np.float32)  # BGR
    m = mask.astype(bool)
    base[m] = base[m] * 0.45 + color * 0.55
    cv2.imwrite(str(out_path), np.clip(base, 0, 255).astype(np.uint8))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-image AutoSAMUS inference")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--checkpoint", required=True, help="AutoSAMUS checkpoint path (.pth)")
    p.add_argument("--samus-repo", required=True, help="Path to local SAMUS repository root")
    p.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--input-size", type=int, default=256, help="Encoder input size (SAMUS default 256)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", None])
    p.add_argument("--output-dir", default="outputs_autosamus")
    return p


def main() -> None:
    args = build_parser().parse_args()

    image_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    repo_path = Path(args.samus_repo)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not repo_path.exists():
        raise FileNotFoundError(f"SAMUS repo not found: {repo_path}")

    add_repo_to_path(repo_path)
    from models.model_dict import get_model  # type: ignore

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    # Minimal args/opt expected by SAMUS model factory.
    model_args = SimpleNamespace(
        modelname="AutoSAMUS",
        encoder_input_size=args.input_size,
        low_image_size=args.input_size // 2,
        vit_name=args.model_type,
        sam_ckpt="",  # not used by AutoSAMUS path in model_dict
    )
    model_opt = SimpleNamespace(load_path=str(ckpt_path))

    model = get_model("AutoSAMUS", args=model_args, opt=model_opt).to(device)
    model.eval()

    image_raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_raw is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_gray = ensure_gray(image_raw)

    x, (h, w) = preprocess(image_gray, args.input_size)
    x = x.to(device)

    pred = run_model(model, x)
    mask_small = logits_to_mask(pred)
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    stem = image_path.stem
    mask_path = out_dir / f"{stem}_autosamus_mask.png"
    overlay_path = out_dir / f"{stem}_autosamus_overlay.png"
    meta_path = out_dir / f"{stem}_autosamus_meta.json"

    cv2.imwrite(str(mask_path), mask * 255)
    save_overlay(image_gray, mask, overlay_path)
    meta = {
        "image": str(image_path),
        "checkpoint": str(ckpt_path),
        "samus_repo": str(repo_path),
        "model_type": args.model_type,
        "input_size": int(args.input_size),
        "device": device,
        "mask_area": int(mask.sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Saved mask: {mask_path.resolve()}")
    print(f"Saved overlay: {overlay_path.resolve()}")
    print(f"Saved metadata: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
