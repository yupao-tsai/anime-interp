"""
Evaluation: PSNR, SSIM, and palette consistency for interpolated frames.

Usage:
  # Compare predicted vs ground-truth frames
  python eval.py --pred outputs/raw --gt data/gt_frames

  # Evaluate palette consistency only (no GT needed)
  python eval.py --pred outputs/snapped --palette_ref reference.png --palette_only

  # Full evaluation with JSON report
  python eval.py --pred outputs/raw --gt data/gt_frames \
      --palette_ref reference.png --output_json results.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans


# ── Metric helpers ────────────────────────────────────────────────────────────

def load_frames_as_array(folder: str, sort: bool = True) -> np.ndarray:
    """Load all PNG/JPG frames from a folder → (N, H, W, 3) uint8."""
    folder = Path(folder)
    paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No frames found in {folder}")
    frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
    return np.stack(frames, axis=0)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR in dB. pred/gt: (N, H, W, 3) uint8."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_frame(pred: np.ndarray, gt: np.ndarray, c1=6.5025, c2=58.5225) -> float:
    """Single-frame SSIM (per-channel mean). pred/gt: (H, W, 3) uint8."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    scores = []
    for c in range(3):
        p = pred[:, :, c]
        g = gt[:, :, c]
        mu_p = p.mean()
        mu_g = g.mean()
        sig_p = p.std()
        sig_g = g.std()
        sig_pg = np.mean((p - mu_p) * (g - mu_g))
        numerator = (2 * mu_p * mu_g + c1) * (2 * sig_pg + c2)
        denominator = (mu_p ** 2 + mu_g ** 2 + c1) * (sig_p ** 2 + sig_g ** 2 + c2)
        scores.append(numerator / denominator)
    return float(np.mean(scores))


def ssim_sequence(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean SSIM over N frames."""
    scores = [ssim_frame(pred[i], gt[i]) for i in range(len(pred))]
    return float(np.mean(scores))


def extract_palette(image_path: str, k: int = 16) -> np.ndarray:
    """K-means palette from reference image → (K, 3) float [0,1]."""
    img = np.array(Image.open(image_path).convert("RGB")).reshape(-1, 3).astype(np.float32) / 255.0
    idx = np.random.choice(len(img), min(50000, len(img)), replace=False)
    km = KMeans(n_clusters=k, n_init=5, random_state=0)
    km.fit(img[idx])
    return km.cluster_centers_


def palette_consistency(frames: np.ndarray, palette: np.ndarray) -> dict:
    """
    Measures how well the frames conform to the target palette.

    Args:
        frames:  (N, H, W, 3) uint8
        palette: (K, 3) float [0, 1]

    Returns:
        dict with keys: mean_dist, p90_dist, snap_psnr
    """
    frames_f = frames.astype(np.float32) / 255.0  # [0,1]
    N, H, W, _ = frames_f.shape
    flat = frames_f.reshape(-1, 3)                 # (N*H*W, 3)

    # nearest-palette distance per pixel
    diffs = flat[:, None, :] - palette[None, :, :]  # (M, K, 3)
    dists = np.linalg.norm(diffs, axis=-1)           # (M, K)
    min_idx = dists.argmin(axis=-1)                  # (M,)
    min_dist = dists[np.arange(len(flat)), min_idx]  # (M,)

    snapped = palette[min_idx].reshape(N, H, W, 3)   # [0,1]
    snap_psnr = psnr(
        (snapped * 255).astype(np.uint8),
        frames,
    )

    return {
        "mean_pixel_palette_dist": float(min_dist.mean()),
        "p90_pixel_palette_dist": float(np.percentile(min_dist, 90)),
        "snap_psnr_db": float(snap_psnr),
    }


def temporal_consistency(frames: np.ndarray) -> float:
    """Mean per-pixel L1 difference between consecutive frames (lower = smoother)."""
    diffs = np.abs(
        frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)
    ).mean()
    return float(diffs)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eval(
    pred_dir: str,
    gt_dir: str | None = None,
    palette_ref: str | None = None,
    palette_k: int = 16,
    palette_only: bool = False,
) -> dict:
    results = {}

    pred_frames = load_frames_as_array(pred_dir)
    print(f"Loaded {len(pred_frames)} predicted frames from {pred_dir}")

    if not palette_only and gt_dir is not None:
        gt_frames = load_frames_as_array(gt_dir)
        n = min(len(pred_frames), len(gt_frames))
        pred_frames_matched = pred_frames[:n]
        gt_frames_matched = gt_frames[:n]
        print(f"Loaded {len(gt_frames)} GT frames — comparing {n} pairs")

        results["psnr_db"] = psnr(pred_frames_matched, gt_frames_matched)
        results["ssim"] = ssim_sequence(pred_frames_matched, gt_frames_matched)
        print(f"  PSNR:  {results['psnr_db']:.2f} dB")
        print(f"  SSIM:  {results['ssim']:.4f}")

    results["temporal_smoothness"] = temporal_consistency(pred_frames)
    print(f"  Temporal L1: {results['temporal_smoothness']:.4f}")

    if palette_ref is not None:
        palette = extract_palette(palette_ref, k=palette_k)
        pal_metrics = palette_consistency(pred_frames, palette)
        results.update(pal_metrics)
        print(f"  Palette mean dist: {pal_metrics['mean_pixel_palette_dist']:.4f}")
        print(f"  Palette p90  dist: {pal_metrics['p90_pixel_palette_dist']:.4f}")
        print(f"  Snap PSNR:         {pal_metrics['snap_psnr_db']:.2f} dB")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Folder of predicted frames")
    parser.add_argument("--gt", default=None, help="Folder of ground-truth frames")
    parser.add_argument("--palette_ref", default=None, help="Reference image for palette")
    parser.add_argument("--palette_k", type=int, default=16)
    parser.add_argument("--palette_only", action="store_true")
    parser.add_argument("--output_json", default=None, help="Save results to JSON")
    args = parser.parse_args()

    results = run_eval(
        pred_dir=args.pred,
        gt_dir=args.gt,
        palette_ref=args.palette_ref,
        palette_k=args.palette_k,
        palette_only=args.palette_only,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
