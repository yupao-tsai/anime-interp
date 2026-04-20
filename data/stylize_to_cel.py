"""
Convert composited anime frames (e.g. YouTube downloads) to cel-style:
  - Bilateral filter to suppress gradients while preserving edges
  - K-means quantization with a CLIP-CONSISTENT palette (no temporal flicker)
  - Configurable palette size K (default 128)

Designed for processing ~270K frames in parallel across multiple CPUs.

Usage:
  # Process one clip (for testing)
  python data/stylize_to_cel.py \
      --input_dir /storage/SSD3/yptsai/data/youtube_anime/frames/CLIP_NAME \
      --output_dir /tmp/stylize_test/CLIP_NAME --K 128

  # Batch process all clips in a directory
  python data/stylize_to_cel.py \
      --batch /storage/SSD3/yptsai/data/youtube_anime/frames \
      --output_root /storage/SSD3/yptsai/data/youtube_anime/cel_frames \
      --K 128 --workers 8
"""
import argparse
import os
from pathlib import Path
from multiprocessing import Pool, get_context

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


# ── Stylization core ─────────────────────────────────────────────────────────

def smooth_frame(img_bgr: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Edge-preserving smoothing to suppress gradients within color blocks."""
    out = img_bgr
    for _ in range(iterations):
        out = cv2.bilateralFilter(out, d=9, sigmaColor=60, sigmaSpace=60)
    return out


def fit_clip_palette(frame_paths: list, K: int, sample_per_frame: int = 5000,
                     max_samples: int = 100_000) -> np.ndarray:
    """Fit a single K-color palette across multiple frames of a clip.

    We sub-sample pixels from a handful of evenly-spaced frames to keep K-means
    fast. The fitted palette is then applied to ALL frames → temporal stability.
    """
    n = len(frame_paths)
    # Sample 5 evenly-spaced frames (or fewer if clip is short)
    n_samples = min(5, n)
    indices = np.linspace(0, n - 1, n_samples).astype(int)

    pixels = []
    for idx in indices:
        img = cv2.imread(str(frame_paths[idx]))
        if img is None:
            continue
        img = smooth_frame(img, iterations=1)
        flat = img.reshape(-1, 3).astype(np.float32)
        n_pick = min(sample_per_frame, len(flat))
        pixels.append(flat[np.random.choice(len(flat), n_pick, replace=False)])

    pixels = np.concatenate(pixels, axis=0)
    if len(pixels) > max_samples:
        pixels = pixels[np.random.choice(len(pixels), max_samples, replace=False)]

    km = MiniBatchKMeans(n_clusters=K, n_init=3, max_iter=100,
                         batch_size=4096, random_state=0)
    km.fit(pixels)
    return km.cluster_centers_  # (K, 3) BGR float32


def apply_palette_vectorised(img_bgr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Snap each pixel to its nearest palette color.
    Vectorised L2 distance via ||p||² + ||c||² − 2·p·c."""
    H, W = img_bgr.shape[:2]
    flat = img_bgr.reshape(-1, 3).astype(np.float32)            # (N, 3)
    p_sq = (flat * flat).sum(axis=1, keepdims=True)              # (N, 1)
    c_sq = (palette * palette).sum(axis=1)                       # (K,)
    pc = flat @ palette.T                                        # (N, K)
    sq_dists = p_sq + c_sq[None, :] - 2 * pc                     # (N, K)
    idx = sq_dists.argmin(axis=1)                                # (N,)
    snapped = palette[idx].reshape(H, W, 3)
    return snapped.astype(np.uint8)


def stylize_clip(input_dir: Path, output_dir: Path, K: int) -> tuple:
    """Stylize an entire clip with a single shared palette.

    Returns:
        (clip_name, n_frames_processed, error_or_None)
    """
    try:
        frame_paths = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
        if len(frame_paths) == 0:
            return (input_dir.name, 0, "no frames")

        # Skip if already done (idempotent batch runs)
        existing = list(output_dir.glob("*.png"))
        if len(existing) >= len(frame_paths):
            return (input_dir.name, len(existing), None)

        output_dir.mkdir(parents=True, exist_ok=True)

        palette = fit_clip_palette(frame_paths, K=K)

        for fp in frame_paths:
            out_path = output_dir / fp.name
            if out_path.exists():
                continue
            img = cv2.imread(str(fp))
            if img is None:
                continue
            smoothed = smooth_frame(img, iterations=2)
            cel = apply_palette_vectorised(smoothed, palette)
            cv2.imwrite(str(out_path), cel)

        # Save palette for inspection
        np.save(str(output_dir / "palette.npy"), palette)

        return (input_dir.name, len(frame_paths), None)
    except Exception as e:
        return (input_dir.name, 0, str(e))


# ── Batch driver ─────────────────────────────────────────────────────────────

def _worker(args):
    input_dir, output_root, K = args
    out_dir = Path(output_root) / input_dir.name
    return stylize_clip(input_dir, out_dir, K=K)


def batch_stylize(input_root: Path, output_root: Path, K: int, workers: int):
    clips = [p for p in sorted(input_root.iterdir()) if p.is_dir()]
    print(f"Found {len(clips)} clips under {input_root}")
    print(f"K={K} colors per palette, {workers} parallel workers")
    print(f"Output: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    args_list = [(c, output_root, K) for c in clips]

    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        ok = 0; fail = 0; total_frames = 0
        for i, (name, n_frames, err) in enumerate(
            pool.imap_unordered(_worker, args_list), 1
        ):
            if err:
                print(f"  [FAIL {i}/{len(clips)}] {name}: {err}")
                fail += 1
            else:
                print(f"  [{i}/{len(clips)}] {name}: {n_frames} frames")
                ok += 1
                total_frames += n_frames

    print(f"\nDone: {ok} ok, {fail} failed, {total_frames:,} frames stylized.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Single clip directory")
    parser.add_argument("--output_dir", help="Output for single clip")
    parser.add_argument("--batch", help="Root containing many clip directories")
    parser.add_argument("--output_root", help="Output root for batch mode")
    parser.add_argument("--K", type=int, default=128, help="Palette size per clip")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.batch:
        assert args.output_root, "--output_root required for batch mode"
        batch_stylize(Path(args.batch), Path(args.output_root), K=args.K, workers=args.workers)
    elif args.input_dir:
        assert args.output_dir, "--output_dir required for single mode"
        name, n, err = stylize_clip(Path(args.input_dir), Path(args.output_dir), K=args.K)
        if err:
            print(f"ERROR: {err}")
        else:
            print(f"Stylized {n} frames → {args.output_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
