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

def smooth_frame(img_bgr: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Edge-preserving smoothing to suppress gradients within color blocks.

    Iterated bilateral filter — each pass strengthens flat regions while
    preserving edges. 3 iterations empirically removes most subtle gradients
    that would otherwise become 'ripple ring' artifacts after quantisation.
    """
    out = img_bgr
    for _ in range(iterations):
        out = cv2.bilateralFilter(out, d=9, sigmaColor=80, sigmaSpace=80)
    return out


def cleanup_ripples(snapped: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median-filter post-process to remove ripple/concentric-ring artifacts.

    K-means snap can split a single physical region into several thin bands
    of close palette colors (looks like contour rings). Median filtering with
    a small kernel collapses those bands into the dominant local color while
    preserving sharp boundaries between genuinely different regions.
    """
    return cv2.medianBlur(snapped, kernel_size)


def detect_line_mask(img_bgr: np.ndarray,
                     dark_threshold: int = 80,
                     canny_low: int = 30,
                     canny_high: int = 100) -> np.ndarray:
    """Detect anime line-art pixels: dark + on edges.

    Returns a boolean mask where True indicates a line pixel that should be
    preserved as pure black in the cel-stylised output. Without this step the
    K-means quantiser merges thin dark lines into the surrounding color
    region, destroying the cel-line look.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    # Dilate by 1 pixel — anti-aliased line halos sit just off the strict edge
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    line_mask = (edges > 0) & (gray < dark_threshold)
    return line_mask


def _merge_close_colors(palette: np.ndarray, min_dist: float,
                        mode: str = "rgb") -> np.ndarray:
    """Merge palette colors that are too perceptually close.

    Two near-identical palette colors create wavy / fuzzy boundaries because
    K-means assignment flips between them at the edge. This greedy filter
    enforces a minimum perceptual distance between any two kept colors.

    mode="rgb":  Euclidean distance in RGB space (recommended, preserves hue).
                 With threshold 50, isoluminant colors of different hues are
                 kept as distinct (e.g. warm skin tone vs white shirt).
    mode="gray": L1 distance in BT.601 luminance only. Stricter — collapses
                 isoluminant colors. Max ~6 colors at threshold 50.

    Algorithm: greedy keep — sort by brightness, keep each color iff it is
    >= min_dist from EVERY already-kept color (no transitive chaining).
    """
    if len(palette) <= 1 or min_dist <= 0:
        return palette

    # BGR luminance for the brightness ordering
    gray = palette @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    order = np.argsort(-gray)  # brightest first

    kept_colors = []
    for idx in order:
        color = palette[idx]
        if mode == "gray":
            g = gray[idx]
            # Skip if close in grayscale to any kept
            if all(abs(g - (kc @ np.array([0.114, 0.587, 0.299])).item()) >= min_dist
                   for kc in kept_colors):
                kept_colors.append(color)
        else:  # rgb
            if all(np.linalg.norm(color - kc) >= min_dist for kc in kept_colors):
                kept_colors.append(color)

    return np.array(kept_colors, dtype=np.float32)


def fit_clip_palette(frame_paths: list, K: int, sample_per_frame: int = 5000,
                     max_samples: int = 100_000,
                     min_color_dist: float = 0.0,
                     dist_mode: str = "rgb",
                     exclude_lines: bool = True) -> np.ndarray:
    """Fit a single K-color palette across multiple frames of a clip.

    Sub-samples pixels from a handful of evenly-spaced frames to keep K-means
    fast. The fitted palette is then applied to ALL frames → temporal stability.

    If `exclude_lines=True`, line-art pixels are excluded from K-means input
    so the palette captures only color regions (lines are reapplied as black
    after quantisation, which preserves them faithfully).

    If `min_color_dist > 0`, colors closer than the threshold are dropped
    via greedy keep, leaving a ≤K palette of perceptually distinct colors.
    """
    n = len(frame_paths)
    n_samples = min(5, n)
    indices = np.linspace(0, n - 1, n_samples).astype(int)

    pixels = []
    for idx in indices:
        img = cv2.imread(str(frame_paths[idx]))
        if img is None:
            continue
        smoothed = smooth_frame(img, iterations=1)
        if exclude_lines:
            line_mask = detect_line_mask(img)
            sample_pool = smoothed[~line_mask].reshape(-1, 3).astype(np.float32)
        else:
            sample_pool = smoothed.reshape(-1, 3).astype(np.float32)
        if len(sample_pool) == 0:
            continue
        n_pick = min(sample_per_frame, len(sample_pool))
        pixels.append(sample_pool[np.random.choice(len(sample_pool), n_pick, replace=False)])

    pixels = np.concatenate(pixels, axis=0)
    if len(pixels) > max_samples:
        pixels = pixels[np.random.choice(len(pixels), max_samples, replace=False)]

    km = MiniBatchKMeans(n_clusters=K, n_init=3, max_iter=100,
                         batch_size=4096, random_state=0)
    km.fit(pixels)
    palette = km.cluster_centers_.astype(np.float32)

    if min_color_dist > 0:
        palette = _merge_close_colors(palette, min_color_dist, mode=dist_mode)
    return palette


def apply_palette_vectorised(img_bgr: np.ndarray, palette: np.ndarray,
                              line_mask: np.ndarray = None,
                              line_color: tuple = (0, 0, 0)) -> np.ndarray:
    """Snap each pixel to its nearest palette color, optionally overlay lines.

    Vectorised L2 distance via ||p||² + ||c||² − 2·p·c.

    If `line_mask` (HxW bool) is provided, those pixels are forced to
    `line_color` (BGR) — preserves cel line art that K-means would otherwise
    blur into the surrounding region.
    """
    H, W = img_bgr.shape[:2]
    flat = img_bgr.reshape(-1, 3).astype(np.float32)
    p_sq = (flat * flat).sum(axis=1, keepdims=True)
    c_sq = (palette * palette).sum(axis=1)
    pc = flat @ palette.T
    sq_dists = p_sq + c_sq[None, :] - 2 * pc
    idx = sq_dists.argmin(axis=1)
    snapped = palette[idx].reshape(H, W, 3).astype(np.uint8)

    if line_mask is not None:
        snapped[line_mask] = np.array(line_color, dtype=np.uint8)
    return snapped


def stylize_clip(input_dir: Path, output_dir: Path, K: int,
                 min_color_dist: float = 0.0, dist_mode: str = "rgb",
                 preserve_lines: bool = True) -> tuple:
    """Stylize an entire clip with a single shared palette.

    Pipeline per frame:
      1. Detect line-art mask from the original frame (Canny + dark threshold)
      2. Bilateral-smooth the frame (suppresses gradients in color regions)
      3. Snap pixels to the clip's K-means palette (computed once over the clip,
         excluding line pixels for a cleaner palette)
      4. Force line pixels to pure black (preserves cel line art)

    Returns:
        (clip_name, n_frames_processed, error_or_None)
    """
    try:
        frame_paths = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
        if len(frame_paths) == 0:
            return (input_dir.name, 0, "no frames")

        # Skip if already done
        existing = list(output_dir.glob("*.png"))
        if len(existing) >= len(frame_paths):
            return (input_dir.name, len(existing), None)

        output_dir.mkdir(parents=True, exist_ok=True)

        palette = fit_clip_palette(frame_paths, K=K,
                                   min_color_dist=min_color_dist,
                                   dist_mode=dist_mode,
                                   exclude_lines=preserve_lines)

        for fp in frame_paths:
            out_path = output_dir / fp.name
            if out_path.exists():
                continue
            img = cv2.imread(str(fp))
            if img is None:
                continue
            line_mask = detect_line_mask(img) if preserve_lines else None
            smoothed = smooth_frame(img)
            cel = apply_palette_vectorised(smoothed, palette, line_mask=None)
            # Remove K-means ripples BEFORE overlaying lines (so median doesn't
            # smear the black lines into surrounding pixels)
            cel = cleanup_ripples(cel, kernel_size=5)
            if line_mask is not None:
                cel[line_mask] = (0, 0, 0)
            cv2.imwrite(str(out_path), cel)

        np.save(str(output_dir / "palette.npy"), palette)

        return (input_dir.name, len(frame_paths), None)
    except Exception as e:
        return (input_dir.name, 0, str(e))


# ── Batch driver ─────────────────────────────────────────────────────────────

def _worker(args):
    input_dir, out_dir, K, min_color_dist, dist_mode, preserve_lines = args
    return stylize_clip(input_dir, out_dir, K=K,
                        min_color_dist=min_color_dist, dist_mode=dist_mode,
                        preserve_lines=preserve_lines)


def _find_clips_recursive(root: Path) -> list:
    """Find all directories that contain image frames directly.
    Supports both flat (root/clip/frames) and nested (root/video/shot/frames) layouts.
    """
    clips = []
    def walk(d: Path):
        try:
            children = list(d.iterdir())
        except OSError:
            return
        has_frames = any(
            c.is_file() and c.suffix.lower() in (".png", ".jpg", ".jpeg")
            for c in children
        )
        if has_frames:
            clips.append(d)
            return
        for sub in sorted(children):
            if sub.is_dir():
                walk(sub)
    walk(root)
    return clips


def batch_stylize(input_root: Path, output_root: Path, K: int,
                  min_color_dist: float, dist_mode: str, preserve_lines: bool,
                  workers: int):
    clips = _find_clips_recursive(input_root)
    print(f"Found {len(clips)} clip directories under {input_root}")
    print(f"K={K} colors max, min_dist={min_color_dist} ({dist_mode}), "
          f"preserve_lines={preserve_lines}")
    print(f"{workers} parallel workers, output: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    args_list = []
    for c in clips:
        rel = c.relative_to(input_root)
        out_dir = output_root / rel
        args_list.append((c, out_dir, K, min_color_dist, dist_mode, preserve_lines))

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
    parser.add_argument("--K", type=int, default=128,
                        help="Palette size cap per clip (K-means clusters)")
    parser.add_argument("--min_color_dist", type=float, default=0.0,
                        help="Optional: drop palette colors closer than this distance "
                             "to any kept color. Default 0 keeps all K colors. "
                             "Use ~30 for moderate de-duplication.")
    parser.add_argument("--dist_mode", choices=["rgb", "gray"], default="rgb",
                        help="Distance metric for the optional merge step.")
    parser.add_argument("--no_preserve_lines", action="store_true",
                        help="Disable line-art preservation (default ON: detect cel "
                             "lines via Canny+darkness and overlay them as black).")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    preserve_lines = not args.no_preserve_lines

    if args.batch:
        assert args.output_root, "--output_root required for batch mode"
        batch_stylize(Path(args.batch), Path(args.output_root), K=args.K,
                      min_color_dist=args.min_color_dist, dist_mode=args.dist_mode,
                      preserve_lines=preserve_lines, workers=args.workers)
    elif args.input_dir:
        assert args.output_dir, "--output_dir required for single mode"
        name, n, err = stylize_clip(Path(args.input_dir), Path(args.output_dir),
                                     K=args.K, min_color_dist=args.min_color_dist,
                                     dist_mode=args.dist_mode,
                                     preserve_lines=preserve_lines)
        if err:
            print(f"ERROR: {err}")
        else:
            print(f"Stylized {n} frames → {args.output_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
