"""
Split each downloaded clip into individual SHOTS (continuous camera takes).
Frame-interpolation across scene cuts is meaningless and would corrupt training.

Pipeline (frame-based, since source videos were already deleted):
  1. Walk consecutive frames in each clip directory
  2. Compute HSV histogram chi-square distance between consecutive frames
  3. A spike above the threshold marks a scene cut
  4. Optionally filter shots: minimum motion (avoids static shots),
     minimum/maximum length

Follows ToonCrafter's methodology (PySceneDetect ContentDetector style)
adapted to operate on already-extracted frames.

Output: shots organised as
  {output_root}/
    {clip_name}__shot000_f0001-f0073/
      000089.png ...
    {clip_name}__shot001_f0074-f0156/
      ...
"""
import argparse
from multiprocessing import Pool, get_context
from pathlib import Path

import cv2
import numpy as np


# ── Scene detection ──────────────────────────────────────────────────────────

def hsv_hist(img_bgr: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    """Normalised 3D HSV histogram for one frame."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                     [0, 180, 0, 256, 0, 256])
    h = h.astype(np.float32)
    h /= (h.sum() + 1e-6)
    return h


def detect_cuts(frame_paths: list, threshold: float = 0.5,
                min_shot_length: int = 17) -> list:
    """Return shot boundaries as a list of (start_idx, end_idx) inclusive.

    threshold:        chi-square distance threshold; higher = fewer cuts
                      Empirical: 0.4-0.6 catches most cuts without false positives
    min_shot_length:  drop shots shorter than this (Stage 1 needs >=17 frames)
    """
    n = len(frame_paths)
    if n < min_shot_length:
        return []

    # Compute histograms incrementally to avoid O(N) memory blowup
    prev_hist = None
    cut_indices = [0]  # always start a new shot at 0

    for i, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        # Downsample for speed (histogram is robust to resolution)
        h, w = img.shape[:2]
        if max(h, w) > 256:
            scale = 256 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        hist = hsv_hist(img)

        if prev_hist is not None:
            # Chi-square distance is robust for histogram comparison
            dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR_ALT) / 2
            if dist > threshold:
                cut_indices.append(i)
        prev_hist = hist

    cut_indices.append(n)  # sentinel end

    # Convert cut points to (start, end-inclusive) shot ranges
    shots = []
    for s, e in zip(cut_indices[:-1], cut_indices[1:]):
        length = e - s
        if length >= min_shot_length:
            shots.append((s, e - 1))
    return shots


def shot_motion_score(frame_paths: list) -> float:
    """Quick motion estimate: mean abs-diff between sub-sampled frame pairs."""
    if len(frame_paths) < 3:
        return 0.0
    samples = np.linspace(0, len(frame_paths) - 1, min(5, len(frame_paths))).astype(int)
    diffs = []
    prev = None
    for idx in samples:
        img = cv2.imread(str(frame_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if prev is not None:
            diffs.append(float(np.mean(np.abs(img.astype(np.int16) - prev.astype(np.int16)))))
        prev = img
    return float(np.mean(diffs)) if diffs else 0.0


# ── Process one clip ─────────────────────────────────────────────────────────

def split_clip(input_dir: Path, output_root: Path, threshold: float,
               min_shot_length: int, min_motion: float,
               link_mode: bool = True) -> tuple:
    """Returns (clip_name, n_shots_kept, n_shots_dropped, error_or_None)."""
    try:
        frame_paths = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
        if len(frame_paths) < min_shot_length:
            return (input_dir.name, 0, 0, f"too few frames ({len(frame_paths)})")

        shots = detect_cuts(frame_paths, threshold=threshold,
                            min_shot_length=min_shot_length)
        kept = 0
        dropped_static = 0
        for shot_idx, (s, e) in enumerate(shots):
            shot_paths = frame_paths[s:e + 1]
            if min_motion > 0:
                motion = shot_motion_score(shot_paths)
                if motion < min_motion:
                    dropped_static += 1
                    continue
            shot_name = f"{input_dir.name}__shot{shot_idx:03d}_n{e-s+1}"
            shot_dir = output_root / shot_name
            shot_dir.mkdir(parents=True, exist_ok=True)

            for src in shot_paths:
                dst = shot_dir / src.name
                if dst.exists() or dst.is_symlink():
                    continue
                if link_mode:
                    dst.symlink_to(src)
                else:
                    import shutil
                    shutil.copy2(src, dst)
            kept += 1
        return (input_dir.name, kept, len(shots) - kept, None)
    except Exception as e:
        return (input_dir.name, 0, 0, str(e))


# ── Batch driver ─────────────────────────────────────────────────────────────

def _worker(args):
    return split_clip(*args)


def batch_split(input_root: Path, output_root: Path, threshold: float,
                min_shot_length: int, min_motion: float, workers: int):
    clips = [p for p in sorted(input_root.iterdir()) if p.is_dir()]
    print(f"Found {len(clips)} clips under {input_root}")
    print(f"Cut threshold={threshold}, min length={min_shot_length}, min motion={min_motion}")
    print(f"Output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    args_list = [(c, output_root, threshold, min_shot_length, min_motion, True)
                 for c in clips]

    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        ok = 0; fail = 0; total_shots = 0; total_dropped = 0
        for i, (name, n_kept, n_drop, err) in enumerate(
                pool.imap_unordered(_worker, args_list), 1):
            if err:
                print(f"  [FAIL {i}/{len(clips)}] {name}: {err}")
                fail += 1
            else:
                print(f"  [{i}/{len(clips)}] {name}: {n_kept} shots kept, {n_drop} dropped (motion)")
                ok += 1
                total_shots += n_kept
                total_dropped += n_drop

    print(f"\nDone. {ok} clips, {fail} failed.")
    print(f"Total shots: {total_shots} kept, {total_dropped} dropped (low motion).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Single clip directory (testing)")
    parser.add_argument("--output_dir", help="For single mode")
    parser.add_argument("--batch", help="Root containing many clip dirs")
    parser.add_argument("--output_root", help="Output root for batch mode")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Chi-square histogram distance threshold for cut detection. "
                             "0.3 catches typical anime cuts (verified on TOHO PVs); "
                             "0.5 misses cuts within visually-similar palette ranges.")
    parser.add_argument("--min_shot_length", type=int, default=17)
    parser.add_argument("--min_motion", type=float, default=1.5,
                        help="Mean abs grayscale diff; ~1.5 filters near-static shots")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--copy", action="store_true", help="Copy frames instead of symlinking")
    args = parser.parse_args()

    if args.batch:
        assert args.output_root, "--output_root required"
        batch_split(Path(args.batch), Path(args.output_root),
                    threshold=args.threshold,
                    min_shot_length=args.min_shot_length,
                    min_motion=args.min_motion,
                    workers=args.workers)
    elif args.input_dir:
        assert args.output_dir, "--output_dir required"
        name, kept, drop, err = split_clip(
            Path(args.input_dir), Path(args.output_dir),
            threshold=args.threshold, min_shot_length=args.min_shot_length,
            min_motion=args.min_motion, link_mode=not args.copy,
        )
        if err:
            print(f"ERROR: {err}")
        else:
            print(f"Split into {kept} shots ({drop} dropped for low motion)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
