"""
Remove shots that are essentially static (no useful motion for interpolation).

Computes mean abs-grayscale-diff over ALL consecutive frame pairs (not a
sub-sample, which can be misled by a single big jump). Two checks:

  1. Static check:        mean_diff < min_mean   → drop (no real motion)
  2. Hidden-cut check:    max > ratio * mean AND mean < hidden_cut_max_mean
                          → flag as likely-missed scene cut (drop or report)

Symlinked frames are followed; the underlying source frames are not touched.

Usage:
  # Dry run (report what would be dropped)
  python data/filter_static.py /storage/SSD3/yptsai/data/youtube_anime/shots --dry_run

  # Actually delete static shots
  python data/filter_static.py /storage/SSD3/yptsai/data/youtube_anime/shots \
      --min_mean 1.5 --workers 16
"""
import argparse
import shutil
from multiprocessing import Pool, get_context
from pathlib import Path

import cv2
import numpy as np


def shot_motion_stats(shot_dir: Path) -> dict:
    """Mean / max / count over ALL consecutive frame pairs."""
    paths = sorted(shot_dir.glob("*.png")) + sorted(shot_dir.glob("*.jpg"))
    if len(paths) < 2:
        return {"shot": shot_dir, "frames": len(paths), "mean": 0.0, "max": 0.0}
    diffs = []
    prev = None
    for fp in paths:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Downsample to 256 long-edge for speed
        h, w = img.shape
        if max(h, w) > 256:
            scale = 256 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        if prev is not None:
            diffs.append(float(np.mean(np.abs(img.astype(np.int16) - prev.astype(np.int16)))))
        prev = img
    if not diffs:
        return {"shot": shot_dir, "frames": len(paths), "mean": 0.0, "max": 0.0}
    arr = np.array(diffs)
    return {
        "shot": shot_dir,
        "frames": len(paths),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def discover_shots(root: Path) -> list:
    """Find all leaf shot directories (containing image frames)."""
    out = []
    def walk(d):
        try:
            children = list(d.iterdir())
        except OSError:
            return
        has_frames = any(
            c.is_file() and c.suffix.lower() in (".png", ".jpg")
            for c in children
        )
        if has_frames:
            out.append(d)
            return
        for sub in sorted(children):
            if sub.is_dir():
                walk(sub)
    walk(root)
    return out


def _worker(args):
    return shot_motion_stats(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Shots root directory (nested or flat)")
    parser.add_argument("--min_mean", type=float, default=1.5,
                        help="Drop shots whose mean consecutive-frame diff < this")
    parser.add_argument("--hidden_cut_ratio", type=float, default=10.0,
                        help="Flag shots where max/mean > ratio AND mean < 5 (likely missed cut)")
    parser.add_argument("--also_drop_hidden_cuts", action="store_true",
                        help="If set, drop shots flagged as likely missed cuts too")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true", help="Report only, don't delete")
    parser.add_argument("--mirror_root", help="Also delete matching dirs under this root "
                        "(e.g. cel_shots if you've already stylized)")
    args = parser.parse_args()

    root = Path(args.root)
    shots = discover_shots(root)
    print(f"Discovered {len(shots)} shot directories under {root}")
    print(f"min_mean={args.min_mean}, hidden_cut_ratio={args.hidden_cut_ratio}")
    print(f"Computing motion stats with {args.workers} workers...")

    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        stats = pool.map(_worker, shots)

    static = []
    hidden_cuts = []
    keep = []
    for s in stats:
        if s["mean"] < args.min_mean:
            static.append(s)
        elif s["mean"] > 0 and s["max"] / s["mean"] > args.hidden_cut_ratio and s["mean"] < 5:
            hidden_cuts.append(s)
        else:
            keep.append(s)

    print(f"\nStatistics:")
    print(f"  KEEP        : {len(keep)} shots")
    print(f"  STATIC      : {len(static)} shots (mean < {args.min_mean})")
    print(f"  HIDDEN_CUT? : {len(hidden_cuts)} shots (max/mean > {args.hidden_cut_ratio} AND mean<5)")

    if static:
        print(f"\nFirst 10 static shots:")
        for s in static[:10]:
            print(f"  {s['shot'].relative_to(root)}: mean={s['mean']:.2f} max={s['max']:.2f} ({s['frames']}f)")

    if hidden_cuts:
        print(f"\nFirst 10 hidden-cut suspects:")
        for s in hidden_cuts[:10]:
            print(f"  {s['shot'].relative_to(root)}: mean={s['mean']:.2f} max={s['max']:.2f} (ratio {s['max']/s['mean']:.1f})")

    to_delete = list(static)
    if args.also_drop_hidden_cuts:
        to_delete += hidden_cuts

    if args.dry_run:
        print(f"\n[DRY RUN] Would delete {len(to_delete)} shots")
        return

    print(f"\nDeleting {len(to_delete)} shots ...")
    mirror_root = Path(args.mirror_root) if args.mirror_root else None
    for s in to_delete:
        shot_path = s["shot"]
        rel = shot_path.relative_to(root)
        # Remove the shot directory
        shutil.rmtree(shot_path, ignore_errors=True)
        # Try to remove the parent video dir if it's now empty
        parent = shot_path.parent
        if parent != root and parent.exists():
            try:
                parent.rmdir()  # only succeeds if empty
            except OSError:
                pass
        # Mirror deletion
        if mirror_root:
            mirror_shot = mirror_root / rel
            if mirror_shot.exists():
                shutil.rmtree(mirror_shot, ignore_errors=True)
                m_parent = mirror_shot.parent
                if m_parent != mirror_root and m_parent.exists():
                    try:
                        m_parent.rmdir()
                    except OSError:
                        pass

    print("Done.")


if __name__ == "__main__":
    main()
