"""
Build a combined cel-style anime dataset by symlinking clip directories
from multiple sources into one flat directory.

Sources (all RGB/RGBA cel-shaded production frames, no 3D renders):
  - BONES, BONES_2nd        (Japanese anime production cels)
  - Fantasia-Sango          (Chinese animation)
  - Trails-of-cold-steel    (game animation cuts)
  - StudioSeven             (Japanese anime production)
  - BV_Studio               (Western cel-shaded cartoon)
  - AnitaDataset            (Western cel-shaded cartoon)

Excluded:
  - AnimeRun_v2  (3D rendered, not cel)
  - XuanYuanSword, FlatColor  (flat directories of unrelated frames)

Output: /storage/SSD3/yptsai/data/cel_combined/<source>_<clip_id>/  (symlinks)

Usage:
    python data/build_combined_dataset.py --min_frames 17
"""
import argparse
import os
from pathlib import Path

DATASETS = {
    # name → (root_path, depth_to_clip_dir)
    # depth=1 means root/clip/[frames]
    # depth=2 means root/episode/clip/[frames]
    # depth=3 means root/episode/episode/clip/[frames]  (BONES_2nd quirk)
    "BONES":               ("/storage/Internal_NAS/dataset/BONES", 1),
    "BONES_2nd":           ("/storage/Internal_NAS/dataset/BONES_2nd", 3),
    "Fantasia-Sango":      ("/storage/Internal_NAS/dataset/Fantasia-Sango", 1),
    "Trails-of-cold-steel":("/storage/Internal_NAS/dataset/Trails-of-cold-steel", 1),
    "StudioSeven":         ("/storage/Internal_NAS/dataset/StudioSeven", 2),
    "BV_Studio":           ("/storage/Internal_NAS/dataset/BV_Studio", 1),
    "AnitaDataset":        ("/storage/Internal_NAS/dataset/AnitaDataset", 4),  # very deep nesting
    # AnimeRun_v2: 3D-rendered cel-shaded animation. Each clip ships 4 color
    # variants (color_1 .. color_4) of the same motion → built-in data aug.
    "AnimeRun_train":      ("/storage/Internal_NAS/dataset/AnimeRun_v2/train/Frame_Anime", 2),
    "AnimeRun_test":       ("/storage/Internal_NAS/dataset/AnimeRun_v2/test/Frame_Anime", 2),
}

# Subdirs to skip (lineart/grayscale base, not cel-shaded)
SKIP_DIR_NAMES = {"original", "Flow", "Segment", "SegMatching", "contour"}

FRAME_EXTS = {".png", ".jpg", ".jpeg", ".tga"}
OUT_ROOT = Path("/storage/SSD3/yptsai/data/cel_combined")

# Skip directories with these suffixes — they're masks/sheets, not animation clips
SKIP_SUFFIXES = ("_mk", "_sheet", "_eff", "_mask", "_alpha")


def find_clips(root: Path, depth_hint: int, min_frames: int):
    """Find all dirs that contain >=min_frames image files. Recurse if needed."""
    clips = []

    def scan(d: Path, depth: int):
        # Skip mask/sheet folders and known non-cel subdirs (lineart, flow, etc.)
        if any(d.name.endswith(suf) for suf in SKIP_SUFFIXES):
            return
        if d.name in SKIP_DIR_NAMES:
            return
        try:
            children = list(d.iterdir())
        except (PermissionError, OSError):
            return
        frames = [c for c in children if c.is_file() and c.suffix.lower() in FRAME_EXTS]
        if len(frames) >= min_frames:
            clips.append((d, len(frames)))
            return  # don't recurse into a clip dir
        if depth < depth_hint + 2:  # allow some extra slack
            for sub in sorted(children):
                if sub.is_dir():
                    scan(sub, depth + 1)

    scan(root, 0)
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_frames", type=int, default=17)
    parser.add_argument("--clean", action="store_true", help="Remove existing symlinks first")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for old in OUT_ROOT.iterdir():
            if old.is_symlink() or old.is_dir():
                if old.is_symlink():
                    old.unlink()
                else:
                    import shutil
                    shutil.rmtree(old)
        print("Cleaned existing entries.")

    total_clips = 0
    total_frames = 0
    for name, (root_str, depth_hint) in DATASETS.items():
        root = Path(root_str)
        if not root.exists():
            print(f"  [skip] {name}: not found at {root}")
            continue

        clips = find_clips(root, depth_hint, args.min_frames)
        added = 0
        for clip_path, n_frames in clips:
            # Build a unique symlink name from the path components
            rel = clip_path.relative_to(root)
            link_name = f"{name}__" + "_".join(str(rel).split(os.sep))
            link_path = OUT_ROOT / link_name
            if link_path.exists() or link_path.is_symlink():
                continue
            link_path.symlink_to(clip_path)
            added += 1
            total_frames += n_frames
        print(f"  {name:25s}: {added:4d} clips added ({sum(n for _,n in clips):,} frames)")
        total_clips += added

    print()
    print(f"Combined dataset: {total_clips} clips, {total_frames:,} frames")
    print(f"Location: {OUT_ROOT}")


if __name__ == "__main__":
    main()
