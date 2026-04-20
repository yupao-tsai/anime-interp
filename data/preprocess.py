"""
Data preprocessing: extract frames from anime videos for training.

Usage:
  # Extract frames from a single video
  python data/preprocess.py --input anime.mp4 --output data/frames/clip001

  # Batch process a directory of videos
  python data/preprocess.py --input_dir /path/to/videos --output_dir data/frames --fps 12

  # Verify extracted dataset
  python data/preprocess.py --verify data/frames --min_frames 33
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def extract_frames(video_path: str, out_dir: str, fps: int = 12, max_dim: int = 768) -> int:
    """Extract frames from a video at the given fps. Returns frame count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scale to max_dim on the long side, keep divisible by 32
    scale_filter = (
        f"scale='if(gt(iw,ih),min({max_dim},iw),-2)':'if(gt(iw,ih),-2,min({max_dim},ih))',"
        f"crop='trunc(iw/32)*32':'trunc(ih/32)*32'"
    )

    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [skip] ffprobe failed for {video_path}")
        return 0

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps},{scale_filter}",
        "-q:v", "2",
        str(out_dir / "%06d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] ffmpeg failed:\n{result.stderr[-500:]}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return 0

    count = len(list(out_dir.glob("*.png")))
    return count


def batch_extract(input_dir: str, output_dir: str, fps: int, min_frames: int, max_dim: int):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
    videos = [p for p in sorted(input_dir.rglob("*")) if p.suffix.lower() in video_exts]

    if not videos:
        print(f"No videos found under {input_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} videos. Extracting at {fps} fps...")
    ok, skip = 0, 0
    for i, video in enumerate(videos, 1):
        clip_name = video.stem
        out_dir = output_dir / clip_name
        if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= min_frames:
            print(f"  [{i}/{len(videos)}] {video.name} — already extracted, skipping")
            ok += 1
            continue

        print(f"  [{i}/{len(videos)}] {video.name} ...", end=" ", flush=True)
        count = extract_frames(str(video), str(out_dir), fps=fps, max_dim=max_dim)
        if count >= min_frames:
            print(f"→ {count} frames ✓")
            ok += 1
        else:
            print(f"→ {count} frames (< {min_frames}, removed)")
            shutil.rmtree(out_dir, ignore_errors=True)
            skip += 1

    print(f"\nDone: {ok} usable clips, {skip} skipped.")


def verify_dataset(data_root: str, min_frames: int):
    data_root = Path(data_root)
    clips = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    ok = [c for c in clips if len(list(c.glob("*.png"))) >= min_frames]
    short = [c for c in clips if len(list(c.glob("*.png"))) < min_frames]

    print(f"Dataset: {data_root}")
    print(f"  Clips with >= {min_frames} frames: {len(ok)}")
    print(f"  Clips with <  {min_frames} frames: {len(short)}")
    if short:
        for c in short[:5]:
            n = len(list(c.glob("*.png")))
            print(f"    {c.name}: {n} frames")
    if len(ok) == 0:
        print("WARNING: No usable clips found!")
    return len(ok)


def main():
    parser = argparse.ArgumentParser(description="Preprocess anime videos into frame datasets")
    subparsers = parser.add_subparsers(dest="cmd")

    p_single = subparsers.add_parser("extract", help="Extract frames from a single video")
    p_single.add_argument("--input", required=True)
    p_single.add_argument("--output", required=True)
    p_single.add_argument("--fps", type=int, default=12)
    p_single.add_argument("--max_dim", type=int, default=768)

    p_batch = subparsers.add_parser("batch", help="Batch extract from directory of videos")
    p_batch.add_argument("--input_dir", required=True)
    p_batch.add_argument("--output_dir", required=True)
    p_batch.add_argument("--fps", type=int, default=12)
    p_batch.add_argument("--min_frames", type=int, default=33)
    p_batch.add_argument("--max_dim", type=int, default=768)

    p_verify = subparsers.add_parser("verify", help="Verify extracted dataset")
    p_verify.add_argument("--data_root", required=True)
    p_verify.add_argument("--min_frames", type=int, default=33)

    # Legacy flat-args mode for convenience
    parser.add_argument("--input", help="Single video file")
    parser.add_argument("--output", help="Output directory for single video")
    parser.add_argument("--input_dir", help="Directory of videos")
    parser.add_argument("--output_dir", help="Output directory for batch")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--min_frames", type=int, default=33)
    parser.add_argument("--max_dim", type=int, default=768)
    parser.add_argument("--verify", metavar="DATA_ROOT", help="Verify dataset at path")

    args = parser.parse_args()

    if args.cmd == "extract" or args.input:
        inp = args.input
        out = args.output
        fps = args.fps
        max_dim = args.max_dim
        count = extract_frames(inp, out, fps=fps, max_dim=max_dim)
        print(f"Extracted {count} frames → {out}")

    elif args.cmd == "batch" or args.input_dir:
        batch_extract(
            args.input_dir, args.output_dir,
            fps=args.fps, min_frames=args.min_frames, max_dim=args.max_dim,
        )

    elif args.cmd == "verify" or args.verify:
        root = args.verify if args.verify else args.data_root
        verify_dataset(root, min_frames=args.min_frames)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
