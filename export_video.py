"""
Export interpolated frames to GIF or MP4.

Usage:
  python export_video.py --frames_dir outputs/snapped --output result.mp4 --fps 12
  python export_video.py --frames_dir outputs/raw --output result.gif --fps 12 --max_dim 512
  python export_video.py --frames_dir outputs/snapped --output side_by_side.mp4 \
      --compare_dir outputs/raw --fps 12
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def load_frames(folder: str) -> list[Image.Image]:
    folder = Path(folder)
    paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No frames found in {folder}")
    return [Image.open(p).convert("RGB") for p in paths]


def resize_frames(frames: list[Image.Image], max_dim: int) -> list[Image.Image]:
    if max_dim <= 0:
        return frames
    out = []
    for f in frames:
        w, h = f.size
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            new_w = int(w * scale) // 2 * 2
            new_h = int(h * scale) // 2 * 2
            f = f.resize((new_w, new_h), Image.LANCZOS)
        out.append(f)
    return out


def side_by_side(frames_a: list[Image.Image], frames_b: list[Image.Image]) -> list[Image.Image]:
    n = min(len(frames_a), len(frames_b))
    out = []
    for a, b in zip(frames_a[:n], frames_b[:n]):
        w = a.width + b.width
        h = max(a.height, b.height)
        canvas = Image.new("RGB", (w, h))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        out.append(canvas)
    return out


def export_gif(frames: list[Image.Image], output: str, fps: int):
    duration_ms = int(1000 / fps)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    size_mb = os.path.getsize(output) / 1e6
    print(f"GIF saved → {output}  ({len(frames)} frames, {size_mb:.1f} MB)")


def export_mp4(frames: list[Image.Image], output: str, fps: int):
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp(prefix="anime_interp_")
    try:
        for i, f in enumerate(frames):
            f.save(os.path.join(tmp_dir, f"{i:06d}.png"))

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "slow",
            output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error:\n{result.stderr[-500:]}")
            return
        size_mb = os.path.getsize(output) / 1e6
        print(f"MP4 saved → {output}  ({len(frames)} frames, {size_mb:.1f} MB)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def export_webp(frames: list[Image.Image], output: str, fps: int):
    duration_ms = int(1000 / fps)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=85,
    )
    size_mb = os.path.getsize(output) / 1e6
    print(f"WebP saved → {output}  ({len(frames)} frames, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True, help="Folder of frame images")
    parser.add_argument("--output", required=True, help="Output path (.mp4 / .gif / .webp)")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max_dim", type=int, default=0, help="Resize long edge to this (0=no resize)")
    parser.add_argument("--compare_dir", default=None, help="Second set of frames for side-by-side")
    args = parser.parse_args()

    frames = load_frames(args.frames_dir)
    print(f"Loaded {len(frames)} frames from {args.frames_dir}")

    if args.compare_dir:
        frames_b = load_frames(args.compare_dir)
        frames = side_by_side(frames, frames_b)
        print(f"Side-by-side: {len(frames)} combined frames")

    if args.max_dim > 0:
        frames = resize_frames(frames, args.max_dim)

    output = args.output
    ext = Path(output).suffix.lower()
    os.makedirs(Path(output).parent, exist_ok=True)

    if ext == ".gif":
        export_gif(frames, output, fps=args.fps)
    elif ext in (".mp4", ".mkv"):
        export_mp4(frames, output, fps=args.fps)
    elif ext == ".webp":
        export_webp(frames, output, fps=args.fps)
    else:
        print(f"Unknown extension '{ext}', defaulting to MP4")
        export_mp4(frames, output + ".mp4", fps=args.fps)


if __name__ == "__main__":
    main()
