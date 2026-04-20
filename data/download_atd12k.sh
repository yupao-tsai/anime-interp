#!/bin/bash
# Download ATD-12K (Animation Triplet Dataset) via gdown.
# Paper: AnimeInterp (CVPR 2021)
# Content: 12,000 anime frame triplets from 30 animation films
# Use: Stage 0 VAE decoder fine-tuning (single-frame quality)
#      Also usable for Stage 1 if combined with yt-dlp data (3-frame clips are minimal)
#
# Storage: ~3-5 GB compressed, ~8-10 GB extracted
# Output: /storage/SSD3/yptsai/data/atd12k/

VENV=/storage/SSD2/users/yptsai/program/venv
OUT_DIR=/storage/SSD3/yptsai/data/atd12k

source "$VENV/bin/activate"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "=== Downloading ATD-12K ==="
echo "  Source: Google Drive (AnimeInterp CVPR 2021)"
echo "  Output: $OUT_DIR"
echo ""

ZIP_FILE="$OUT_DIR/ATD-12K.zip"
# OneDrive link (Google Drive quota often exceeded due to popularity)
ONEDRIVE_URL="https://entuedu-my.sharepoint.com/:u:/g/personal/siyao002_e_ntu_edu_sg/EY3SG0-IajxKj9HMPz__zOMBvyJdrA-SlwpyHYFkDsQtng?e=q7nGlu&download=1"

if [ -f "$ZIP_FILE" ]; then
    echo "ZIP already exists, skipping download."
else
    echo "  Downloading ~18.6 GB via OneDrive, this will take ~5-15 minutes..."
    wget -c --show-progress -O "$ZIP_FILE" "$ONEDRIVE_URL" \
        || { echo "OneDrive failed. Try Google Drive manually:"; \
             echo "  https://drive.google.com/file/d/1XBDuiEgdd6c0S4OXLF4QvgSn_XNPwc-g/view"; \
             exit 1; }
fi

echo ""
echo "=== Extracting ==="
unzip -q "$ZIP_FILE" -d "$OUT_DIR" && echo "Extracted OK"

echo ""
echo "=== Dataset summary ==="
python3 - <<'EOF'
from pathlib import Path
import sys

root = Path("/storage/SSD3/yptsai/data/atd12k")
for split in ["train", "test"]:
    d = root / split
    if not d.exists():
        continue
    clips = [c for c in d.iterdir() if c.is_dir()]
    frames_per = [len(list(c.glob("*.png"))) for c in clips]
    total = sum(frames_per)
    print(f"  {split}: {len(clips)} triplets, {total} frames total")
EOF

echo ""
echo "Done. Use DATA_ROOT=$OUT_DIR/train for Stage 0 VAE training."
echo "  python train_vae.py --data_root $OUT_DIR/train --num_frames 1"
