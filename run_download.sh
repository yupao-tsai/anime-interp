#!/bin/bash
# Download LTX-Video 2b model and T5 text encoder from HuggingFace.
# Run this once before training.

VENV=/storage/SSD2/users/yptsai/program/venv
MODEL_DIR=/storage/SSD2/yptsai/models/ltx_video

mkdir -p "$MODEL_DIR"

source "$VENV/bin/activate"

echo "=== Downloading LTX-Video 2b (v0.9.1) ==="
python - <<EOF
from huggingface_hub import hf_hub_download
import os

out = "$MODEL_DIR/ltx-video-2b-v0.9.1.safetensors"
if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    path = hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename="ltx-video-2b-v0.9.1.safetensors",
        local_dir="$MODEL_DIR",
    )
    print(f"Downloaded → {path}")
EOF

echo "=== Downloading T5 text encoder (PixArt-alpha/PixArt-XL-2-1024-MS) ==="
python - <<EOF
from huggingface_hub import snapshot_download
import os

out = "$MODEL_DIR/pixart_text_encoder"
if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    path = snapshot_download(
        repo_id="PixArt-alpha/PixArt-XL-2-1024-MS",
        allow_patterns=["text_encoder/**", "tokenizer/**"],
        local_dir=out,
    )
    print(f"Downloaded → {path}")
EOF

echo ""
echo "Done. Model files at: $MODEL_DIR"
echo "Pass these to training scripts:"
echo "  --ckpt_path $MODEL_DIR/ltx-video-2b-v0.9.1.safetensors"
echo "  --text_encoder_path $MODEL_DIR/pixart_text_encoder"
