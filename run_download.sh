#!/bin/bash
# Download T5 text encoder from HuggingFace.
# The LTX-Video 19b model is already available locally at:
#   /storage/SSD2/program/LTX-2/models/checkpoints/ltx-2-19b-dev.safetensors
#
# Run this once before training to cache the text encoder.

VENV=/storage/SSD2/users/yptsai/program/venv

source "$VENV/bin/activate"

echo "=== Downloading T5 text encoder (PixArt-alpha/PixArt-XL-2-1024-MS) ==="
echo "    This will be cached in ~/.cache/huggingface/"
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="PixArt-alpha/PixArt-XL-2-1024-MS",
    allow_patterns=["text_encoder/**", "tokenizer/**"],
)
print("Text encoder cached.")
EOF

echo ""
echo "Done. Model is ready:"
echo "  CKPT: /storage/SSD2/program/LTX-2/models/checkpoints/ltx-2-19b-dev.safetensors"
echo "  TEXT: PixArt-alpha/PixArt-XL-2-1024-MS (HuggingFace cache)"
