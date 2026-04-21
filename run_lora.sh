#!/bin/bash
# Stage 1: LoRA training with palette conditioning.
# Uses LTX-Video 19b model (local). Requires ~48GB VRAM with bf16.

VENV=/storage/SSD2/users/yptsai/program/venv
REPO=/storage/SSD2/users/yptsai/program/anime_interp
LTX_REPO=/storage/SSD2/users/yptsai/program/LTX-Video

# LTX-Video 2b (v0.9.6-dev) — 6.3GB, compatible with our cloned LTX-Video repo.
# The 19b local checkpoint is LTX-2 architecture (has audio cross-attention)
# and incompatible with the cloned Transformer3DModel code path.
CKPT=/storage/SSD2/yptsai/models/ltx_video/ltxv-2b-0.9.6-dev-04-25.safetensors
TEXT_ENC="PixArt-alpha/PixArt-XL-2-1024-MS"   # downloads from HF on first use

# Stage 0 VAE decoder checkpoint (shared across 2b/13b — same CausalVideoAutoencoder)
DECODER_CKPT=$REPO/runs/vae_finetune/decoder_step_8000.pt

# Combined cel dataset: 678 clips, 38,709 frames from 8 cel-style sources.
DATA_ROOT=/storage/SSD3/yptsai/data/cel_combined
OUTPUT_DIR=$REPO/runs/lora_train

GPU=1
LR=1e-4
TOTAL_STEPS=30000
BATCH=1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONPATH="$LTX_REPO:$PYTHONPATH"

source "$VENV/bin/activate"

mkdir -p "$OUTPUT_DIR"

DECODER_ARG=""
if [ -n "$DECODER_CKPT" ]; then
    DECODER_ARG="--decoder_ckpt $DECODER_CKPT"
fi

python "$REPO/train_lora.py" \
    --ckpt_path "$CKPT" \
    --text_encoder_path "$TEXT_ENC" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    $DECODER_ARG \
    --lr "$LR" \
    --total_steps "$TOTAL_STEPS" \
    --batch_size "$BATCH" \
    --height 256 \
    --width 384 \
    --num_frames 17 \
    --palette_k 16 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --w_palette 0.3 \
    --save_interval 5000 \
    --log_interval 50 \
    --num_workers 4 \
    --gpu "$GPU" \
    --precision bf16
