#!/bin/bash
# Stage 0: Fine-tune VAE decoder on anime frames.
# Encoder frozen, decoder trained for sharp edges and flat colors.
# ~2-3 hours for 10k steps on a single A6000.

VENV=/storage/SSD2/users/yptsai/program/venv
REPO=/storage/SSD2/users/yptsai/program/anime_interp
LTX_REPO=/storage/SSD2/users/yptsai/program/LTX-Video

CKPT=/storage/SSD2/program/LTX-2/models/checkpoints/ltx-2-19b-dev.safetensors

DATA_ROOT="/storage/SSD2/users/ryan/dataset/FlatColorData/GT"
OUTPUT_DIR=$REPO/runs/vae_finetune

GPU=0
LR=1e-4
TOTAL_STEPS=10000
BATCH=4

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONPATH="$LTX_REPO:$PYTHONPATH"

source "$VENV/bin/activate"

mkdir -p "$OUTPUT_DIR"

python "$REPO/train_vae.py" \
    --ckpt_path "$CKPT" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --total_steps "$TOTAL_STEPS" \
    --batch_size "$BATCH" \
    --height 256 \
    --width 384 \
    --num_frames 1 \
    --w_l1 1.0 \
    --w_perc 0.1 \
    --w_edge 0.5 \
    --save_interval 2000 \
    --log_interval 50 \
    --gpu "$GPU"
