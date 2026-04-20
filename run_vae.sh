#!/bin/bash
# Stage 0: Fine-tune VAE decoder on anime frames.
# Run on a single GPU; takes ~2-3 hours for 10k steps.

VENV=/storage/SSD2/users/yptsai/program/venv
REPO=/storage/SSD2/users/yptsai/program/EdgeEnhancement
LTX_REPO=/storage/SSD2/users/yptsai/program/LTX-Video

MODEL_DIR=/storage/SSD2/yptsai/models/ltx_video
CKPT="$MODEL_DIR/ltx-video-2b-v0.9.1.safetensors"

DATA_ROOT="/path/to/anime/frames"   # <-- change this
OUTPUT_DIR=/storage/SSD2/yptsai/exp_result_segmentation/ltx_vae_ft

GPU=6
LR=1e-4
TOTAL_STEPS=10000
BATCH=4

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONPATH="$REPO:$LTX_REPO:$PYTHONPATH"

source "$VENV/bin/activate"

python "$REPO/model_interp/train_vae.py" \
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
    --log_interval 100 \
    --gpu "$GPU"
