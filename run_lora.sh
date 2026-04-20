#!/bin/bash
# Stage 1: LoRA training with palette conditioning.
# Single GPU training (fits on 48GB A6000 with bf16 + 2b model).

VENV=/storage/SSD2/users/yptsai/program/venv
REPO=/storage/SSD2/users/yptsai/program/EdgeEnhancement
LTX_REPO=/storage/SSD2/users/yptsai/program/LTX-Video

MODEL_DIR=/storage/SSD2/yptsai/models/ltx_video
CKPT="$MODEL_DIR/ltx-video-2b-v0.9.1.safetensors"
TEXT_ENC="$MODEL_DIR/pixart_text_encoder"

# Optional: fine-tuned decoder from Stage 0
# DECODER_CKPT=/storage/SSD2/yptsai/exp_result_segmentation/ltx_vae_ft/decoder_final.pt
DECODER_CKPT=""

DATA_ROOT="/path/to/anime/frames"   # <-- change this
OUTPUT_DIR=/storage/SSD2/yptsai/exp_result_segmentation/ltx_lora

GPU=6
LR=1e-4
TOTAL_STEPS=30000
BATCH=1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONPATH="$REPO:$LTX_REPO:$PYTHONPATH"

source "$VENV/bin/activate"

DECODER_ARG=""
if [ -n "$DECODER_CKPT" ]; then
    DECODER_ARG="--decoder_ckpt $DECODER_CKPT"
fi

python "$REPO/model_interp/train_lora.py" \
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
