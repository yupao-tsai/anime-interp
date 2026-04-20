# Anime Frame Interpolation with LTX-Video + Palette Conditioning

Given 5 sparse keyframes and a color palette reference, generate smooth intermediate frames while preserving the flat-color anime aesthetic.

## How It Works

Two-stage training built on top of **LTX-Video 19b**:

### Stage 0 — VAE Decoder Fine-tuning (`train_vae.py`)
Fine-tune the LTX-Video VAE decoder (encoder frozen) on anime frames.

| Loss component | Weight | Purpose |
|---|---|---|
| L1 reconstruction | 1.0 | Pixel fidelity |
| VGG perceptual | 0.1 | Texture / feature matching |
| Sobel edge | 0.5 | Sharp line art |

This step is **required** — without it, the VAE will blur flat colors and soften line edges.

### Stage 1 — LoRA + Palette Conditioning (`train_lora.py`)
Fine-tune the LTX-Video transformer with LoRA. A small MLP (`PaletteEncoder`) maps K dominant colors → K cross-attention tokens, appended to the T5 text tokens.

**Training task**: masked-frame prediction. Given a clip of T frames with 5 keyframes known, predict all T frames. Loss on non-keyframe positions only.

```
palette_tokens = PaletteEncoder(palette)   # (B, K, D)
text_tokens    = T5Encoder("anime cel-shaded animation")
cond_tokens    = cat([text_tokens, palette_tokens], dim=1)
# → fed into LTX cross-attention; no architecture change needed
```

**LoRA targets**: `to_q`, `to_k`, `to_v`, `to_out.0` in all attention blocks (both spatial and temporal). Rank 32, alpha 16.

**Training losses**:
- L1 reconstruction on predicted frames
- Palette adhesion loss: per-pixel nearest-color snap, then L1

### Inference
1. Encode 5 keyframes → latents
2. Inject at positions `[0, T/4, T/2, 3T/4, T-1]`
3. Denoise (RF scheduler, 50 steps), re-injecting keyframe latents every step
4. Decode → hard palette snap (nearest-neighbor color quantization)

---

## Setup

### 1. Clone LTX-Video repository
```bash
git clone https://github.com/Lightricks/LTX-Video /path/to/LTX-Video
pip install -e /path/to/LTX-Video
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download text encoder
```bash
bash run_download.sh
```

The LTX-Video 19b model is already available locally at:
```
/storage/SSD2/program/LTX-2/models/checkpoints/ltx-2-19b-dev.safetensors
```

---

## Data Preparation

### Extract frames from videos
```bash
# Single video
python data/preprocess.py --input anime.mp4 --output data/frames/clip001 --fps 12

# Batch (directory of videos)
python data/preprocess.py --input_dir /videos --output_dir data/frames --fps 12

# Verify dataset
python data/preprocess.py --verify data/frames --min_frames 33
```

Frame naming: sequential PNGs (`000001.png`, `000002.png`, ...).  
The dataset loader discovers all subdirectories with ≥ `num_frames` images.

---

## Training

### Stage 0: VAE Decoder Fine-tuning
```bash
# Edit DATA_ROOT in the script first
bash run_vae.sh
```

Or manually:
```bash
export PYTHONPATH=/path/to/LTX-Video:$PYTHONPATH

python train_vae.py \
    --ckpt_path /storage/SSD2/program/LTX-2/models/checkpoints/ltx-2-19b-dev.safetensors \
    --data_root data/frames \
    --output_dir runs/vae_finetune \
    --total_steps 10000 --batch_size 4 --gpu 0
```

Checkpoint saved as `runs/vae_finetune/decoder_final.pt`.

### Stage 1: LoRA + Palette Training
```bash
# Edit DATA_ROOT in the script first
bash run_lora.sh
```

Or manually:
```bash
python train_lora.py \
    --ckpt_path /storage/SSD2/program/LTX-2/models/checkpoints/ltx-2-19b-dev.safetensors \
    --decoder_ckpt runs/vae_finetune/decoder_final.pt \
    --data_root data/frames \
    --output_dir runs/lora_train \
    --total_steps 30000 --lora_rank 32 --gpu 0
```

Monitor training:
```bash
tensorboard --logdir runs/
```

---

## Inference

```bash
python infer.py \
    --lora_dir runs/lora_train/final \
    --keyframe_dir /path/to/5_keyframes/ \
    --palette_ref /path/to/reference.png \
    --output_dir outputs/my_clip \
    --num_frames 35 \
    --export_mp4 --fps 12
```

**Outputs:**
```
outputs/my_clip/
├── raw/          ← decoded frames before palette snap
├── snapped/      ← hard-palette-quantized frames
└── output.mp4    ← (if --export_mp4)
```

---

## Evaluation

```bash
# Compare against ground-truth frames
python eval.py \
    --pred outputs/my_clip/snapped \
    --gt /path/to/gt_frames \
    --palette_ref /path/to/reference.png \
    --output_json results.json

# Palette consistency only (no GT needed)
python eval.py \
    --pred outputs/my_clip/snapped \
    --palette_ref /path/to/reference.png \
    --palette_only
```

**Metrics:**
| Metric | Description |
|---|---|
| PSNR (dB) | Pixel fidelity vs ground truth |
| SSIM | Structural similarity vs ground truth |
| Mean palette dist | Average nearest-color distance [0,1] |
| Snap PSNR | PSNR of hard-snapped vs raw output |
| Temporal L1 | Frame-to-frame smoothness |

---

## Export

```bash
# MP4
python export_video.py --frames_dir outputs/snapped --output result.mp4 --fps 12

# GIF (resized)
python export_video.py --frames_dir outputs/snapped --output result.gif --fps 12 --max_dim 512

# Side-by-side comparison (raw vs snapped)
python export_video.py \
    --frames_dir outputs/snapped \
    --compare_dir outputs/raw \
    --output comparison.mp4 --fps 12
```

---

## Key Design Decisions

- **Hard palette snap at inference only** — not a differentiable loss during training, just the final post-process
- **Palette extracted from reference sheet** (character color reference), not from each frame dynamically
- **LoRA rank 32** minimum — anime style shift requires higher capacity than typical fine-tuning
- **Temporal LoRA is essential** — spatial-only LoRA fails to learn anime motion timing
- **VAE decoder fine-tune is non-optional** — the pretrained VAE blurs flat cel colors

---

## Project Structure

```
anime_interp/
├── data/
│   └── preprocess.py    # video → frames extraction
├── dataset.py           # AnimeClipDataset
├── eval.py              # PSNR, SSIM, palette consistency
├── export_video.py      # frames → MP4 / GIF / WebP
├── infer.py             # inference pipeline
├── palette_encoder.py   # PaletteEncoder MLP
├── train_lora.py        # Stage 1 training
├── train_vae.py         # Stage 0 VAE decoder fine-tuning
├── run_download.sh      # download T5 text encoder
├── run_lora.sh          # Stage 1 training launcher
└── run_vae.sh           # Stage 0 training launcher
```
