"""
Inference: given 5 keyframes + palette → generate 35 intermediate frames.

Usage:
  python model_interp/infer.py \
      --ckpt_path /path/to/ltx.safetensors \
      --lora_dir runs/lora_train/final \
      --keyframe_dir /path/to/5_keyframes/ \
      --palette_ref  /path/to/reference.png \
      --output_dir   outputs/
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LTX-Video"))

from peft import PeftModel
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.schedulers.rf import RectifiedFlowScheduler

from palette_encoder import PaletteEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--text_encoder_path", default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--keyframe_dir", required=True, help="Folder with exactly 5 keyframes")
    parser.add_argument("--palette_ref", required=True, help="Image to extract palette from")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--num_frames", type=int, default=35)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--palette_k", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", default="bf16")
    return parser.parse_args()


def load_image(path: str, height: int, width: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)
    t = TVF.to_tensor(img)  # (3, H, W) [0,1]
    return t * 2.0 - 1.0   # [-1, 1]


def extract_palette(image_path: str, k: int = 16) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).reshape(-1, 3).astype(np.float32) / 255.0
    idx = np.random.choice(len(pixels), min(50000, len(pixels)), replace=False)
    pixels = pixels[idx]
    km = KMeans(n_clusters=k, n_init=5, random_state=0)
    km.fit(pixels)
    return torch.tensor(km.cluster_centers_, dtype=torch.float32)  # (K, 3)


def hard_palette_snap(frames: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """
    Args:
        frames:  (T, H, W, 3) in [0, 1]
        palette: (K, 3)       in [0, 1]
    Returns:
        snapped: (T, H, W, 3) in [0, 1]
    """
    T, H, W, _ = frames.shape
    K = palette.shape[0]
    flat = frames.reshape(-1, 3)          # (T*H*W, 3)
    diff = flat.unsqueeze(1) - palette.unsqueeze(0)   # (T*H*W, K, 3)
    idx = diff.norm(dim=2).argmin(dim=1)              # (T*H*W,)
    snapped = palette[idx].reshape(T, H, W, 3)
    return snapped


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ── Load models ──────────────────────────────────────────────────────
    print("Loading VAE...")
    vae = CausalVideoAutoencoder.from_pretrained(args.ckpt_path).to(device, dtype=dtype)
    vae.eval()

    print("Loading transformer + LoRA...")
    base_transformer = Transformer3DModel.from_pretrained(args.ckpt_path).to(dtype=dtype)
    transformer = PeftModel.from_pretrained(base_transformer, args.lora_dir).to(device)
    transformer.eval()

    print("Loading text encoder...")
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder_path, subfolder="text_encoder"
    ).to(device, dtype=dtype)
    text_encoder.eval()

    text_dim = text_encoder.config.d_model
    palette_enc = PaletteEncoder(palette_k=args.palette_k, token_dim=text_dim).to(device, dtype=dtype)
    pal_ckpt = os.path.join(args.lora_dir, "palette_encoder.pt")
    palette_enc.load_state_dict(torch.load(pal_ckpt, map_location=device))
    palette_enc.eval()

    patchifier = SymmetricPatchifier(patch_size=1)
    scheduler = RectifiedFlowScheduler.from_pretrained(args.ckpt_path)

    # ── Prepare inputs ───────────────────────────────────────────────────
    kf_paths = sorted(Path(args.keyframe_dir).glob("*.png")) + sorted(Path(args.keyframe_dir).glob("*.jpg"))
    assert len(kf_paths) == 5, f"Expected 5 keyframes, got {len(kf_paths)}"
    keyframes = [load_image(str(p), args.height, args.width) for p in kf_paths]

    palette = extract_palette(args.palette_ref, k=args.palette_k).to(device, dtype=dtype)

    # Keyframe positions spread over num_frames
    T = args.num_frames
    kf_positions = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    # Encode text
    tokens = tokenizer(
        "anime cel-shaded animation",
        max_length=256, padding="max_length", truncation=True, return_tensors="pt"
    )
    text_hidden = text_encoder(
        input_ids=tokens.input_ids.to(device),
        attention_mask=tokens.attention_mask.to(device),
    ).last_hidden_state  # (1, L, D)

    palette_tokens = palette_enc(palette.unsqueeze(0))  # (1, K, D)
    encoder_hidden = torch.cat([text_hidden, palette_tokens], dim=1)  # (1, L+K, D)
    text_mask = tokens.attention_mask.to(device)
    pal_mask = torch.ones(1, args.palette_k, device=device, dtype=text_mask.dtype)
    encoder_mask = torch.cat([text_mask, pal_mask], dim=1)

    # Encode keyframes to latents
    kf_tensor = torch.stack(keyframes, dim=0).unsqueeze(0)  # (1, 5, 3, H, W)
    # Build full frame sequence: noisy everywhere, clean at keyframe positions
    noise_frames = torch.randn(1, T, 3, args.height, args.width, device=device, dtype=dtype) * 0.1

    with torch.no_grad():
        kf_bcthw = kf_tensor.permute(0, 2, 1, 3, 4).to(device, dtype=dtype)  # (1, 3, 5, H, W)
        kf_latents = vae_encode(kf_bcthw, vae)  # (1, latent_C, 5', H', W')

    # ── Denoising loop ────────────────────────────────────────────────────
    latent_T = max(1, (T - 1) // 4 + 1)  # temporal stride = 4
    latent_H, latent_W = args.height // 8, args.width // 8
    latent_C = kf_latents.shape[1]

    latents = torch.randn(1, latent_C, latent_T, latent_H, latent_W, device=device, dtype=dtype)

    # Inject known keyframe latents at their positions
    kf_lat_positions = [min(int(round(p * latent_T / T)), latent_T - 1) for p in kf_positions]
    for i, lat_idx in enumerate(kf_lat_positions):
        src_t = min(i, kf_latents.shape[2] - 1)
        latents[:, :, lat_idx] = kf_latents[:, :, src_t]

    timesteps = scheduler.timesteps if hasattr(scheduler, "timesteps") else torch.linspace(1, 0, args.steps)

    for i, t in enumerate(timesteps):
        t_val = t.item() if hasattr(t, "item") else float(t)
        t_batch = torch.tensor([t_val * 1000], device=device, dtype=torch.long)

        patches, coords = patchifier.patchify(latents)
        frac_coords = coords.to(torch.float32)
        frac_coords[:, 0] = frac_coords[:, 0] * (1.0 / 24.0)

        pred = transformer(
            hidden_states=patches,
            indices_grid=frac_coords,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            timestep=t_batch,
            return_dict=False,
        )[0]

        pred_latents = patchifier.unpatchify(pred, latent_H, latent_W, latent_C)

        # RF step: x_{t-1} = x_t - (dt) * (x_t - x0_pred)
        if i < len(timesteps) - 1:
            dt = t_val - (timesteps[i + 1].item() if hasattr(timesteps[i + 1], "item") else float(timesteps[i + 1]))
        else:
            dt = t_val
        latents = latents - dt * (latents - pred_latents)

        # Re-inject known keyframes at each step
        for ki, lat_idx in enumerate(kf_lat_positions):
            src_t = min(ki, kf_latents.shape[2] - 1)
            latents[:, :, lat_idx] = kf_latents[:, :, src_t]

        if (i + 1) % 10 == 0:
            print(f"  step {i+1}/{len(timesteps)}")

    # ── Decode & snap ─────────────────────────────────────────────────────
    with torch.no_grad():
        pixels = vae_decode(latents.float(), vae, is_video=True)  # (1, 3, T', H, W)

    pixels = pixels[0].permute(1, 2, 3, 0).clamp(-1, 1)  # (T', H, W, 3)
    pixels = (pixels + 1.0) / 2.0  # [0, 1]

    palette_cpu = palette.cpu().float()
    snapped = hard_palette_snap(pixels.cpu().float(), palette_cpu)

    # Save frames
    raw_dir = os.path.join(args.output_dir, "raw")
    snap_dir = os.path.join(args.output_dir, "snapped")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)

    for i, (raw_frame, snap_frame) in enumerate(zip(pixels.cpu().float(), snapped)):
        raw_img = Image.fromarray((raw_frame.numpy() * 255).astype(np.uint8))
        raw_img.save(os.path.join(raw_dir, f"frame_{i:04d}.png"))

        snap_img = Image.fromarray((snap_frame.numpy() * 255).astype(np.uint8))
        snap_img.save(os.path.join(snap_dir, f"frame_{i:04d}.png"))

    print(f"Saved {len(pixels)} raw frames → {raw_dir}")
    print(f"Saved {len(snapped)} snapped frames → {snap_dir}")


if __name__ == "__main__":
    main()
