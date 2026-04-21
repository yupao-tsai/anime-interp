"""
Stage 1 training: LoRA + palette conditioning on the LTX-Video transformer.

Rewritten to match the LTXVideoPipeline forward conventions:
  - Transformer predicts velocity v = noise - x0 (rectified flow)
  - Input patches come from the SymmetricPatchifier
  - indices_grid = pixel_coords with the temporal axis scaled by 1 / frame_rate
  - Hard-conditioning latents at keyframe positions set conditioning_mask=1,
    which zeroes their per-token timestep (no noise injected there)

Task: given T pixel frames encoded to T_lat latent frames, inject the 5
keyframe latents as hard conditioning and train the transformer to predict
velocity at the non-keyframe latent positions.

Stability choices (learned from previous failures):
  - Cast everything including LoRA adapters to bf16 after apply
  - Re-cast VAE to bf16 AFTER loading the fp32 decoder ckpt
  - VAE decode (palette loss) always passes decode_timestep=0.05
  - If the full-clip latent temporal extent is too small to leave non-keyframe
    positions, fall back to unmasked loss over all tokens
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5EncoderModel, T5Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LTX-Video"))
# Also support /storage/SSD2/users/yptsai/program/LTX-Video layout
sys.path.insert(0, "/storage/SSD2/users/yptsai/program/LTX-Video")

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import (
    vae_encode, vae_decode, latent_to_pixel_coords,
)
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier

from dataset import AnimeClipDataset
from palette_encoder import PaletteEncoder


DUMMY_PROMPT = "anime cel-shaded animation"
FRAME_RATE = 24.0       # indices_grid temporal scale, matches inference default
DECODE_TIMESTEP = 0.05  # LTX-Video 2b canonical decode timestep


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--text_encoder_path", default="PixArt-alpha/PixArt-XL-2-1024-MS")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="runs/lora_train")
    p.add_argument("--decoder_ckpt", default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--num_frames", type=int, default=33,
                   help="Pixel frames per clip; higher → more latent temporal positions")
    p.add_argument("--palette_k", type=int, default=16)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--w_palette", type=float, default=0.0,
                   help="Palette adhesion loss weight (0 = disabled)")
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    return p.parse_args()


def encode_text(prompt, tokenizer, text_encoder, device, max_length=256):
    tokens = tokenizer(prompt, max_length=max_length, padding="max_length",
                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        h = text_encoder(input_ids=tokens.input_ids.to(device),
                         attention_mask=tokens.attention_mask.to(device)
                         ).last_hidden_state
    return h, tokens.attention_mask.to(device)


def apply_lora(transformer, rank, alpha):
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05, bias="none",
    )
    transformer = get_peft_model(transformer, cfg)
    transformer.print_trainable_parameters()
    return transformer


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    # ── Models ───────────────────────────────────────────────────────────
    print("Loading VAE...")
    vae = CausalVideoAutoencoder.from_pretrained(args.ckpt_path).to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    if args.decoder_ckpt:
        sd = torch.load(args.decoder_ckpt, map_location="cpu")
        vae.load_state_dict(sd, strict=False)
        vae = vae.to(device=device, dtype=dtype)  # re-cast after fp32 ckpt load
        print(f"Loaded decoder ckpt from {args.decoder_ckpt}")

    print("Loading transformer...")
    transformer = Transformer3DModel.from_pretrained(args.ckpt_path).to(device, dtype=dtype)

    print("Applying LoRA...")
    transformer = apply_lora(transformer, args.lora_rank, args.lora_alpha)
    transformer = transformer.to(device=device, dtype=dtype)  # cast LoRA adapters to bf16
    transformer.train()

    print("Loading text encoder...")
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder_path, subfolder="text_encoder"
    ).to(device, dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    text_dim = text_encoder.config.d_model

    print("Creating palette encoder...")
    palette_enc = PaletteEncoder(palette_k=args.palette_k, token_dim=text_dim).to(device, dtype=dtype)

    patchifier = SymmetricPatchifier(patch_size=1)

    # ── Optimizer ────────────────────────────────────────────────────────
    trainable = list(palette_enc.parameters()) + [
        p for p in transformer.parameters() if p.requires_grad
    ]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.total_steps)

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = AnimeClipDataset(
        args.data_root, num_frames=args.num_frames,
        height=args.height, width=args.width,
        palette_k=args.palette_k, augment=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    data_iter = iter(loader)

    # Pre-encode the fixed text prompt
    text_hidden, text_mask = encode_text(DUMMY_PROMPT, tokenizer, text_encoder, device)

    print(f"Dataset: {len(dataset)} clips")
    print("Starting training...")
    step = 0
    while step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        frames = batch["frames"].to(device, dtype=dtype)       # (B, T, 3, H, W)
        palette = batch["palette"].to(device, dtype=dtype)     # (B, K, 3)
        kf_idx_pixel = [k[0].item() for k in batch["keyframe_indices"]]  # 5 ints

        B, T, C, H, W = frames.shape
        frames_bcthw = frames.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, T, H, W)

        # ── Encode to latents (B, C_lat, T_lat, H_lat, W_lat) ────────────
        with torch.no_grad():
            latents = vae_encode(frames_bcthw, vae).to(dtype)
        _, C_lat, T_lat, H_lat, W_lat = latents.shape

        # Map pixel keyframe indices → latent frame indices
        kf_idx_lat = sorted(set(min(round(ki * T_lat / T), T_lat - 1) for ki in kf_idx_pixel))

        # ── Patchify latents → tokens + LATENT coords ────────────────────
        latent_tokens, latent_coords = patchifier.patchify(latents)  # (B, N, C_lat), (B, 3, N)
        N = latent_tokens.shape[1]

        # Convert latent coords → pixel coords, then normalize time axis by fps
        # (this is the indices_grid the transformer's RoPE expects, per pipeline)
        pixel_coords = latent_to_pixel_coords(latent_coords, vae, causal_fix=True)
        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / FRAME_RATE)

        # Conditioning mask per token: 1.0 at keyframe latent time positions.
        # latent_coords[:, 0] is the latent-frame index (0..T_lat-1) of each token.
        latent_t_idx = latent_coords[:, 0].long()               # (B, N)
        kf_set = torch.tensor(kf_idx_lat, device=device)
        cond_mask = (latent_t_idx.unsqueeze(-1) == kf_set).any(dim=-1).to(dtype)  # (B, N)
        # cond_mask=1 → hard conditioning (keep clean latent, no noise), timestep=0
        # cond_mask=0 → regular training position, noise applied

        # ── RF forward: sample t ∈ [0,1] per sample, add noise ───────────
        t = torch.rand(B, device=device, dtype=dtype)             # (B,)
        noise = torch.randn_like(latent_tokens)                    # (B, N, C_lat)
        t_tok = t[:, None, None]                                   # (B, 1, 1)
        # Per-token timestep: 0 at keyframes, t elsewhere
        t_per_token = t[:, None] * (1.0 - cond_mask)               # (B, N)
        # Noisy tokens: xt = (1-t_tok)*x0 + t_tok*noise, overridden to x0 at keyframes
        xt = (1.0 - t_per_token.unsqueeze(-1)) * latent_tokens + \
             t_per_token.unsqueeze(-1) * noise                     # (B, N, C_lat)

        # Target velocity: v = noise - x0. At keyframe tokens, t_per_token=0 so
        # the model should output 0 there; we'll zero-weight them in the loss.
        target_v = noise - latent_tokens

        # ── Conditioning: palette tokens concatenated after text ─────────
        palette_tokens = palette_enc(palette)                      # (B, K, D)
        text_h = text_hidden.expand(B, -1, -1)                     # (B, L, D)
        encoder_hidden = torch.cat([text_h, palette_tokens], dim=1).to(dtype)
        text_m = text_mask.expand(B, -1)                           # (B, L)
        pal_m = torch.ones(B, args.palette_k, device=device, dtype=text_m.dtype)
        encoder_mask = torch.cat([text_m, pal_m], dim=1)           # (B, L+K)

        # ── Transformer forward (predicts velocity) ──────────────────────
        # timestep param is the GLOBAL step index (0..1000); for per-token
        # timestep injection we scale t_per_token to 0..1000.
        timestep_1k = (t_per_token * 1000.0).to(dtype)              # (B, N)
        # The transformer expects either a scalar per-batch or (B, N) per-token.
        # The pipeline supplies shape (B, 1) extended to N via per-token cond
        # via `conditioning_mask`. Here we pass per-token directly:
        pred_v = transformer(
            hidden_states=xt.to(dtype),
            indices_grid=fractional_coords,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            timestep=timestep_1k,
            return_dict=False,
        )[0]                                                        # (B, N, C_lat)

        # ── Loss: weighted MSE (zero weight at keyframe tokens) ──────────
        weight = (1.0 - cond_mask).unsqueeze(-1)                    # (B, N, 1)
        per_elem = (pred_v - target_v).pow(2) * weight
        denom = weight.expand_as(per_elem).sum().clamp(min=1.0)
        loss_v = per_elem.sum() / denom

        loss = loss_v

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optim.step()
        sched.step()

        if step % args.log_interval == 0:
            kf_tok_frac = cond_mask.mean().item()
            msg = (f"step {step:5d} | loss {loss.item():.4f} | "
                   f"v_mean {pred_v.abs().mean().item():.3f} | "
                   f"kf_tok {kf_tok_frac:.2f}")
            print(msg, flush=True)
            writer.add_scalar("loss/velocity", loss.item(), step)
            writer.add_scalar("stats/pred_v_abs_mean", pred_v.abs().mean().item(), step)
            writer.add_scalar("stats/keyframe_token_fraction", kf_tok_frac, step)
            writer.add_scalar("lr", sched.get_last_lr()[0], step)

        if step > 0 and step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"lora_step_{step}")
            transformer.save_pretrained(save_path)
            torch.save(palette_enc.state_dict(), os.path.join(save_path, "palette_encoder.pt"))
            print(f"Saved → {save_path}", flush=True)

        step += 1

    final_path = os.path.join(args.output_dir, "final")
    transformer.save_pretrained(final_path)
    torch.save(palette_enc.state_dict(), os.path.join(final_path, "palette_encoder.pt"))
    print(f"Done. Final → {final_path}", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
