"""
Stage 1: LoRA fine-tuning of LTX-Video transformer + palette conditioning.

Training scheme: masked-frame prediction
  - Take a clip of T frames
  - 5 frames are "keyframes" (known, not noised)
  - Model must predict all T frames given keyframes
  - Loss computed only on non-keyframe positions

Palette conditioning:
  - MLP encodes K palette colors → K tokens
  - Appended to T5 encoder_hidden_states before cross-attention
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

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode, normalize_latents
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.schedulers.rf import RectifiedFlowScheduler

from dataset import AnimeClipDataset
from palette_encoder import PaletteEncoder


DUMMY_PROMPT = "anime cel-shaded animation"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        default="/storage/SSD2/program/LTX-2/models/checkpoints/ltx-2-19b-dev.safetensors",
        help="LTX safetensors or HF dir",
    )
    parser.add_argument("--text_encoder_path", default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="runs/lora_train")
    parser.add_argument("--decoder_ckpt", default=None, help="Fine-tuned decoder state_dict (optional)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--palette_k", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--w_palette", type=float, default=0.3)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    return parser.parse_args()


def encode_text(prompt: str, tokenizer, text_encoder, device, max_length=256):
    tokens = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        hidden = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return hidden, attention_mask  # (1, L, D), (1, L)


def apply_lora(transformer: Transformer3DModel, rank: int, alpha: int):
    target_modules = []
    for name, module in transformer.named_modules():
        if hasattr(module, "to_q") or hasattr(module, "to_k"):
            # Collect attention linear layer names
            pass

    # Target all attention projections in BasicTransformerBlocks
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    return transformer


def inject_keyframes(noisy_latents: torch.Tensor, clean_latents: torch.Tensor, keyframe_indices: list):
    """
    Replace noisy latent at keyframe positions with clean latent.
    noisy_latents: (B, C, T, H, W)
    clean_latents: (B, C, T, H, W)
    """
    result = noisy_latents.clone()
    for ki in keyframe_indices:
        result[:, :, ki, :, :] = clean_latents[:, :, ki, :, :]
    return result


def compute_latent_loss_mask(num_frames: int, keyframe_indices: list, device) -> torch.Tensor:
    """Returns (T,) mask: 1 for non-keyframe positions, 0 for keyframes."""
    mask = torch.ones(num_frames, device=device)
    for ki in keyframe_indices:
        mask[ki] = 0.0
    return mask


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    # ── Load models ──────────────────────────────────────────────────────
    print("Loading VAE...")
    vae = CausalVideoAutoencoder.from_pretrained(args.ckpt_path).to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    if args.decoder_ckpt:
        sd = torch.load(args.decoder_ckpt, map_location="cpu")
        vae.load_state_dict(sd, strict=False)
        print(f"Loaded fine-tuned decoder from {args.decoder_ckpt}")

    print("Loading transformer...")
    transformer = Transformer3DModel.from_pretrained(args.ckpt_path).to(device, dtype=dtype)

    print("Applying LoRA...")
    transformer = apply_lora(transformer, rank=args.lora_rank, alpha=args.lora_alpha)
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
    scheduler = RectifiedFlowScheduler.from_pretrained(args.ckpt_path)

    # ── Optimizer: only LoRA + palette encoder ────────────────────────────
    trainable_params = list(palette_enc.parameters()) + [
        p for p in transformer.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = AnimeClipDataset(
        args.data_root,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        palette_k=args.palette_k,
        augment=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    data_iter = iter(loader)

    # Pre-encode the fixed text prompt (same for all batches)
    text_hidden, text_mask = encode_text(DUMMY_PROMPT, tokenizer, text_encoder, device)
    # text_hidden: (1, L, D),  text_mask: (1, L)

    print("Starting training...")
    step = 0
    while step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        frames = batch["frames"].to(device, dtype=dtype)      # (B, T, 3, H, W) [-1,1]
        palette = batch["palette"].to(device, dtype=dtype)    # (B, K, 3) [0,1]
        keyframe_indices = batch["keyframe_indices"]          # list of 5 tensors

        B, T, C, H, W = frames.shape
        # Rearrange for VAE: (B, 3, T, H, W)
        frames_bcthw = frames.permute(0, 2, 1, 3, 4)

        # ── Encode frames to latents ──────────────────────────────────────
        with torch.no_grad():
            latents = vae_encode(frames_bcthw, vae)   # (B, latent_C, T', H', W')
            # VAE downsamples spatial by 8, temporal by (T+1)//... depends on config
            # latent T' may differ from T; for CausalVideoAutoencoder with default config:
            #   temporal stride=4, spatial stride=8

        # ── Sample timestep & add noise ───────────────────────────────────
        t = torch.rand(B, device=device)  # [0,1] for rectified flow
        noise = torch.randn_like(latents)
        # x_t = (1 - t) * x0 + t * noise  (rectified flow)
        t_bcthw = t.view(B, 1, 1, 1, 1)
        noisy_latents = (1.0 - t_bcthw) * latents + t_bcthw * noise

        # ── Inject keyframes (undo noise at known positions) ─────────────
        # Map pixel-space keyframe indices to latent-space frame indices
        latent_T = latents.shape[2]
        scale = latent_T / T
        latent_kf_indices = [
            min(int(round(ki[0].item() * scale)), latent_T - 1)
            for ki in keyframe_indices
        ]
        conditioned_latents = inject_keyframes(noisy_latents, latents, latent_kf_indices)

        # Loss mask: only penalize non-keyframe latent frames
        frame_mask = compute_latent_loss_mask(latent_T, latent_kf_indices, device)
        # frame_mask: (T',) → broadcast to (B, latent_C, T', H', W')

        # ── Patchify ──────────────────────────────────────────────────────
        latent_patches, latent_coords = patchifier.patchify(conditioned_latents)
        # latent_patches: (B, N, latent_C),  latent_coords: (B, 3, N)

        # Fractional coords for RoPE
        _, latent_C, lat_T, lat_H, lat_W = latents.shape
        fractional_coords = latent_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / 24.0)  # 24 fps default

        # ── Palette → tokens, concat with text ───────────────────────────
        palette_tokens = palette_enc(palette)                          # (B, K, D)
        text_h = text_hidden.expand(B, -1, -1)                        # (B, L, D)
        encoder_hidden = torch.cat([text_h, palette_tokens], dim=1)   # (B, L+K, D)

        # Encoder attention mask: all ones for palette tokens
        text_m = text_mask.expand(B, -1)                              # (B, L)
        palette_mask = torch.ones(B, args.palette_k, device=device, dtype=text_m.dtype)
        encoder_mask = torch.cat([text_m, palette_mask], dim=1)       # (B, L+K)

        # ── Timestep embedding ────────────────────────────────────────────
        timestep = (t * 1000).long()   # transformer expects 0..1000 range

        # ── Forward ───────────────────────────────────────────────────────
        pred = transformer(
            hidden_states=latent_patches,
            indices_grid=fractional_coords,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            timestep=timestep,
            return_dict=False,
        )[0]  # (B, N, latent_C)

        # ── Unpatchify predicted latents ──────────────────────────────────
        pred_latents = patchifier.unpatchify(
            latents=pred,
            output_height=lat_H,
            output_width=lat_W,
            out_channels=latent_C,
        )  # (B, latent_C, T', H', W')

        # ── Rectified flow target: predict x0 ────────────────────────────
        # Under RF: x_t = (1-t)*x0 + t*noise  →  x0 = (x_t - t*pred) / (1-t)
        # Simplified: direct x0 prediction loss
        target = latents  # predict the clean latent

        # Apply frame mask
        mask_full = frame_mask.view(1, 1, latent_T, 1, 1)  # broadcast
        loss_recon = (F.mse_loss(pred_latents, target, reduction="none") * mask_full).mean()

        # Palette adhesion loss (soft snap to nearest palette color in pixel space)
        if args.w_palette > 0 and step % 5 == 0:
            with torch.no_grad():
                pred_pixels = vae_decode(pred_latents.detach().float(), vae, is_video=True)
                # (B, 3, T', H', W') → pick middle frame for efficiency
                mid_t = lat_T // 2
                pred_mid = pred_pixels[:, :, mid_t].detach()  # (B, 3, H, W)

                p01 = (palette + 0.0) * 2 - 1  # [0,1] → [-1,1]
                # nearest palette color per pixel
                diff = pred_mid.unsqueeze(2) - p01.unsqueeze(-1).unsqueeze(-1)  # (B,3,K,H,W)
                nearest_idx = diff.norm(dim=1).argmin(dim=1)  # (B, K, H, W)  ← wrong shape
                # fix: diff is (B,K,H,W) after norm over channel dim
                diff = pred_mid.unsqueeze(2) - p01.view(B, args.palette_k, 3, 1, 1)  # (B,K,3,H,W)
                dist = diff.norm(dim=2)              # (B, K, H, W)
                nearest_idx = dist.argmin(dim=1)     # (B, H, W)
                # gather nearest palette color
                p_flat = p01.view(B, args.palette_k, 3)   # (B, K, 3)
                idx_expand = nearest_idx.view(B, 1, *nearest_idx.shape[1:]).expand(B, 3, *nearest_idx.shape[1:])
                # p_flat indexed by (B, H, W) → (B, 3, H, W)
                p_indexed = torch.gather(
                    p_flat.unsqueeze(-1).unsqueeze(-1).expand(B, args.palette_k, 3, *pred_mid.shape[2:]),
                    dim=1,
                    index=nearest_idx.unsqueeze(2).unsqueeze(1).expand(B, 1, 3, *pred_mid.shape[2:]),
                ).squeeze(1)  # (B, 3, H, W)
                snap_target = p_indexed.detach()

            pred_mid_train = vae_decode(pred_latents[:, :, mid_t:mid_t+1].float(), vae, is_video=True)
            pred_mid_train = pred_mid_train[:, :, 0]  # (B, 3, H, W)
            loss_palette = F.l1_loss(pred_mid_train, snap_target)
        else:
            loss_palette = torch.tensor(0.0, device=device)

        loss = loss_recon + args.w_palette * loss_palette

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        lr_scheduler.step()

        if step % args.log_interval == 0:
            print(
                f"step {step:5d} | loss {loss.item():.4f} "
                f"| recon {loss_recon.item():.4f} | pal {loss_palette.item():.4f}"
            )
            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("loss/recon", loss_recon.item(), step)
            writer.add_scalar("loss/palette", loss_palette.item(), step)

        if step > 0 and step % args.save_interval == 0:
            save_dir = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            transformer.save_pretrained(save_dir)
            torch.save(palette_enc.state_dict(), os.path.join(save_dir, "palette_encoder.pt"))
            print(f"Saved checkpoint → {save_dir}")

        step += 1

    # Final save
    save_dir = os.path.join(args.output_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    transformer.save_pretrained(save_dir)
    torch.save(palette_enc.state_dict(), os.path.join(save_dir, "palette_encoder.pt"))
    print(f"Done. Final saved → {save_dir}")
    writer.close()


if __name__ == "__main__":
    main()
