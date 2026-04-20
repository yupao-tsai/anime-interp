"""
Stage 0: Fine-tune VAE decoder on anime frames.
Encoder is frozen; only decoder is updated.
Goal: reproduce flat colors and sharp line edges.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LTX-Video"))

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode

from dataset import AnimeClipDataset


class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3).expand(3, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).expand(3, 1, 3, 3))

    def forward(self, x):  # (B, 3, H, W) in [-1,1]
        gx = F.conv2d(x, self.kx, padding=1, groups=3)
        gy = F.conv2d(x, self.ky, padding=1, groups=3)
        return (gx**2 + gy**2).sqrt()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features[:16]
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg

    def forward(self, pred, target):  # both (B, 3, H, W) in [-1,1]
        p = (pred + 1) / 2
        t = (target + 1) / 2
        return F.l1_loss(self.vgg(p), self.vgg(t))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="Path to LTX safetensors")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="runs/vae_finetune")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--w_l1", type=float, default=1.0)
    parser.add_argument("--w_perc", type=float, default=0.1)
    parser.add_argument("--w_edge", type=float, default=0.5)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    print(f"Loading VAE from {args.ckpt_path}")
    vae = CausalVideoAutoencoder.from_pretrained(args.ckpt_path).to(device)

    # Freeze encoder, train only decoder
    for name, p in vae.named_parameters():
        p.requires_grad = "decoder" in name or "post_quant" in name

    trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Trainable VAE params: {trainable:,}")

    sobel = SobelEdge().to(device)
    perc_loss_fn = PerceptualLoss().to(device)

    dataset = AnimeClipDataset(
        args.data_root,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in vae.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)

    step = 0
    data_iter = iter(loader)

    while step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # (B, T, 3, H, W) → take first frame for single-frame training
        frames = batch["frames"].to(device)  # (B, T, 3, H, W) [-1,1]
        if args.num_frames == 1:
            frames = frames[:, 0]  # (B, 3, H, W)
            frames = frames.unsqueeze(2)  # (B, 3, 1, H, W)
        else:
            frames = frames.permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)

        with torch.no_grad():
            latents = vae_encode(frames, vae)

        recon = vae_decode(latents, vae, is_video=True)  # (B, 3, T, H, W)

        # Flatten time for loss
        B, C, T, H, W = recon.shape
        recon_flat = recon.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        target_flat = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        l1 = F.l1_loss(recon_flat, target_flat)
        perc = perc_loss_fn(recon_flat.clamp(-1, 1), target_flat)
        edge = F.l1_loss(sobel(recon_flat), sobel(target_flat))

        loss = args.w_l1 * l1 + args.w_perc * perc + args.w_edge * edge

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | l1 {l1.item():.4f} | perc {perc.item():.4f} | edge {edge.item():.4f}")
            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("loss/l1", l1.item(), step)
            writer.add_scalar("loss/perc", perc.item(), step)
            writer.add_scalar("loss/edge", edge.item(), step)

        if step % (args.log_interval * 5) == 0:
            with torch.no_grad():
                grid = make_grid(
                    torch.cat([target_flat[:4].clamp(-1, 1), recon_flat[:4].clamp(-1, 1)], dim=0),
                    nrow=4, normalize=True, value_range=(-1, 1),
                )
            writer.add_image("recon", grid, step)

        if step > 0 and step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"decoder_step_{step}.pt")
            decoder_sd = {k: v for k, v in vae.state_dict().items() if "decoder" in k or "post_quant" in k}
            torch.save(decoder_sd, save_path)
            print(f"Saved decoder → {save_path}")

        step += 1

    # Final save
    save_path = os.path.join(args.output_dir, "decoder_final.pt")
    decoder_sd = {k: v for k, v in vae.state_dict().items() if "decoder" in k or "post_quant" in k}
    torch.save(decoder_sd, save_path)
    print(f"Done. Final decoder saved → {save_path}")
    writer.close()


if __name__ == "__main__":
    main()
