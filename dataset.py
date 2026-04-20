import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Dataset


class AnimeClipDataset(Dataset):
    """
    Loads consecutive anime frames from a directory of video frame folders.
    Each subfolder contains sequentially numbered frames from one video clip.

    Returns:
        frames: (T, 3, H, W) in [-1, 1]
        keyframe_indices: list of 5 indices into [0..T-1]
        palette: (K, 3) in [0, 1]
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 33,
        height: int = 512,
        width: int = 768,
        palette_k: int = 16,
        augment: bool = True,
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.palette_k = palette_k
        self.augment = augment

        self.clips = self._discover_clips(data_root)
        if len(self.clips) == 0:
            raise ValueError(f"No clips found under {data_root}")

    def _discover_clips(self, root: str):
        clips = []
        root = Path(root)
        for subdir in sorted(root.iterdir()):
            if subdir.is_dir():
                frames = sorted(subdir.glob("*.png")) + sorted(subdir.glob("*.jpg"))
                if len(frames) >= self.num_frames:
                    clips.append(frames)
        # Also accept flat directories of images if no subdirs found
        if not clips:
            frames = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
            if len(frames) >= self.num_frames:
                clips.append(frames)
        return clips

    def _load_frame(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height), Image.BILINEAR)
        t = TVF.to_tensor(img)  # (3, H, W) [0,1]
        t = t * 2.0 - 1.0      # → [-1, 1]
        return t

    def _extract_palette(self, frames: list[torch.Tensor]) -> torch.Tensor:
        # Sample pixels from middle frame for palette extraction
        mid = frames[len(frames) // 2]  # (3, H, W) in [-1, 1]
        pixels = mid.permute(1, 2, 0).reshape(-1, 3).numpy()
        pixels = (pixels + 1.0) / 2.0  # → [0, 1]

        # Subsample for speed
        idx = np.random.choice(len(pixels), min(10000, len(pixels)), replace=False)
        pixels = pixels[idx]

        kmeans = MiniBatchKMeans(n_clusters=self.palette_k, n_init=3, random_state=0)
        kmeans.fit(pixels)
        palette = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # (K, 3) [0,1]
        return palette

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frames_paths = self.clips[idx]

        # Sample a contiguous window
        start = random.randint(0, len(frames_paths) - self.num_frames)
        selected = frames_paths[start : start + self.num_frames]

        frames = [self._load_frame(p) for p in selected]  # list of (3,H,W)

        # Apply consistent augmentation
        if self.augment and random.random() < 0.5:
            frames = [TVF.hflip(f) for f in frames]

        palette = self._extract_palette(frames)
        frames_tensor = torch.stack(frames, dim=0)  # (T, 3, H, W)

        # Pick 5 keyframe indices spread across the clip
        T = self.num_frames
        keyframe_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]

        return {
            "frames": frames_tensor,          # (T, 3, H, W) [-1,1]
            "keyframe_indices": keyframe_indices,
            "palette": palette,               # (K, 3) [0,1]
        }
