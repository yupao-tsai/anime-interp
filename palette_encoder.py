import torch
import torch.nn as nn


class PaletteEncoder(nn.Module):
    """
    Maps a palette of K RGB colors → K cross-attention tokens of dimension D.
    These tokens are concatenated with the T5 text tokens before cross-attention.
    """

    def __init__(self, palette_k: int = 16, token_dim: int = 4096):
        super().__init__()
        self.palette_k = palette_k
        self.token_dim = token_dim
        self.mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, token_dim),
        )

    def forward(self, palette: torch.Tensor) -> torch.Tensor:
        """
        Args:
            palette: (B, K, 3) float in [0, 1]
        Returns:
            tokens: (B, K, token_dim)
        """
        return self.mlp(palette)


def get_text_encoder_dim(text_encoder) -> int:
    """Get the hidden dim from a T5EncoderModel."""
    return text_encoder.config.d_model
