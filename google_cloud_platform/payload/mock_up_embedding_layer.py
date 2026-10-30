"""Mock-up embedding layer for temporal image stacks.
Fully convolutional (no attention, no positional encoding)
Each time step is embedded independently with shared weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_channel_stats(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """NaN-aware per-channel mean and std.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(..., C, H, W)``, e.g. ``(N, T, C, H, W)``. NaNs are ignored.
    eps : float, optional
        Lower bound for the std, to avoid division by zero.

    Returns
    -------
    mean, std : torch.Tensor
        Each of shape ``(C,)``. Pass them to :class:`ImageStackEmbedding`.
    """
    flat = x.movedim(-3, 0).reshape(x.shape[-3], -1)
    valid = torch.isfinite(flat)
    n = valid.sum(1).clamp(min=1)
    zero = torch.zeros_like(flat)
    mean = torch.where(valid, flat, zero).sum(1) / n
    var = (torch.where(valid, flat - mean[:, None], zero) ** 2).sum(1) / n
    return mean, var.sqrt().clamp(min=eps)


class _LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels at each spatial position, for ``(B, C, H, W)`` input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _Block(nn.Module):
    """Residual block: depthwise conv -> LayerNorm -> pointwise MLP."""

    def __init__(self, dim: int, expansion: int, kernel_size: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = _LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, 1)
        self.pw2 = nn.Conv2d(dim * expansion, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))


class ImageStackEmbeddingMockUp(nn.Module):
    """Embed a stack of multi-channel images into a grid of vectors.

    Missing data (NaN) is replaced by 0 after normalisation, and a validity mask is
    appended as extra input channels so the network can tell "zero" from "missing".
    Image sizes not divisible by ``patch_size`` are zero-padded on the bottom/right
    (padding is marked invalid in the mask).

    Parameters
    ----------
    in_channels : int
        Number of channels ``C`` per image.
    embed_dim : int, optional
        Embedding dimension ``D``. Default ``128``.
    patch_size : int, optional
        Side of the square patch mapped to one vector (stride of the first conv).
        Default ``8``.
    depth : int, optional
        Number of residual conv blocks after patchifying. Default ``2``.
    expansion : int, optional
        Hidden-width multiplier of the pointwise MLP in each block. Default ``4``.
    kernel_size : int, optional
        Depthwise kernel size, in patches. Default ``7``.
    mean, std : torch.Tensor | None, optional
        Per-channel normalisation, shape ``(C,)``; see :func:`compute_channel_stats`.
        Default is no normalisation. Channels differ in scale by orders of magnitude
        (reflectance, dB, mm, degC), so you almost always want to set these.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        patch_size: int = 8,
        depth: int = 2,
        expansion: int = 4,
        kernel_size: int = 7,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.register_buffer(
            "mean", torch.zeros(in_channels) if mean is None else torch.as_tensor(mean).float()
        )
        self.register_buffer(
            "std", torch.ones(in_channels) if std is None else torch.as_tensor(std).float()
        )
        self.patchify = nn.Conv2d(2 * in_channels, embed_dim, patch_size, stride=patch_size)
        self.blocks = nn.Sequential(*[_Block(embed_dim, expansion, kernel_size) for _ in range(depth)])
        self.norm = _LayerNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed an image stack.

        Parameters
        ----------
        x : torch.Tensor
            Float tensor ``(B, T, C, H, W)``, or ``(T, C, H, W)`` for a single sample.
            NaN marks missing data. ``T`` may vary between calls.

        Returns
        -------
        torch.Tensor
            ``(B, T, D, ceil(H/p), ceil(W/p))`` (or without ``B`` for 4-D input):
            one ``D``-dimensional vector per patch and time step.
        """
        squeeze = x.dim() == 4
        if squeeze:
            x = x.unsqueeze(0)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)

        valid = torch.isfinite(x)
        x = torch.where(valid, x, torch.zeros_like(x))
        x = (x - self.mean[:, None, None]) / self.std[:, None, None] * valid
        x = torch.cat([x, valid.to(x.dtype)], dim=1)

        x = F.pad(x, (0, (-w) % self.patch_size, 0, (-h) % self.patch_size))

        x = self.patchify(x)
        x = self.blocks(x)
        x = self.norm(x)

        x = x.reshape(b, t, *x.shape[1:])
        return x[0] if squeeze else x
