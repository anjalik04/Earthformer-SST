"""
Data augmentation utilities for SST spatiotemporal tensors.

Applied during the training step (not in the datamodule), so the existing
SSTPatchDataModule remains unchanged.
"""
import torch


def random_temporal_flip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Randomly flip the time axis of a spatiotemporal tensor.

    Parameters
    ----------
    x : Tensor
        Shape (B, T, H, W, C) or (B, T, C, H, W).
    p : float
        Probability of flipping.

    Returns
    -------
    Tensor
        Same shape, with time axis possibly reversed.
    """
    if torch.rand(1).item() < p:
        return x.flip(dims=[1])
    return x


def gaussian_noise(
    x: torch.Tensor,
    sigma: float = 0.01,
    p: float = 0.5,
) -> torch.Tensor:
    """Add Gaussian noise to the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor (any shape).
    sigma : float
        Standard deviation of the noise.
    p : float
        Probability of applying noise.

    Returns
    -------
    Tensor
        Input with optional additive Gaussian noise.
    """
    if torch.rand(1).item() < p:
        noise = torch.randn_like(x) * sigma
        return x + noise
    return x
