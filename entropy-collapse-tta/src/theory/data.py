"""Synthetic Gaussian mixture data generator for theory track.

Generates a balanced K-class Gaussian mixture in d dimensions with configurable
class centroid geometry, separation, and covariance.
"""

from typing import Optional, Tuple
import numpy as np
import torch


def generate_gaussian_mixture(
    num_classes: int = 3,
    dim: int = 2,
    samples_per_class: int = 500,
    separation: float = 2.0,
    noise_std: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic balanced Gaussian mixture dataset.

    Geometry:
        Class centroids are placed symmetrically on the (dim-1)-sphere or regular
        polygon of radius `separation`. Each class has isotropic covariance sigma^2 * I.

    Args:
        num_classes: Number of distinct classes K.
        dim: Dimensionality of input space d (d >= 2).
        samples_per_class: Number of samples generated for each class.
        separation: Radius of class centroid placement (determines cluster separation).
        noise_std: Standard deviation sigma of the isotropic Gaussian noise.
        seed: Random seed for exact reproducibility.

    Returns:
        X: Tensor of shape (K * samples_per_class, dim), float32.
        y: Tensor of shape (K * samples_per_class,), int64 (for evaluation ONLY).
        centroids: Tensor of shape (K, dim), class center locations.
    """
    if dim < 2:
        raise ValueError("Dimension must be at least 2.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Place centroids regularly on a circle in the first 2 dimensions
    angles = torch.linspace(0, 2 * np.pi, steps=num_classes + 1)[:-1]
    centroids = torch.zeros(num_classes, dim, dtype=torch.float32)
    centroids[:, 0] = separation * torch.cos(angles)
    centroids[:, 1] = separation * torch.sin(angles)

    X_list = []
    y_list = []

    for k in range(num_classes):
        mean = centroids[k]
        noise = torch.randn(samples_per_class, dim, generator=generator) * noise_std
        X_k = mean + noise
        y_k = torch.full((samples_per_class,), k, dtype=torch.long)
        X_list.append(X_k)
        y_list.append(y_k)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    # Deterministic permutation to interleave classes in batches if needed
    perm = torch.randperm(X.size(0), generator=generator)
    return X[perm], y[perm], centroids
