import numpy as np
import torch
import warnings


def jitter_torch(x, sigma=0.02):
    noise = torch.randn_like(x) * sigma
    return x + noise


def scaling_torch(x, sigma=0.1):

    B, T, F = x.shape
    factor = torch.normal(
        mean=1.0,
        std=sigma,
        size=(B, 1, F),
        device=x.device
    )

    return x * factor


def permutation_torch(x, max_segments=5):

    B, T, F = x.shape
    x_perm = x.clone()

    for b in range(B):

        num_segs = np.random.randint(1, max_segments)

        if num_segs > 1:

            split_points = np.random.choice(
                T-1,
                num_segs-1,
                replace=False
            )

            split_points.sort()

            segments = np.split(np.arange(T), split_points)
            np.random.shuffle(segments)

            perm_idx = np.concatenate(segments)

            x_perm[b] = x[b, perm_idx]

    return x_perm
