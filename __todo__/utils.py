import numpy as np


def softclip(x, lo=None, hi=None, lo_pct=2, hi_pct=98, smoothness=3.0):
    """Soft clip values using a robust percentile anchor
    and a smooth sigmoidal transition to preserve highlight/shadow detail.
    """
    if lo is None or hi is None:
        lo, hi = np.percentile(x, (lo_pct, hi_pct))

    # Avoid division by zero
    span = (hi - lo) if (hi - lo) > 1e-9 else 1e-9

    # Normalize center around 0 and scale by smoothness
    # We map the [lo, hi] range approximately to [-0.5, 0.5]
    y = (x - lo) / span

    # Apply a smooth sigmoidal transfer function (e.g., scaled tanh or logistic)
    # This smoothly asymptotes towards 0 and 1 without harsh flattening
    rescaled = 0.5 * (np.tanh(smoothness * (y - 0.5)) / np.tanh(0.5 * smoothness) + 1)

    # Fallback clip to strictly keep output in [0, 1] range for matplotlib
    return np.clip(rescaled, 0, 1)
