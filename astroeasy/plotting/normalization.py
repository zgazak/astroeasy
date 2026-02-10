"""Image normalization utilities for astronomical visualization."""

import numpy as np
from astropy.visualization import ZScaleInterval


def zscale(data: np.ndarray, contrast: float = 0.2) -> np.ndarray:
    """Apply astronomical zscale stretch for visualization.

    This function handles non-finite values (NaN/Inf) that commonly appear
    in astronomical FITS data (e.g., from divide-by-flat or masked pixels).

    Args:
        data: Input image data array.
        contrast: Contrast parameter for ZScaleInterval (default 0.2).

    Returns:
        Normalized array with values scaled to [0, 1] range.
    """
    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)

    # Replace NaN/Inf with median so ZScaleInterval can compute limits
    fill = float(np.median(arr[finite]))
    if not np.all(finite):
        arr = arr.copy()
        arr[~finite] = fill

    norm = ZScaleInterval(contrast=contrast)
    out = norm(arr)

    # Ensure output is finite
    out = np.asarray(out, dtype=np.float32)
    out_finite = np.isfinite(out)
    if not np.any(out_finite):
        return np.zeros_like(out)
    if not np.all(out_finite):
        out = out.copy()
        out[~out_finite] = float(np.median(out[out_finite]))

    return out
