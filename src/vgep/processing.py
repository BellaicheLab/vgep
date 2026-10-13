"""Data normalization and scaling utilities."""

import numpy as np
from anndata import AnnData


def normalize_01(matrix, lower=1, upper=99):
    """Normalize matrix to [0, 1] based on percentiles.

    Parameters
    ----------
    matrix : array-like
        Input matrix (may contain NaN).
    lower : float
        Lower percentile for clipping (default 1).
    upper : float
        Upper percentile for clipping (default 99).

    Returns
    -------
    np.ndarray
        Normalized matrix with values in [0, 1].
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    vals = matrix[~np.isnan(matrix)].flatten()
    if len(vals) == 0:
        return matrix
    vmin, vmax = np.nanpercentile(vals, [lower, upper])
    if vmax == vmin:
        vmax = vmin + 1e-8  # Avoid division by zero
    normed = np.clip((matrix - vmin) / (vmax - vmin), 0, 1)
    return normed


def scale_data(single_cells: AnnData, library_size=10000):
    """Scale UMI counts by library size.

    Divides each cell's counts by its total UMI count, then multiplies
    by ``library_size``. Stores the result in the ``"scaled_data"`` layer.

    Parameters
    ----------
    single_cells : AnnData
        Single-cell data with raw counts in ``.X``.
    library_size : int
        Target library size for normalization.

    Returns
    -------
    AnnData
        Input object with an added ``"scaled_data"`` layer.
    """

    raw_single_cell = single_cells.X

    umi_by_cell = np.sum(raw_single_cell, axis=1).squeeze()
    lib_sizes = np.tile(umi_by_cell, (raw_single_cell.shape[1], 1)).T

    raw_single_cell = raw_single_cell / lib_sizes
    raw_single_cell *= library_size

    single_cells.layers["scaled_data"] = raw_single_cell.astype(float)

    return single_cells
