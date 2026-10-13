"""Per-gene Pearson correlation between a predicted and a reference tensor."""

from __future__ import annotations

import numpy as np


def per_gene_pearson_r(
    pred: np.ndarray,
    ref: np.ndarray,
    gene_axis: int = 0,
    mask_nan: bool = True,
) -> np.ndarray:
    """Return (G,) float32 Pearson r per gene.

    NaN is returned for genes that are all-NaN after masking or constant (zero variance)
    in either input. No warnings are emitted for constant / masked-out genes.
    """
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} ref={ref.shape}")
    # Move gene axis to position 0 so we can iterate simply.
    pred_g = np.moveaxis(pred, gene_axis, 0)
    ref_g = np.moveaxis(ref, gene_axis, 0)
    G = pred_g.shape[0]
    # Flatten trailing dims; Pearson r is invariant to flattening.
    p2 = pred_g.reshape(G, -1).astype(np.float64, copy=False)
    r2 = ref_g.reshape(G, -1).astype(np.float64, copy=False)

    out = np.full(G, np.nan, dtype=np.float32)
    for i in range(G):
        pi, ri = p2[i], r2[i]
        if mask_nan:
            valid = np.isfinite(pi) & np.isfinite(ri)
            if valid.sum() < 2:
                continue
            pi, ri = pi[valid], ri[valid]
        pvar = np.var(pi)
        rvar = np.var(ri)
        if pvar == 0.0 or rvar == 0.0 or not np.isfinite(pvar) or not np.isfinite(rvar):
            continue
        # Avoid RuntimeWarning from np.corrcoef on constant inputs
        pmean, rmean = pi.mean(), ri.mean()
        num = ((pi - pmean) * (ri - rmean)).sum()
        den = np.sqrt(((pi - pmean) ** 2).sum() * ((ri - rmean) ** 2).sum())
        if den == 0.0:
            continue
        out[i] = np.float32(num / den)
    return out
