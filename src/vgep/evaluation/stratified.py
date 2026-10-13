"""Stratified CDF comparison (e.g. protein-antibody vs smiFISH validation cohorts)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _subset_corrs(
    corr: np.ndarray, gene_names: Sequence[str], subset_genes: Sequence[str]
) -> np.ndarray:
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    idxs = [name_to_idx[g] for g in subset_genes if g in name_to_idx]
    if not idxs:
        return np.empty(0, dtype=corr.dtype)
    return corr[np.asarray(idxs)]


def plot_protein_vs_fish_cdf(
    corr: np.ndarray,
    gene_names: Sequence[str],
    protein_genes: Sequence[str],
    fish_genes: Sequence[str],
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Figure:
    """Render CDFs of per-gene Pearson r stratified by protein vs smiFISH cohorts.

    Overlays a small stats box reporting KS and Mann-Whitney U p-values when both
    cohorts have >= 3 finite samples.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    p_vals = _subset_corrs(corr, gene_names, protein_genes)
    f_vals = _subset_corrs(corr, gene_names, fish_genes)

    for values, label, color in [
        (p_vals, "protein", "#1f77b4"),
        (f_vals, "fish", "#d62728"),
    ]:
        arr = values[np.isfinite(values)]
        if arr.size == 0:
            continue
        sorted_vals = np.sort(arr)
        cdf = np.arange(1, sorted_vals.size + 1) / sorted_vals.size
        ax.plot(sorted_vals, cdf, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Pearson r")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.0)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)

    p_clean = p_vals[np.isfinite(p_vals)]
    f_clean = f_vals[np.isfinite(f_vals)]
    if p_clean.size >= 3 and f_clean.size >= 3:
        from scipy.stats import ks_2samp, mannwhitneyu

        ks = ks_2samp(p_clean, f_clean)
        mwu = mannwhitneyu(p_clean, f_clean, alternative="two-sided")
        txt = (
            f"KS p={ks.pvalue:.2g}\n"
            f"MWU p={mwu.pvalue:.2g}\n"
            f"n_protein={p_clean.size}, n_fish={f_clean.size}"
        )
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=6,
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=2),
        )
    return fig
