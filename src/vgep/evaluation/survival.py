"""Survival (1 - CDF) plot over per-gene Pearson r, one curve per method."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_survival_curves(
    corr_by_method: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    title: str = "",
    xlabel: str = "Pearson r",
    ax: plt.Axes | None = None,
    show_medians: bool = True,
    show_legend: bool = True,
) -> plt.Figure:
    """Render one empirical survival-function line per method. Returns the Figure.

    Uses the offset `1 - CDF + 1/n` so the curve starts at 1 and ends at 1/n
    rather than 0 (useful for log-scaled y-axes).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for method, values in corr_by_method.items():
        arr = np.asarray(values)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        sorted_vals = np.sort(arr)
        n = sorted_vals.size
        survival = 1.0 - np.arange(0, n) / n
        color = colors.get(method, None)
        ax.plot(sorted_vals, survival, color=color, label=method, linewidth=1.5)
        if show_medians:
            median = float(np.median(sorted_vals))
            ax.axvline(median, color=color, linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Survival (1 - CDF)")
    if title:
        ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    if show_legend:
        ax.legend(loc="best", frameon=False)
    return fig
