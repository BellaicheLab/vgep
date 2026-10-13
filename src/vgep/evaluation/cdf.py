"""CDF plot over per-gene Pearson r, one curve per method."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_cdf_curves(
    corr_by_method: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    title: str = "",
    xlabel: str = "Pearson r",
    ax: plt.Axes | None = None,
    show_medians: bool = True,
    show_legend: bool = True,
    markers: Mapping[str, str] | None = None,
    linestyles: Mapping[str, str] | None = None,
    markersize: float = 4.0,
    linewidth: float = 1.5,
) -> plt.Figure:
    """Render one empirical CDF line per method. Returns the Figure.

    Each curve takes its own ``color``, ``marker`` and ``linestyle``; each
    per-gene Pearson r is drawn as a point.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    markers = markers or {}
    linestyles = linestyles or {}

    for method, values in corr_by_method.items():
        arr = np.asarray(values)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        sorted_vals = np.sort(arr)
        cdf = np.arange(1, sorted_vals.size + 1) / sorted_vals.size
        color = colors.get(method, None)
        marker = markers.get(method, None)
        linestyle = linestyles.get(method, "-")
        ax.plot(
            sorted_vals,
            cdf,
            color=color,
            label=method,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            markerfacecolor=color,
            markeredgecolor=color,
        )
        if show_medians:
            median = float(np.median(sorted_vals))
            ax.axvline(median, color=color, linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    if title:
        ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    if show_legend:
        ax.legend(loc="best", frameon=False)
    return fig
