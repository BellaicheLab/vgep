"""Spatial gene expression heatmaps on tissue geometry."""

from typing import Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData, read_zarr
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Circle
from mudata import MuData
from pandas.api.types import is_integer_dtype

from ..utils import convert_to_numeric, remove_index_prefix, to_matrix_form
from ._utils import get_cropping_boundaries


def geometric(
    transcriptome: Union[AnnData, MuData, pd.DataFrame, zarr.Group],
    color: Union[str, Sequence[str], None] = None,
    geometry: pd.DataFrame = None,
    ax: Optional[Axes] = None,
    title: Union[str, Sequence[str]] = None,
    annotation: pd.DataFrame = None,
    mirrored: bool = False,
    colorbar: bool = True,
    colorbar_horizontal=True,
    spines=False,
    cropped=True,
    show=True,
    nr=1,
    nc=1,
    fr=10,
    fc=10,
    n_max=None,
    **imshow_kwargs,
):
    """Plot gene expression or observation on the tissue geometry.

    Renders a spatial heatmap of one or more features from the spatial
    transcriptome. Supports AnnData, MuData, DataFrame, and Zarr inputs.
    Discrete (categorical) and continuous data are handled automatically.

    Parameters
    ----------
    transcriptome : AnnData, MuData, pd.DataFrame, or zarr.Group
        Spatial transcriptome data.
    color : str or list of str, optional
        Gene name(s) or ``obs`` column(s) to plot. If a list, creates a
        multi-panel figure.
    geometry : pd.DataFrame, optional
        Spatial coordinates with 'x' and 'y' columns. Required for
        DataFrame and Zarr inputs; extracted automatically from MuData.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. Only valid for single-gene plots.
    title : str or list of str, optional
        Title(s) for the panel(s).
    annotation : pd.DataFrame, optional
        Morphological landmark positions to overlay as circles.
    mirrored : bool
        If True, flip the tissue map horizontally.
    colorbar : bool
        If True, display a colorbar.
    colorbar_horizontal : bool
        If True, colorbar is horizontal; otherwise vertical.
    spines : bool
        If True, show axis spines.
    cropped : bool
        If True, crop to tissue boundaries.
    show : bool
        If True, call ``plt.show()``. If False, return the axes.
    nr : int
        Number of subplot rows for multi-gene layout.
    nc : int
        Number of subplot columns for multi-gene layout.
    fr : float
        Figure height scaling factor (per row).
    fc : float
        Figure width scaling factor (per column).
    n_max : int, optional
        Maximum number of genes to plot (truncates the list).
    **imshow_kwargs
        Additional keyword arguments passed to ``ax.imshow()``.

    Returns
    -------
    Axes or list of Axes or None
        The matplotlib axes if ``show=False``, otherwise None.
    """
    if annotation is not None:
        annotation = annotation.copy()

    if isinstance(transcriptome, zarr.Group):
        transcriptome = read_zarr(transcriptome)

    # Handling the type of transcriptome
    if isinstance(transcriptome, MuData):
        transcriptome = transcriptome["st"]

    if isinstance(transcriptome, AnnData):
        geometry = transcriptome.obs.loc[:, ["x", "y"]]

        if "morphological_marks" in transcriptome.uns:
            annotation = transcriptome.uns["morphological_marks"]

        obs = transcriptome.obs
        transcriptome = transcriptome.to_df()
        transcriptome.columns = remove_index_prefix(transcriptome.columns)
        transcriptome = pd.concat([transcriptome, obs], axis=1)

    elif isinstance(transcriptome, pd.DataFrame):
        # If geometry is not provided, but the DataFrame index is a MultiIndex with "x" and "y", extract it.
        if geometry is None:
            if isinstance(transcriptome.index, pd.MultiIndex) and set(
                transcriptome.index.names
            ) == {"x", "y"}:
                geometry = transcriptome.index.to_frame(index=False)
            else:
                raise ValueError(
                    "The geometry of the tissue must be provided if the transcriptome is a pd.DataFrame "
                    "and its index is not a MultiIndex with levels 'x' and 'y'."
                )

    color = [color] if isinstance(color, str) or color is None else list(color)
    if title is not None:
        title = [title] if isinstance(title, str) else list(title)

    # Layout
    if ax is None:
        if nr * nc >= len(color):
            fig, ax = plt.subplots(nr, nc, figsize=(fr, fc))
        else:
            fig, ax = plt.subplots(len(color), 1, figsize=(fr, fc))
    if len(color) == 1:
        ax = [ax]

    if len(color) != len(ax):
        raise ValueError("The number of genes and axes must be the same")

    # Plotting

    for i in range(len(color)):

        if annotation is not None:
            # Handling type of the Series
            n_annot = annotation.shape[1]
            annotations = annotation.copy()
            annotations.columns = ["m" + str(i) for i in range(1, n_annot + 1)]

        series = transcriptome[color[i]]
        # Treat as discrete if pandas categorical OR small set of integer labels
        discrete_threshold = n_max if n_max is not None else 50
        is_discrete = isinstance(series.dtype, pd.CategoricalDtype) or (
            is_integer_dtype(series) and series.dropna().nunique() <= discrete_threshold
        )

        labels = None
        n = None

        if is_discrete:
            if isinstance(series.dtype, pd.CategoricalDtype):
                numeric_series, labels = convert_to_numeric(series)
                labels = [str(lab) for lab in labels]
            else:
                # Map integer labels to consecutive indices starting at 0
                cats = np.sort(series.dropna().astype(int).unique())
                labels = [str(c) for c in cats]
                cat_to_index = {c: idx for idx, c in enumerate(cats)}
                numeric_series = series.map(
                    lambda v: cat_to_index.get(int(v)) if pd.notna(v) else np.nan
                )

            expr = to_matrix_form(numeric_series, geometry)
        else:
            expr = to_matrix_form(transcriptome[color[i]], geometry)

            if mirrored is True:

                cropped = True
                # Before mirroring, crop the indices that are not needed
                x_min, x_max, y_min, y_max = get_cropping_boundaries(expr)
                depth_y = y_max - y_min

                # Matrix rows are geometry "x" but cropping calls them "y", hence [y_min, x_min]
                expr = to_matrix_form(
                    transcriptome[color[i]], geometry - [y_min, x_min]
                )

                expr_mirr = np.flipud(expr)
                expr = np.concatenate((expr_mirr, expr), axis=0)

                if annotation is not None:
                    for m in range(1, n_annot + 1):
                        annotations.loc[0, "-m" + str(m)] = (
                            annotations.loc[0, "m" + str(m)] - x_min
                        )
                        annotations.loc[1, "-m" + str(m)] = (
                            -annotations.loc[1, "m" + str(m)] + y_min + depth_y
                        )
                        annotations.loc[0, "m" + str(m)] = (
                            annotations.loc[0, "m" + str(m)] - x_min
                        )
                        annotations.loc[1, "m" + str(m)] = (
                            annotations.loc[1, "m" + str(m)] - y_min + depth_y + 1
                        )

        if is_discrete:
            n = len(np.unique(expr[~np.isnan(expr)]))
            if n_max is not None:
                if n_max < n:
                    raise ValueError(
                        f"n_max ({n_max}) is less than the number of unique categories ({n})."
                    )
                n = n_max

            # Create boundaries for categories
            bounds = np.arange(n + 1)

            # Build kwargs safely to avoid duplicate keys
            _imshow_kwargs = dict(imshow_kwargs)

            # Create the colormap only if not provided
            if "cmap" in _imshow_kwargs:
                cmap = _imshow_kwargs["cmap"]
            else:
                cmap = matplotlib.colormaps["tab20"].resampled(n)
                _imshow_kwargs["cmap"] = cmap

            # Create norm with boundaries only if not provided
            if "norm" not in _imshow_kwargs:
                _imshow_kwargs["norm"] = BoundaryNorm(bounds, cmap.N)

            im = ax[i].imshow(expr, aspect="auto", **_imshow_kwargs)
        else:

            im = ax[i].imshow(expr, aspect="auto", **imshow_kwargs)

        if cropped is True:
            x_min, x_max, y_min, y_max = get_cropping_boundaries(expr)
            ax[i].set_xlim((x_min, x_max))
            ax[i].set_ylim((y_max, y_min))

        if colorbar is True:
            # Remove the colorbar of the previous plot
            images = ax[i].images
            for image in images:
                cbar = image.colorbar
                if cbar is not None:
                    cbar.remove()

            if is_discrete:
                orientation = "horizontal" if colorbar_horizontal else "vertical"
                # Ensure colorbar uses the same bin boundaries so ticks center perfectly
                cbar = plt.colorbar(
                    im,
                    ax=ax[i],
                    orientation=orientation,
                    aspect=10,
                    boundaries=np.arange(n + 1),
                    ticks=np.arange(n) + 0.5,
                )

                # Set labels centered on bins
                if colorbar_horizontal:
                    cbar.ax.set_xticklabels(
                        labels if labels is not None else [str(i) for i in range(n)]
                    )
                else:
                    cbar.ax.set_yticklabels(
                        labels if labels is not None else [str(i) for i in range(n)]
                    )
            else:
                if colorbar_horizontal:
                    plt.colorbar(im, ax=ax[i], orientation="horizontal")
                else:
                    plt.colorbar(im, ax=ax[i])

        if title is not None:
            ax[i].set_title(title[i])

        if not spines:
            ax[i].set_yticks([])
            ax[i].set_xticks([])

            ax[i].spines["top"].set_visible(False)
            ax[i].spines["left"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)

        if annotation is not None:
            for m in range(annotations.shape[1]):
                circ1 = Circle(
                    (annotations.iloc[0, m], annotations.iloc[1, m]),
                    0.3,
                    ec="red",
                    fc="red",
                )
                ax[i].add_patch(circ1)

        ax[i].set_aspect("equal")
    if show:
        plt.show()
    else:
        return ax
