"""Recapitulative report figure for spatial transcriptome optimization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mudata import MuData
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..utils import remove_index_prefix
from ._geometricplot import geometric


def _calculate_correlation_map(projection, reference, geometry, genes):
    """Compute per-position correlation map between projection and reference."""
    df_location_correlation = projection[genes].corrwith(reference[genes], axis=1)

    x = geometry["x"].values.astype(int)
    y = geometry["y"].values.astype(int)

    map_correlation = np.empty((np.nanmax(x) + 1, np.nanmax(y) + 1))
    map_correlation[:] = np.nan
    map_correlation[x, y] = df_location_correlation.values
    return map_correlation


def _plot_matrix_distribution(distribution_matrix, ax=None, **hist_kwargs):
    """Plot distribution of a matrix (histogram with viridis coloring)."""
    distribution_matrix = distribution_matrix[~np.isnan(distribution_matrix)].flatten()

    if ax is None:
        ax = plt.gca()

    n, bins, patches = ax.hist(distribution_matrix, **hist_kwargs)
    n = n.astype("int")

    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.viridis(bins[i]))

    ax.axvline(distribution_matrix.mean(), color="k", linestyle="dashed", linewidth=1)
    ax.text(
        distribution_matrix.mean() * 1.01,
        max(n) * 1.01,
        "Mean: {:.2f}".format(distribution_matrix.mean()),
    )

    return ax, n


def _plot_correlations_histogram(correlations, ax=None, **hist_kwargs):
    """Plot horizontal bar chart of per-gene correlations."""
    labels = list(correlations.index)
    ordering = list(range(len(labels)))
    values = correlations.values.ravel()

    if ax is None:
        ax = plt.gca()

    ax.barh(ordering, values, align="center", **hist_kwargs)
    ax.set_yticks(ordering)
    ax.set_yticklabels(labels)

    ax.axvline(
        np.mean(values), color="black", linestyle="dashed", linewidth=1, label="Mean"
    )
    ax.axvline(np.median(values), color="black", linewidth=1, label="Median")

    ax.set_xlabel("Correlation Value")
    ax.set_ylabel("Gene")

    return ax


def _plot_gene_correlation(projection, reference, gene, ax=None, lowess_fraction=0.66):
    """Scatter plot of projection vs reference expression for a single gene."""
    if ax is None:
        ax = plt.gca()

    x = projection[gene].values
    y = reference[gene].values

    good_x = np.where(~np.isnan(x))
    good_y = np.where(~np.isnan(y))
    good_locations = np.intersect1d(good_x, good_y)

    x = x[good_locations]
    y = y[good_locations]

    ax.scatter(x, y, s=0.1, c="black")
    ax.set_title(gene)
    ax.set_xlabel("Mapping expression")
    ax.set_ylabel("Reference")

    lowess_fit = lowess(y, x, lowess_fraction)
    ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], c="blue")

    return ax


def plot_recap(
    spatial_transcriptome: MuData,
    atlas,
    validation,
    include_notum_expression=True,
    path=None,
):
    """Generate a recapitulative report figure for spatial transcriptome quality.

    Produces a multi-panel PDF/figure with validation statistics tables,
    per-position correlation maps, gene correlation scatterplots with LOWESS
    fits, and per-gene histograms. Requires :func:`~vgep.spatial_transcriptome.transform`
    and :func:`~vgep.spatial_transcriptome.assess` to have been run on the transcriptome.

    Parameters
    ----------
    spatial_transcriptome : MuData
        Spatial transcriptome with ``"st"``, ``"sc"``, and ``"meta"``
        modalities. Must contain ``uns["results_recap"]`` and
        ``uns["correlations_genes_training"]``.
    atlas : pd.DataFrame
        Training atlas used for the reconstruction.
    validation : pd.DataFrame
        Held-out validation gene expression dataset.
    include_notum_expression : bool
        If True, plot individual gene expression maps on the tissue.
    path : str, optional
        Path to save the figure as PDF. If None, the figure is displayed.

    Returns
    -------
    matplotlib.figure.Figure
        The generated report figure.
    """
    atlas.index = atlas.index.astype(str)
    validation.index = validation.index.astype(str)

    mapping = pd.DataFrame(
        data=spatial_transcriptome["st"].X,
        index=spatial_transcriptome["st"].obs_names,
        columns=remove_index_prefix(spatial_transcriptome["st"].var_names),
    )

    single_cells = spatial_transcriptome["sc"].to_df()
    metadata = spatial_transcriptome["st"].uns
    geometry = spatial_transcriptome["st"].obs.loc[:, ["x", "y"]]
    morphological_marks = spatial_transcriptome["st"].uns.get("morphological_marks")

    genes_train = np.intersect1d(mapping.columns, atlas.columns)
    genes_valid = np.intersect1d(mapping.columns, validation.columns)
    n_genes = len(genes_train) + len(genes_valid)

    if include_notum_expression:
        fig = plt.figure(figsize=(20, 4 * n_genes + 35))
    else:
        fig = plt.figure(figsize=(20, 35))

    n_figures = 8
    if include_notum_expression:
        n_figures += n_genes

    heights = [0.3, 0.3, 0.1, 0.3, 0.1, 0.3, 0.4, 0.4]
    if include_notum_expression:
        heights += [0.3 for _ in range(n_genes)]

    gs = fig.add_gridspec(n_figures, 3, height_ratios=heights)

    j = 6
    ax_recap = fig.add_subplot(gs[0, :])
    ax_params = fig.add_subplot(gs[1, :])
    ax_train = fig.add_subplot(gs[2, :])
    ax_hist_train = fig.add_subplot(gs[3, :])
    ax_valid = fig.add_subplot(gs[4, :])
    ax_hist_valid = fig.add_subplot(gs[5, :])

    ax_recap.axis("off")
    ax_params.axis("off")
    ax_train.axis("off")
    ax_valid.axis("off")

    ax_corr_train_map = fig.add_subplot(gs[j, :2])
    ax_corr_train_hist = fig.add_subplot(gs[j, 2:])
    j += 1
    ax_corr_valid_map = fig.add_subplot(gs[j, :2])
    ax_corr_valid_hist = fig.add_subplot(gs[j, 2:])
    j += 1

    if include_notum_expression:
        ax_notum_map_mapping = [fig.add_subplot(gs[i + j, 0]) for i in range(n_genes)]
        ax_notum_map_reference = [fig.add_subplot(gs[i + j, 1]) for i in range(n_genes)]
        ax_correlation_scatter = [fig.add_subplot(gs[i + j, 2]) for i in range(n_genes)]

    # Tables
    results_recap = metadata["results_recap"]
    recap = ax_recap.table(
        results_recap.values.round(2).astype(str).tolist(),
        rowLabels=results_recap.index,
        colLabels=results_recap.columns,
        loc="center",
    )
    recap.auto_set_font_size(False)
    recap.set_fontsize(24)
    recap.scale(1, 4)

    ax_params.text(0, 0.5, f"Cutoff : {metadata['cutoff']}", fontsize=30)
    ax_params.text(
        0, 0.25,
        f"number of cells selected : {str(int((1 - metadata['cutoff']) * single_cells.shape[0]))}",
        fontsize=30,
    )

    train_table = ax_train.table(
        metadata["correlations_genes_training"].values.T.round(2).astype(str).tolist(),
        rowLabels=metadata["correlations_genes_training"].columns,
        colLabels=metadata["correlations_genes_training"].index,
        loc="center",
        colColours=["#0099CC" for _ in metadata["correlations_genes_training"].index],
    )
    train_table.auto_set_font_size(False)
    train_table.set_fontsize(20)
    train_table.scale(1, 2)
    ax_train.set_title("Correlations between mapping and training Genes")

    valid_table = ax_valid.table(
        metadata["correlations_genes_validation"].values.T.round(2).astype(str).tolist(),
        colLabels=metadata["correlations_genes_validation"].index,
        loc="center",
        colColours=["green" for _ in metadata["correlations_genes_validation"].index],
    )
    valid_table.auto_set_font_size(False)
    valid_table.set_fontsize(20)
    valid_table.scale(1, 2)
    ax_valid.set_title("Correlations between mapping and validation genes")

    _plot_correlations_histogram(
        metadata["correlations_genes_training"],
        ax=ax_hist_train, color="#0099CC", alpha=0.7,
    )
    _plot_correlations_histogram(
        metadata["correlations_genes_validation"],
        ax=ax_hist_valid, color="green", alpha=0.7,
    )

    map_correlation_train = _calculate_correlation_map(
        mapping, atlas, geometry, metadata["training_genes"]
    )
    map_correlation_valid = _calculate_correlation_map(
        mapping, validation, geometry, metadata["validation_genes"]
    )

    mini = min(np.nanmin(map_correlation_train), np.nanmin(map_correlation_valid))
    maxi = max(np.nanmax(map_correlation_train), np.nanmax(map_correlation_valid))
    im1 = ax_corr_train_map.imshow(map_correlation_train, vmin=mini, vmax=maxi)
    im2 = ax_corr_valid_map.imshow(map_correlation_valid, vmin=mini, vmax=maxi)
    plt.colorbar(im1, ax=ax_corr_train_map)
    plt.colorbar(im2, ax=ax_corr_valid_map)

    ax_corr_train_map.set_title("Projection and Training Atlas Map")
    ax_corr_valid_map.set_title("Projection and Validation Atlas Map")
    ax_corr_train_map.set_xlabel("X coordinates ")
    ax_corr_valid_map.set_xlabel("X coordinates ")
    ax_corr_train_map.set_ylabel("Y coordinates")
    ax_corr_valid_map.set_ylabel("Y coordinates")
    ax_corr_train_map.set_ylim(40, 0)

    ax1 = ax_corr_train_map.twinx()
    ax1.set_ylabel("Correlation value")
    ax1.set_yticks([])
    ax1.yaxis.set_label_coords(1.17, 0.5)

    ax2 = ax_corr_valid_map.twinx()
    ax2.set_ylabel("Correlation value")
    ax2.set_yticks([])
    ax2.yaxis.set_label_coords(1.17, 0.5)

    _, occurences_train = _plot_matrix_distribution(
        map_correlation_train, ax=ax_corr_train_hist, bins=100, range=[mini, maxi]
    )
    _, occurences_valid = _plot_matrix_distribution(
        map_correlation_valid, ax=ax_corr_valid_hist, bins=100, range=[mini, maxi]
    )

    max_occurences = 1.1 * max((max(occurences_valid), max(occurences_train)))
    ax_corr_train_hist.set_ylim(0, max_occurences)
    ax_corr_train_hist.set_ylabel("Occurences of the correlation value")
    ax_corr_train_hist.set_xlabel("Correlation value")
    ax_corr_valid_hist.set_ylim(0, max_occurences)
    ax_corr_valid_hist.set_ylabel("Occurences of the correlation value")
    ax_corr_valid_hist.set_xlabel("Correlation value")

    if include_notum_expression:
        for i in range(len(genes_train)):
            geometric(
                spatial_transcriptome["st"], color=genes_train[i],
                ax=ax_notum_map_mapping[i], show=False, cropped=True,
            )
            geometric(
                atlas, color=genes_train[i], geometry=geometry,
                ax=ax_notum_map_reference[i], annotation=morphological_marks,
                show=False, cropped=True,
            )
            _plot_gene_correlation(
                mapping, atlas, genes_train[i], ax=ax_correlation_scatter[i]
            )

        for i in range(len(genes_valid)):
            geometric(
                spatial_transcriptome["st"], color=genes_valid[i],
                ax=ax_notum_map_mapping[i + len(genes_train)], cropped=True, show=False,
            )
            geometric(
                validation, color=genes_valid[i], geometry=geometry,
                annotation=morphological_marks,
                ax=ax_notum_map_reference[i + len(genes_train)], cropped=True, show=False,
            )
            _plot_gene_correlation(
                mapping, validation, genes_valid[i],
                ax=ax_correlation_scatter[i + len(genes_train)],
            )

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format="pdf")

    return fig
