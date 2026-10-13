"""Tau score and gene specificity plots."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_scores(
    combinations_df,
    cluster_name,
    tau_scores,
    fig_size=(5, 6),
    title=None,
    color="#1f77b4",
    save_path=None,
    dpi=300,
):
    """Publication-ready barplot of genes sorted by decreasing scores for a cluster.

    Parameters
    ----------
    combinations_df : pd.DataFrame
        Binary matrix (genes x clusters) indicating cluster specificity.
    cluster_name : str or int
        Cluster column to filter by.
    tau_scores : pd.Series
        Scores indexed by gene name.
    fig_size : tuple
        Figure size ``(width, height)`` in inches.
    title : str, optional
        Plot title. Defaults to ``'Cluster {cluster_name} Specific Genes'``.
    color : str
        Bar colour.
    save_path : str or Path, optional
        Path to save the figure.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    filtered_genes = combinations_df[combinations_df[cluster_name] == 1].index
    gene_scores = tau_scores.loc[tau_scores.index.intersection(filtered_genes)]
    gene_scores_sorted = gene_scores.sort_values(ascending=True)

    auto_height = max(8, len(gene_scores_sorted) * 0.4)
    actual_fig_size = (fig_size[0], min(fig_size[1], auto_height))

    fig, ax = plt.subplots(figsize=actual_fig_size)
    gene_scores_sorted.plot(kind="barh", color=color, ax=ax)

    if title is None:
        title = f"Cluster {cluster_name} Specific Genes"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("$\\tau$ score", fontsize=12)
    ax.set_ylabel("Gene", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f8f8f8")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_tau_distribution(
    tau_scores, cutoff=None, title=None, bins=40, fig_size=(6, 4), save_path=None
):
    """Distribution plot of tau scores.

    Parameters
    ----------
    tau_scores : pd.Series
        Tau scores indexed by gene name.
    cutoff : float, optional
        Vertical line indicating the selection cutoff.
    title : str, optional
        Plot title. Defaults to ``'Tau score distribution'``.
    bins : int
        Number of histogram bins.
    fig_size : tuple
        Figure size in inches.
    save_path : str or Path, optional
        Path to save the figure (SVG). If *None*, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=fig_size)
    sns.histplot(
        tau_scores.dropna().values,
        bins=bins,
        kde=False,
        stat="count",
        color="#4C72B0",
        edgecolor="white",
        alpha=0.9,
        ax=ax,
    )
    if cutoff is not None:
        ax.axvline(cutoff, color="crimson", linestyle="--", linewidth=1.5)
        ymax = ax.get_ylim()[1]
        ax.text(
            cutoff,
            0.98 * ymax,
            f"cutoff = {cutoff:.2f}",
            color="crimson",
            ha="right",
            va="top",
            fontsize=5,
        )
    if title is None:
        title = "Tau score distribution"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("$\\tau$ score", fontsize=7)
    ax.set_ylabel("Number of genes", fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f8f8f8")
    ax.tick_params(axis="x", labelsize=5)
    ax.tick_params(axis="y", labelsize=5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="svg", dpi=1200, bbox_inches="tight")

    return fig
