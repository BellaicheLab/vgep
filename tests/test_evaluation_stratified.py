import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vgep.evaluation.stratified import plot_protein_vs_fish_cdf


def test_returns_figure_with_two_cdf_lines():
    gene_names = ["C15", "caup", "Dl", "slow", "sli"]
    protein = ["C15", "caup", "Dl"]
    fish = ["slow", "sli"]
    corr = np.array([0.9, 0.8, 0.7, 0.5, 0.4])
    fig = plot_protein_vs_fish_cdf(corr, gene_names, protein, fish)
    ax = fig.axes[0]
    labels = [l.get_label() for l in ax.lines]
    assert "protein" in labels
    assert "fish" in labels
    plt.close(fig)


def test_handles_missing_genes():
    # Genes in the protein list that are not in gene_names are silently skipped
    gene_names = ["C15", "slow"]
    protein = ["C15", "caup"]  # caup not in gene_names
    fish = ["slow"]
    corr = np.array([0.9, 0.5])
    fig = plot_protein_vs_fish_cdf(corr, gene_names, protein, fish)
    ax = fig.axes[0]
    lines_by_label = {l.get_label(): l for l in ax.lines if l.get_label() in ("protein", "fish")}
    assert len(lines_by_label["protein"].get_xdata()) == 1
    plt.close(fig)


def test_adds_stats_annotation():
    # With enough samples, stats annotation (KS + Mann-Whitney) should exist as a text artist
    rng = np.random.default_rng(2)
    protein = [f"p{i}" for i in range(10)]
    fish = [f"f{i}" for i in range(10)]
    gene_names = protein + fish
    corr = rng.normal(size=20)
    fig = plot_protein_vs_fish_cdf(corr, gene_names, protein, fish)
    ax = fig.axes[0]
    texts = ax.texts
    assert any("KS" in t.get_text() or "MWU" in t.get_text() for t in texts)
    plt.close(fig)
