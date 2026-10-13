import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import numpy as np

from vgep.evaluation.cdf import plot_cdf_curves


def test_returns_figure_with_one_line_per_method():
    rng = np.random.default_rng(0)
    corr = {
        "vgep": rng.normal(size=20),
        "novosparc": rng.normal(size=20),
        "perler": rng.normal(size=20),
    }
    colors = {"vgep": "#1f77b4", "novosparc": "#ff7f0e", "perler": "#2ca02c"}
    fig = plot_cdf_curves(corr, colors, title="T")
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    cdf_lines = [l for l in ax.lines if l.get_label() in corr]
    assert len(cdf_lines) == 3
    plt.close(fig)


def test_honors_provided_axes():
    fig, ax = plt.subplots()
    corr = {"a": np.array([0.1, 0.5, 0.9])}
    colors = {"a": "red"}
    result_fig = plot_cdf_curves(corr, colors, ax=ax)
    assert result_fig is fig
    plt.close(fig)


def test_median_lines_drawn_when_show_medians_true():
    corr = {"a": np.array([0.1, 0.5, 0.9]), "b": np.array([0.0, 0.4, 0.8])}
    colors = {"a": "red", "b": "blue"}
    fig = plot_cdf_curves(corr, colors, show_medians=True)
    ax = fig.axes[0]
    # Expect at least 2 vertical median lines (one per method) in addition to the 2 CDF lines
    assert len(ax.lines) >= 4
    plt.close(fig)


def test_empty_method_is_skipped_gracefully():
    corr = {"a": np.array([0.1, 0.5]), "b": np.array([])}
    colors = {"a": "red", "b": "blue"}
    fig = plot_cdf_curves(corr, colors)
    ax = fig.axes[0]
    cdf_lines = [l for l in ax.lines if l.get_label() in corr]
    assert len(cdf_lines) == 1  # b skipped
    plt.close(fig)


def test_nan_values_dropped_from_cdf():
    corr = {"a": np.array([0.1, np.nan, 0.5, np.nan, 0.9])}
    colors = {"a": "red"}
    fig = plot_cdf_curves(corr, colors)
    ax = fig.axes[0]
    line = [l for l in ax.lines if l.get_label() == "a"][0]
    x = line.get_xdata()
    assert len(x) == 3
    plt.close(fig)
