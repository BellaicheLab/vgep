import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vgep.evaluation.survival import plot_survival_curves


def test_returns_figure_with_one_line_per_method():
    rng = np.random.default_rng(1)
    corr = {
        "vgep": rng.normal(size=30),
        "novosparc": rng.normal(size=30),
    }
    colors = {"vgep": "#1f77b4", "novosparc": "#ff7f0e"}
    fig = plot_survival_curves(corr, colors)
    ax = fig.axes[0]
    sur_lines = [l for l in ax.lines if l.get_label() in corr]
    assert len(sur_lines) == 2
    plt.close(fig)


def test_survival_endpoints():
    # Ascending sorted survival curve starts near 1 and ends near 1/n
    vals = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    fig = plot_survival_curves({"a": vals}, {"a": "red"}, show_medians=False)
    ax = fig.axes[0]
    line = [l for l in ax.lines if l.get_label() == "a"][0]
    y = line.get_ydata()
    assert np.isclose(y[0], 1.0)
    assert np.isclose(y[-1], 1.0 / len(vals))
    plt.close(fig)


def test_nan_values_dropped():
    vals = np.array([0.1, np.nan, 0.5])
    fig = plot_survival_curves({"a": vals}, {"a": "red"})
    ax = fig.axes[0]
    line = [l for l in ax.lines if l.get_label() == "a"][0]
    assert len(line.get_xdata()) == 2
    plt.close(fig)
