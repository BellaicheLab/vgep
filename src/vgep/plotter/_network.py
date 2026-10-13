"""Gene regulatory network graph visualization."""

import logging

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

_NODE_STYLE = {
    "cluster": {"shape": "o", "size": 1800, "color": "red"},
    "tf": {"shape": "s", "size": 1100, "color": "green"},
    "target": {"shape": "^", "size": 700, "color": "blue"},
    "unknown": {"shape": "o", "size": 400, "color": "gray"},
}

_FONT_SIZES = {"cluster": 14, "tf": 10, "target": 9, "unknown": 8}


def _compute_layout(H, layout):
    """Return a (pos, target_modules) dict. ``target_modules`` is used by
    the ``shell`` layout to group targets around their TF's Leiden module."""
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(H)
    if layout == "shell":
        tfs = [n for n, d in H.nodes(data=True) if d.get("node_type") == "tf"]
        targets = [n for n, d in H.nodes(data=True) if d.get("node_type") != "tf"]
        shells = [shell for shell in (tfs, targets) if shell]
        return (
            nx.shell_layout(H, nlist=shells) if shells else nx.spring_layout(H, seed=42)
        )
    # Default: spring, tuned for crowded GRNs
    n = H.number_of_nodes()
    k = 6.0 / np.sqrt(max(n, 1))  # 6x the networkx default spacing
    return nx.spring_layout(H, k=k, iterations=700, seed=42)


def plot_grn(
    G,
    ax=None,
    title=None,
    figsize=(16, 12),
    color_by="node_type",
    hide_cluster_seed=False,
    label="all",
    layout="spring",
    save_path=None,
):
    """Plot a gene regulatory network with node-type differentiation.

    Nodes are drawn with distinct shapes per ``node_type`` attribute
    (``'cluster'`` / ``'tf'`` / ``'target'``). Colours can follow either
    the node type or the Leiden ``cluster`` attribute written by
    :func:`vgep.grn.network.annotate_network_with_leiden_clusters`.

    Parameters
    ----------
    G : networkx.Graph
        The graph to visualise. Nodes should carry a ``node_type`` attribute.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If ``None``, a new figure is created.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size (only used when ``ax`` is ``None``).
    color_by : {"node_type", "leiden"}
        ``"node_type"`` (default) colours by role (cluster/tf/target).
        ``"leiden"`` colours by the Leiden ``cluster`` attribute; nodes
        without an annotation fall back to grey.
    hide_cluster_seed : bool
        Strip nodes with ``node_type == "cluster"`` before layout.
    label : {"all", "tf", "none"}
        Which nodes to label. Labels carry a white halo for readability.
    layout : {"spring", "kamada_kawai", "shell"}
        Node positioning algorithm. ``"spring"`` (tuned) is good for
        general use; ``"kamada_kawai"`` suits small graphs; ``"shell"``
        puts TFs in an inner ring and targets in an outer ring.
    save_path : str or Path, optional
        If given, save the figure. Only meaningful when ``ax`` is ``None``.
    """
    if color_by not in ("node_type", "leiden"):
        raise ValueError("color_by must be 'node_type' or 'leiden'")
    if label not in ("all", "tf", "none"):
        raise ValueError("label must be 'all', 'tf', or 'none'")
    if layout not in ("spring", "kamada_kawai", "shell"):
        raise ValueError("layout must be 'spring', 'kamada_kawai', or 'shell'")

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)

    H = G.copy()
    if hide_cluster_seed:
        seeds = [n for n, d in H.nodes(data=True) if d.get("node_type") == "cluster"]
        H.remove_nodes_from(seeds)

    if H.number_of_nodes() == 0:
        ax.set_title(f"{title} — empty" if title else "empty")
        ax.axis("off")
        return

    pos = _compute_layout(H, layout)

    if color_by == "leiden":
        modules = sorted(
            {
                d.get("cluster")
                for _, d in H.nodes(data=True)
                if d.get("cluster") is not None
            }
        )
        palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(modules), 1)))
        module_color = {m: palette[i] for i, m in enumerate(modules)}

    def node_colour(n):
        if color_by == "leiden":
            m = H.nodes[n].get("cluster")
            return module_color.get(m, (0.6, 0.6, 0.6, 1.0))
        style = _NODE_STYLE.get(
            H.nodes[n].get("node_type", "unknown"), _NODE_STYLE["unknown"]
        )
        return style["color"]

    # Edges, split by type so specificity (cluster→TF, weight=1) stays visible
    # even when regulation edges carry much larger GENIE3 importance weights.
    specificity = [(u, v) for u, v, d in H.edges(data=True) if d.get("edge_type") == "specificity"]
    regulation  = [(u, v) for u, v, d in H.edges(data=True) if d.get("edge_type") != "specificity"]

    if regulation:
        reg_weights = [H[u][v].get("weight", 1) for u, v in regulation]
        max_reg = max(reg_weights)
        reg_widths = [2 * w / max_reg for w in reg_weights]
        nx.draw_networkx_edges(
            H, pos, ax=ax, edgelist=regulation,
            width=reg_widths, alpha=0.5, edge_color="#888888",
        )

    if specificity:
        nx.draw_networkx_edges(
            H, pos, ax=ax, edgelist=specificity,
            width=1.4, alpha=0.7, edge_color="#cc2222", style="dashed",
        )

    # Nodes, grouped by shape
    by_shape = {}
    for n in H.nodes():
        style = _NODE_STYLE.get(
            H.nodes[n].get("node_type", "unknown"), _NODE_STYLE["unknown"]
        )
        by_shape.setdefault(style["shape"], []).append((n, style["size"]))

    for shape, items in by_shape.items():
        nodes = [n for n, _ in items]
        sizes = [s for _, s in items]
        nx.draw_networkx_nodes(
            H,
            pos,
            ax=ax,
            nodelist=nodes,
            node_color=[node_colour(n) for n in nodes],
            node_size=sizes,
            node_shape=shape,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
        )

    # Labels — drawn centred on each node, with a white halo so they stay
    # readable when they overflow the node glyph.
    if label != "none":
        if label == "tf":
            keep = [n for n, d in H.nodes(data=True) if d.get("node_type") == "tf"]
        else:
            keep = list(H.nodes())

        halo = [path_effects.withStroke(linewidth=2.5, foreground="white")]

        for ntype in ("cluster", "tf", "target"):
            nodes = [n for n in keep if H.nodes[n].get("node_type") == ntype]
            if not nodes:
                continue
            texts = nx.draw_networkx_labels(
                H,
                pos,
                {n: n for n in nodes},
                font_size=_FONT_SIZES[ntype],
                font_weight="bold",
                ax=ax,
            )
            for t in texts.values():
                t.set_path_effects(halo)

    # Legend (node-type colour key only when colour map is by node_type)
    if color_by == "node_type":
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker=_NODE_STYLE["cluster"]["shape"],
                color="w",
                markerfacecolor=_NODE_STYLE["cluster"]["color"],
                markersize=12,
                label="Cluster",
            ),
            plt.Line2D(
                [0],
                [0],
                marker=_NODE_STYLE["tf"]["shape"],
                color="w",
                markerfacecolor=_NODE_STYLE["tf"]["color"],
                markersize=10,
                label="TF",
            ),
            plt.Line2D(
                [0],
                [0],
                marker=_NODE_STYLE["target"]["shape"],
                color="w",
                markerfacecolor=_NODE_STYLE["target"]["color"],
                markersize=8,
                label="Target",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")

    if owns_figure:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.debug("Saved figure to %s", save_path)
