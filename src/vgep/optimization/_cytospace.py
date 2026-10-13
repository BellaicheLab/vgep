"""CytoSPACE parameter optimization via a balanced linear-assignment placement.

CytoSPACE (Vahid et al., Nature Biotechnology 2023) in single-type mode: the
atlas is one pseudo cell type, so cells are matched to bins by a balanced linear
assignment under a correlation cost, solved with
``scipy.optimize.linear_sum_assignment``. Inputs use the benchmark's normalized
scales (atlas percentile-rescaled, single cells log-normalized).

``generate_cytospace`` runs one placement; ``optimize_cytospace`` grid-searches
the parameters, scored by validation-gene Pearson r (mean + median).
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from vgep.evaluation.correlations import per_gene_pearson_r

logger = logging.getLogger(__name__)


def _per_gene_r(pred, atlas_df, genes):
    """Per-gene Pearson r between projected expression and the atlas over bins.

    ``pred`` is (n_bins, len(genes)) aligned to ``genes``; ``atlas_df`` is the
    (n_bins, G) atlas restricted to the same bins. NaN bins are dropped.
    """
    ref = np.column_stack([
        atlas_df[g].to_numpy(dtype=np.float64) if g in atlas_df.columns else np.full(len(pred), np.nan)
        for g in genes
    ])
    return dict(zip(genes, per_gene_pearson_r(pred, ref, gene_axis=1).astype(float)))


def _correlation_cost(atlas_bins, sc_cells, distance_metric):
    """LAP cost ``-corr(bin, cell)`` over the gene axis: (n_bins, n_cells).

    Correlation is computed per (bin, cell) profile across the shared genes;
    degenerate (zero-variance) profiles yield correlation 0 (neutral cost).
    """
    X = np.asarray(atlas_bins, dtype=np.float64)
    S = np.asarray(sc_cells, dtype=np.float64)
    if distance_metric == "spearman":
        X = rankdata(X, axis=1)
        S = rankdata(S, axis=1)
    Xz = X - X.mean(axis=1, keepdims=True)
    Sz = S - S.mean(axis=1, keepdims=True)
    Xz /= np.linalg.norm(Xz, axis=1, keepdims=True)
    Sz /= np.linalg.norm(Sz, axis=1, keepdims=True)
    corr = np.nan_to_num(Xz @ Sz.T, nan=0.0)
    return -corr


def _solve_placement(cost, n_bins, cells_per_bin):
    """Balanced assignment of cells to bins -> P_to_cells (n_bins, n_cells).

    ``cost`` is (n_bins, n_cells). Each bin is expanded into ``k`` slots and the
    LAP assigns one cell per slot, so every bin receives exactly ``k`` cells and
    ``P_to_cells`` is row-stochastic.
      - ``cells_per_bin="single"``: k = 1.
      - ``cells_per_bin="dense"``  : k = n_cells // n_bins (place all cells once).
      - ``cells_per_bin=<int>``    : k = that integer (fixed cells per bin).

    When there are at least as many cells as slots the assignment uses *distinct*
    cells (rectangular LAP over all cells). If bins outnumber cells (rare, small
    masks) cells are reused via a deterministic tiled pool — CytoSPACE's own
    "duplicates" fallback — keeping the result well-defined.
    """
    n_cells = cost.shape[1]
    if cells_per_bin == "single":
        k = 1
    elif cells_per_bin == "dense":
        k = max(1, n_cells // n_bins)
    else:
        k = max(1, int(cells_per_bin))
    n_slots = n_bins * k
    slot_bin = np.repeat(np.arange(n_bins), k)

    P = np.zeros((n_bins, n_cells), dtype=np.float64)
    if n_slots <= n_cells:
        slot_idx, cell_idx = linear_sum_assignment(cost[slot_bin, :])
        P[slot_bin[slot_idx], cell_idx] = 1.0
    else:
        pool = np.arange(n_slots) % n_cells  # deterministic reuse of cells
        slot_idx, pool_idx = linear_sum_assignment(cost[slot_bin][:, pool])
        np.add.at(P, (slot_bin[slot_idx], pool[pool_idx]), 1.0)
    P /= P.sum(axis=1, keepdims=True)
    return P


def generate_cytospace(
    single_cell,
    atlas_df,
    training_genes,
    distance_metric="pearson",
    cells_per_bin="dense",
    random_state=None,
    validation_genes=None,
):
    """Single CytoSPACE placement with known parameters.

    Parameters
    ----------
    single_cell : AnnData
        Single-cell data (full transcriptome) with log-normalized ``.X``.
    atlas_df : pd.DataFrame
        Atlas expression, (n_bins, G), bins as rows. Training genes drive the
        cost; validation genes (if scored) must also be columns here.
    training_genes : list
        Genes shared with the atlas that drive the assignment cost.
    distance_metric : str
        ``"pearson"`` or ``"spearman"`` correlation cost.
    cells_per_bin : str
        ``"dense"`` (k = n_cells // n_bins cells per bin, all cells placed) or
        ``"single"`` (one cell per bin).
    random_state : unused
        Ignored; the assignment is deterministic.
    validation_genes : list, optional
        If given, returns per-gene Pearson r of the projected validation genes.

    Returns
    -------
    dict
        ``{"P_to_cells": P (n_bins, n_cells), "train_r": dict,
        "val_r": dict | None}``.
    """
    tr = list(training_genes)
    atlas_bins = atlas_df[tr].to_numpy(dtype=np.float64)

    S_tr = single_cell[:, tr].X
    S_tr = S_tr.toarray() if hasattr(S_tr, "toarray") else np.asarray(S_tr)

    cost = _correlation_cost(atlas_bins, S_tr.astype(np.float64), distance_metric)
    P = _solve_placement(cost, atlas_bins.shape[0], cells_per_bin)

    def _project(genes):
        S = single_cell[:, list(genes)].X
        S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
        return P @ S.astype(np.float64)

    train_r = _per_gene_r(_project(tr), atlas_df, tr)
    val_r = None
    if validation_genes is not None:
        val_r = _per_gene_r(_project(validation_genes), atlas_df, validation_genes)

    return {"P_to_cells": P, "train_r": train_r, "val_r": val_r}


_DEFAULT_SEARCH_SPACE = {
    "distance_metric": ["pearson", "spearman"],
    "cells_per_bin": ["single", 2, 4, 8, "dense"],
}


def optimize_cytospace(
    single_cell,
    atlas_df,
    training_genes,
    validation_genes,
    output_dir="optimisation_cytospace_output",
    search_space=None,
    random_state=42,
):
    """Grid-search CytoSPACE knobs, scoring on validation genes.

    The knobs are discrete and few (correlation metric x cells-per-bin), so the
    search is an exhaustive loop over the grid; the objective is
    ``mean_r + median_r`` over the validation genes (matching
    ``optimize_tangram`` / ``optimize_novosparc``). Per-trial rows are appended
    to ``optimization_results.csv`` and the best placement is pickled to
    ``best_cytospace.pkl`` so a rerun returns instantly.

    Returns
    -------
    dict
        ``{"best_params", "best_score", "best_correlations",
        "best_cytospace_obj"}`` where ``best_cytospace_obj`` is ``P_to_cells``.
    """
    space = {**_DEFAULT_SEARCH_SPACE, **(search_space or {})}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "optimization_results.csv"
    best_obj_file = output_dir / "best_cytospace.pkl"

    # Resume: a finished run pickles the best placement; reload and return.
    if best_obj_file.exists():
        with open(best_obj_file, "rb") as f:
            info = pickle.load(f)
        return {
            "best_params": info["best_params"],
            "best_score": info["best_score"],
            "best_correlations": info["best_correlations"],
            "best_cytospace_obj": info["P_to_cells"],
        }

    # Deterministic grid → reuse combos already in the CSV; compute only the new ones.
    prev = pd.read_csv(results_file).to_dict("records") if results_file.exists() else []
    computed = {(r["distance_metric"], str(r["cells_per_bin"])) for r in prev}
    results: list[dict] = list(prev)

    for metric in space["distance_metric"]:
        for cpb in space["cells_per_bin"]:
            if (metric, str(cpb)) in computed:
                continue
            gen = generate_cytospace(
                single_cell=single_cell,
                atlas_df=atlas_df,
                training_genes=training_genes,
                distance_metric=metric,
                cells_per_bin=cpb,
                random_state=random_state,
                validation_genes=validation_genes,
            )
            val_vals = [v for v in gen["val_r"].values() if not np.isnan(v)]
            mean_r = float(np.mean(val_vals)) if val_vals else np.nan
            median_r = float(np.median(val_vals)) if val_vals else np.nan
            row = {
                "distance_metric": metric,
                "cells_per_bin": cpb,
                "mean_r": mean_r,
                "median_r": median_r,
                "score": (mean_r + median_r) if val_vals else -np.inf,
            }
            for g, r in {**gen["train_r"], **gen["val_r"]}.items():
                row[f"r_{g}"] = r
            results.append(row)
            pd.DataFrame(results).to_csv(results_file, index=False)
            logger.info("cytospace combo %s/%s: score=%.4f", metric, cpb, row["score"])

    if not results:
        raise RuntimeError("optimize_cytospace: empty grid — nothing evaluated.")

    # Best over the full grid (reused + new); regenerate its P object (a reused
    # combo has no in-memory object, so re-run its placement).
    best_row = max(results, key=lambda r: float(r["score"]))
    cpb = best_row["cells_per_bin"]
    cpb = int(str(cpb)) if str(cpb).isdigit() else cpb
    best_params = {"distance_metric": best_row["distance_metric"], "cells_per_bin": cpb}
    gen = generate_cytospace(
        single_cell=single_cell,
        atlas_df=atlas_df,
        training_genes=training_genes,
        validation_genes=validation_genes,
        random_state=random_state,
        **best_params,
    )
    best_score = float(best_row["score"])
    best_correlations = [v for v in gen["val_r"].values() if not np.isnan(v)]
    with open(best_obj_file, "wb") as f:
        pickle.dump(
            {
                "P_to_cells": gen["P_to_cells"],
                "best_params": best_params,
                "best_score": best_score,
                "best_correlations": best_correlations,
            },
            f,
        )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_correlations": best_correlations,
        "best_cytospace_obj": gen["P_to_cells"],
    }
