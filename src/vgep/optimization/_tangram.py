"""Tangram parameter optimization via Optuna search over the cells-mode mapping.

Tangram (Biancalani et al., Nature Methods 2021) learns a cell->voxel mapping
matrix M by gradient descent, maximizing the per-gene cosine similarity between
the mapped single-cell expression and the spatial reference on the shared
(training) genes. Unlike novosparc it uses no cell embedding, so the spatial fit
is latent-invariant.

``generate_tangram`` runs one mapping with known hyperparameters.
``optimize_tangram`` searches hyperparameters with Optuna's TPE sampler, scoring
each trial by the validation-gene Pearson r (mean + median, matching the
novosparc selection objective). Results are streamed to a CSV and the best
mapping is pickled so reruns resume instantly.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from vgep.evaluation.correlations import per_gene_pearson_r

logger = logging.getLogger(__name__)


def _resolve_device(device):
    """Return 'cuda' when available unless a device is pinned explicitly."""
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _projected_expression(mapping_matrix, single_cell, genes):
    """Project ``genes`` of ``single_cell`` through the map: (n_spots, len(genes)).

    ``mapping_matrix`` is (n_cells, n_spots) row-stochastic over spots, so
    ``M.T @ S`` superposes cell expression at each spot.
    """
    S = single_cell[:, list(genes)].X
    S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    return mapping_matrix.T @ S.astype(np.float64)


def _per_gene_r(pred, atlas_df, genes):
    """Per-gene Pearson r between projected expression and the atlas over spots.

    ``pred`` is (n_spots, len(genes)) aligned to ``genes``; ``atlas_df`` is the
    (n_spots, G) atlas restricted to the same spots. NaN bins are dropped.
    """
    ref = np.column_stack([
        atlas_df[g].to_numpy(dtype=np.float64) if g in atlas_df.columns else np.full(len(pred), np.nan)
        for g in genes
    ])
    return dict(zip(genes, per_gene_pearson_r(pred, ref, gene_axis=1).astype(float)))


def generate_tangram(
    single_cell,
    atlas_df,
    training_genes,
    mode="cells",
    density_prior="uniform",
    num_epochs=1000,
    learning_rate=0.1,
    lambda_g2=0.0,
    lambda_r=0.0,
    device=None,
    random_state=None,
    validation_genes=None,
):
    """Single tangram reconstruction with known parameters.

    Parameters
    ----------
    single_cell : AnnData
        Single-cell data (full transcriptome) with log-normalized ``.X``.
    atlas_df : pd.DataFrame
        Atlas expression, (n_spots, G), spots as rows. Training genes drive the
        fit; validation genes (if scored) must also be columns here.
    training_genes : list
        Genes shared with the atlas that drive the mapping.
    mode : str
        Tangram mode: ``"cells"`` (default), ``"clusters"`` or ``"constrained"``.
    density_prior : str
        ``"uniform"`` or ``"rna_count_based"``.
    num_epochs, learning_rate, lambda_g2, lambda_r : optimizer knobs.
    device : str or None
        ``"cpu"``/``"cuda"``; ``None`` auto-selects cuda when available.
    validation_genes : list, optional
        If given, returns per-gene Pearson r of the projected validation genes
        against the atlas.

    Returns
    -------
    dict
        ``{"mapping": M (n_cells, n_spots), "adata_map": AnnData,
        "train_r": dict, "val_r": dict | None}``.
    """
    import anndata as ad
    import tangram as tg

    device = _resolve_device(device)

    adata_sc = single_cell.copy()
    G = atlas_df[list(training_genes)].to_numpy(dtype="float32")
    adata_sp = ad.AnnData(X=G)
    adata_sp.var_names = list(training_genes)
    adata_sp.obs_names = [str(i) for i in atlas_df.index]

    # gene_to_lowercase=False preserves case-sensitive fly names (H = Hairless
    # vs hh = hedgehog). pp_adatas mutates its inputs, hence the adata_sc copy.
    tg.pp_adatas(
        adata_sc, adata_sp, genes=list(training_genes), gene_to_lowercase=False
    )

    adata_map = tg.map_cells_to_space(
        adata_sc,
        adata_sp,
        mode=mode,
        density_prior=density_prior,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        lambda_g2=lambda_g2,
        lambda_r=lambda_r,
        device=device,
        random_state=random_state,
        verbose=False,
    )

    M = np.asarray(adata_map.X, dtype=np.float64)  # (n_cells, n_spots)

    train_r = _per_gene_r(
        _projected_expression(M, single_cell, training_genes), atlas_df, training_genes
    )
    val_r = None
    if validation_genes is not None:
        val_r = _per_gene_r(
            _projected_expression(M, single_cell, validation_genes),
            atlas_df,
            validation_genes,
        )

    return {"mapping": M, "adata_map": adata_map, "train_r": train_r, "val_r": val_r}


_DEFAULT_SEARCH_SPACE = {
    "density_prior": ["uniform", "rna_count_based"],
    "num_epochs": [500, 1000, 2000, 3000],
    "learning_rate": (1e-3, 0.3),  # log-uniform range
    "lambda_g2": [0.0, 1.0],
    "lambda_r": [0.0, 1e-3, 1e-2],
}


def optimize_tangram(
    single_cell,
    atlas_df,
    training_genes,
    validation_genes,
    output_dir="optimisation_tangram_output",
    n_trials=50,
    mode="cells",
    device=None,
    search_space=None,
    random_state=42,
    initial_params=None,
):
    """Optimize tangram hyperparameters with Optuna, scoring on validation genes.

    Each trial maps the single cells to space and scores the projected
    validation genes against the atlas; the objective is ``mean_r + median_r``
    (matching ``optimize_novosparc``). Per-trial rows are appended to
    ``optimization_results.csv`` and the best mapping is pickled to
    ``best_tangram.pkl`` so a rerun returns instantly. ``initial_params`` (a list of
    parameter dicts) are evaluated first, so a known optimum is always part of the search.

    Returns
    -------
    dict
        ``{"best_params", "best_score", "best_correlations",
        "best_tangram_obj"}`` where ``best_tangram_obj`` is the AnnData mapping.
    """
    space = {**_DEFAULT_SEARCH_SPACE, **(search_space or {})}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "optimization_results.csv"
    best_obj_file = output_dir / "best_tangram.pkl"

    # Resume: a finished run pickles the best mapping; reload it without importing optuna/torch.
    if best_obj_file.exists():
        with open(best_obj_file, "rb") as f:
            info = pickle.load(f)
        return {
            "best_params": info["best_params"],
            "best_score": info["best_score"],
            "best_correlations": info["best_correlations"],
            "best_tangram_obj": info["adata_map"],
        }

    import optuna

    results: list[dict] = []
    best = {"score": -np.inf, "params": None, "adata_map": None, "correlations": None}

    def objective(trial):
        params = {
            "mode": mode,
            "density_prior": trial.suggest_categorical(
                "density_prior", space["density_prior"]
            ),
            "num_epochs": trial.suggest_categorical("num_epochs", space["num_epochs"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True
            ),
            "lambda_g2": trial.suggest_categorical("lambda_g2", space["lambda_g2"]),
            "lambda_r": trial.suggest_categorical("lambda_r", space["lambda_r"]),
        }
        gen = generate_tangram(
            single_cell=single_cell,
            atlas_df=atlas_df,
            training_genes=training_genes,
            device=device,
            random_state=random_state,
            validation_genes=validation_genes,
            **params,
        )
        val_vals = [v for v in gen["val_r"].values() if not np.isnan(v)]
        mean_r = float(np.mean(val_vals)) if val_vals else np.nan
        median_r = float(np.median(val_vals)) if val_vals else np.nan
        score = mean_r + median_r if val_vals else -np.inf

        row = {**params, "mean_r": mean_r, "median_r": median_r, "score": score}
        for g, r in {**gen["train_r"], **gen["val_r"]}.items():
            row[f"r_{g}"] = r
        results.append(row)
        pd.DataFrame(results).to_csv(results_file, index=False)

        if score > best["score"]:
            best.update(
                score=score,
                params=params,
                adata_map=gen["adata_map"],
                correlations=val_vals,
            )
            logger.info("tangram new best: score=%.4f params=%s", score, params)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    for params in initial_params or []:
        study.enqueue_trial(params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if best["adata_map"] is None:
        raise RuntimeError(
            "optimize_tangram: no valid score across the search. Check that "
            f"atlas_df has finite non-constant values for {validation_genes!r}."
        )

    with open(best_obj_file, "wb") as f:
        pickle.dump(
            {
                "adata_map": best["adata_map"],
                "best_params": best["params"],
                "best_score": best["score"],
                "best_correlations": best["correlations"],
            },
            f,
        )

    return {
        "best_params": best["params"],
        "best_score": best["score"],
        "best_correlations": best["correlations"],
        "best_tangram_obj": best["adata_map"],
    }
