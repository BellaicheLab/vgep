"""Novosparc parameter optimization via grid search."""

import copy
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import to_matrix_form
from vgep.evaluation.correlations import per_gene_pearson_r

logger = logging.getLogger(__name__)

# Silence novosparc's hard-coded `print('Trying with epsilon...')`, which ignores `verbose`.
# novosparc is imported lazily, so vgep stays importable without it.
try:
    import novosparc.common._tissue as _novosparc_tissue
    _novosparc_tissue.print = lambda *a, **kw: None
except ImportError:
    pass


def generate_novosparc(
    single_cell,
    geometry,
    atlas_df,
    training_genes,
    alpha,
    epsilon,
    num_neighbors_s,
    num_neighbors_t,
    dge_rep=None,
    validation_genes=None,
):
    """Single novosparc reconstruction with known parameters.

    Parameters
    ----------
    single_cell : AnnData
        Preprocessed single cell data.
    geometry : pd.DataFrame
        Spatial coordinates (must contain ``x``, ``y``, ``zcoord`` columns).
    atlas_df : pd.DataFrame
        Atlas gene expression data.
    training_genes : list
        List of training gene names.
    alpha : float
        Linear cost weight.
    epsilon : float
        Entropy regularization parameter.
    num_neighbors_s : int
        Number of spatial neighbors.
    num_neighbors_t : int
        Number of expression neighbors.
    dge_rep : array-like, optional
        Representation for smooth costs (PCA or variable genes).
        Defaults to ``single_cell.to_df()``.
    validation_genes : list, optional
        If provided, compute per-gene Pearson correlations against
        the atlas. If ``None``, skip validation.

    Returns
    -------
    dict
        ``{"expression": DataFrame, "tissue": tissue,
        "correlations": list | None}``
    """
    import novosparc

    if dge_rep is None:
        dge_rep = single_cell.to_df()

    atlas_matrix = atlas_df[training_genes].values
    markers = np.array(
        [single_cell.to_df().columns.get_loc(gene) for gene in training_genes]
    )

    tissue = novosparc.cm.Tissue(
        dataset=single_cell, locations=geometry[["x", "y", "zcoord"]]
    )
    tissue.setup_linear_cost(markers, atlas_matrix)
    tissue.setup_smooth_costs(
        dge_rep=dge_rep,
        num_neighbors_s=num_neighbors_s,
        num_neighbors_t=num_neighbors_t,
    )
    tissue.reconstruct(alpha_linear=alpha, epsilon=epsilon, verbose=False)

    expression = pd.DataFrame(
        tissue.sdge.T, index=geometry.index, columns=single_cell.var_names
    )

    correlations = None
    if validation_genes is not None:
        correlations = [
            float(per_gene_pearson_r(
                to_matrix_form(expression[gene], geometry=geometry)[None],
                to_matrix_form(atlas_df[gene], geometry=geometry)[None],
            )[0])
            for gene in validation_genes
        ]

    return {"expression": expression, "tissue": tissue, "correlations": correlations}


def optimize_novosparc(
    single_cell,
    geometry,
    atlas_df,
    training_genes,
    validation_genes,
    output_dir="optimisation_novosparc_output",
    alpha_values=None,
    epsilon_values=None,
    t_values=None,
    s_values=None,
    rep_options=None,
):
    """Optimize novosparc parameters using grid search.

    Saves/loads results to allow resuming and updates the CSV file
    continuously during the optimization run.

    Parameters
    ----------
    single_cell : AnnData
        Single cell data.
    geometry : pd.DataFrame
        Spatial coordinates (must contain ``x``, ``y``, ``zcoord`` columns).
    atlas_df : pd.DataFrame
        Atlas gene expression data.
    training_genes : list
        List of training gene names.
    validation_genes : list
        List of validation gene names.
    output_dir : str or Path, optional
        Output directory for results and best object.
    alpha_values : array-like, optional
        Alpha values to search. Defaults to ``np.linspace(0, 1.0, 11)``.
    epsilon_values : array-like, optional
        Epsilon values to search. Defaults to ``np.logspace(-2, -1, 3)``.
    t_values : list of int, optional
        Number of neighbors (t) to search. Defaults to ``[5, 8, 12]``.
    s_values : list of int, optional
        Number of neighbors (s) to search. Defaults to ``[2, 3, 5, 10, 15, 20, 25]``.
    rep_options : dict[str, array-like], optional
        Cell-cell-cost representations to grid-search as ``dge_rep``. Keys are
        labels stored in the ``rep`` column of the results CSV. Defaults to
        ``{"variable": single_cell.to_df(), "pca": single_cell.obsm["X_pca"]}``.

    Returns
    -------
    dict
        Dictionary with keys ``best_params``, ``best_score``,
        ``alpha_values``, ``epsilon_values``, ``t_values``, ``s_values``,
        ``best_correlations``, ``best_novosparc_obj``.
    """
    import novosparc

    # Set random seed for reproducibility
    np.random.seed(42)

    # Default grid parameters
    if alpha_values is None:
        alpha_values = np.linspace(0, 1.0, 11)
    else:
        alpha_values = np.asarray(alpha_values)
    if epsilon_values is None:
        epsilon_values = np.logspace(-2, -1, 3)
    else:
        epsilon_values = np.asarray(epsilon_values)
    if t_values is None:
        t_values = [5, 8, 12]
    if s_values is None:
        s_values = [2, 3, 5, 10, 15, 20, 25]

    # Ensure the output directory exists.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "optimization_results.csv"
    best_obj_file = output_dir / "best_novosparc.pkl"

    # If best novosparc object already exists, load and return it.
    if best_obj_file.exists():
        with open(best_obj_file, "rb") as f:
            best_obj_info = pickle.load(f)
        if results_file.exists():
            results_df = pd.read_csv(results_file)
            best_row = results_df.loc[results_df["score"].idxmax()]
            best_params = {
                "alpha": best_row["alpha"],
                "epsilon": best_row["epsilon"],
                "t": best_row["t"],
                "s": best_row["s"],
                "rep": best_row.get("rep", "variable"),
            }
            best_score = best_row["score"]
            best_correlations = [best_row[f"r_{gene}"] for gene in validation_genes]
        else:
            best_params, best_score, best_correlations = None, None, None
        return {
            "best_params": best_params,
            "best_score": best_score,
            "alpha_values": alpha_values.tolist(),
            "epsilon_values": epsilon_values.tolist(),
            "t_values": t_values,
            "s_values": s_values,
            "best_correlations": best_correlations,
            "best_novosparc_obj": best_obj_info["tissue"],
        }

    # Load already computed results if available
    if results_file.exists():
        results_df = pd.read_csv(results_file)
        results = results_df.to_dict(orient="records")
        computed_combinations = {
            (
                row["rep"],
                int(row["s"]),
                int(row["t"]),
                round(float(row["alpha"]), 8),
                round(float(row["epsilon"]), 8),
            )
            for row in results
        }
        if results:
            best_row = results_df.loc[results_df["score"].idxmax()]
            best_score = best_row["score"]
            best_params = {
                "alpha": best_row["alpha"],
                "epsilon": best_row["epsilon"],
                "t": best_row["t"],
                "s": best_row["s"],
                "rep": best_row.get("rep", "variable"),
            }
            best_correlations = [best_row[f"r_{gene}"] for gene in validation_genes]
        else:
            best_score = float("-inf")
            best_params = None
            best_correlations = None
            results = []
            computed_combinations = set()
    else:
        results = []
        computed_combinations = set()
        best_score = float("-inf")
        best_params = None
        best_correlations = None

    # Define representations (caller-overridable)
    if rep_options is None:
        rep_options = {
            "variable": single_cell.to_df(),
            "pca": single_cell.obsm["X_pca"],
        }

    # Pre-calculate the atlas matrix and markers.
    atlas_matrix = atlas_df[training_genes].values
    markers = np.array(
        [single_cell.to_df().columns.get_loc(gene) for gene in training_genes]
    )

    # Create a template tissue.
    base_tissue_template = novosparc.cm.Tissue(
        dataset=single_cell, locations=geometry[["x", "y", "zcoord"]]
    )
    base_tissue_template.setup_linear_cost(markers, atlas_matrix)

    # Determine total number of evaluations for the progress bar.
    total_iterations = (
        len(rep_options)
        * len(s_values)
        * len(t_values)
        * len(alpha_values)
        * len(epsilon_values)
    )
    pbar = tqdm(total=total_iterations, desc="Optimizing novosparc parameters")

    best_tissue = None

    # Main grid search loop.
    for rep_name, dge_rep in rep_options.items():
        for s in s_values:
            for t in t_values:
                # Check if entire (rep, s, t) block already computed.
                keys_for_combo = {
                    (rep_name, int(s), int(t), round(alpha, 8), round(epsilon, 8))
                    for alpha in alpha_values
                    for epsilon in epsilon_values
                }
                if keys_for_combo.issubset(computed_combinations):
                    pbar.update(len(keys_for_combo))
                    continue

                # Set up smooth_costs once per (rep, s, t) combination.
                base_tissue = copy.deepcopy(base_tissue_template)
                base_tissue.setup_smooth_costs(
                    dge_rep=dge_rep, num_neighbors_s=s, num_neighbors_t=t
                )

                for alpha in alpha_values:
                    for epsilon in epsilon_values:
                        key = (
                            rep_name,
                            int(s),
                            int(t),
                            round(alpha, 8),
                            round(epsilon, 8),
                        )
                        if key in computed_combinations:
                            pbar.update(1)
                            if best_correlations is not None:
                                logger.debug(
                                    "Current best median r so far: %s",
                                    np.median(best_correlations),
                                )
                            continue
                        else:
                            logger.debug("Exploring key: %s", key)

                        tissue = copy.deepcopy(base_tissue)
                        tissue.reconstruct(
                            alpha_linear=alpha, epsilon=epsilon, verbose=False
                        )

                        # Per-gene Pearson r vs atlas for training and validation
                        # genes; the optimization score uses validation genes only.
                        reconstructed_expression = pd.DataFrame(
                            tissue.sdge.T,
                            index=geometry.index,
                            columns=single_cell.var_names,
                        )

                        def _gene_r(gene):
                            return float(per_gene_pearson_r(
                                to_matrix_form(reconstructed_expression[gene], geometry=geometry)[None],
                                to_matrix_form(atlas_df[gene], geometry=geometry)[None],
                            )[0])

                        train_r = {g: _gene_r(g) for g in training_genes}
                        val_r = {g: _gene_r(g) for g in validation_genes}

                        # Score on validation genes, dropping NaNs (genes with no valid pixels).
                        correlations = [v for v in val_r.values() if not np.isnan(v)]
                        mean_r = np.mean(correlations)
                        median_r = np.median(correlations)
                        score = mean_r + median_r

                        result = {
                            "alpha": alpha,
                            "epsilon": epsilon,
                            "t": t,
                            "s": s,
                            "rep": rep_name,
                            "mean_r": mean_r,
                            "median_r": median_r,
                            "score": score,
                        }
                        for gene, corr in {**train_r, **val_r}.items():
                            result[f"r_{gene}"] = corr

                        results.append(result)
                        computed_combinations.add(key)

                        # Update CSV file immediately.
                        pd.DataFrame(results).to_csv(results_file, index=False)

                        if score > best_score:
                            best_score = score
                            best_params = {
                                "alpha": alpha,
                                "epsilon": epsilon,
                                "t": t,
                                "s": s,
                                "rep": rep_name,
                            }
                            best_correlations = correlations
                            best_tissue = tissue
                            logger.info("New best median r found: %s", median_r)

                        if best_correlations is not None:
                            logger.debug(
                                "Current best median r so far: %s",
                                np.median(best_correlations),
                            )

                        pbar.update(1)

    pbar.close()

    # If the grid was fully cached (CSV existed, all combos skipped),
    # best_tissue is still None. Rerun with best params to get the object.
    if best_tissue is None and best_params is not None:
        logger.info(
            "All combinations already computed. Rerunning best params to "
            "reconstruct tissue: %s",
            best_params,
        )
        result = generate_novosparc(
            single_cell=single_cell,
            geometry=geometry,
            atlas_df=atlas_df,
            training_genes=training_genes,
            alpha=best_params["alpha"],
            epsilon=best_params["epsilon"],
            num_neighbors_s=best_params["s"],
            num_neighbors_t=best_params["t"],
            dge_rep=rep_options.get(best_params.get("rep", "variable")),
            validation_genes=validation_genes,
        )
        best_tissue = result["tissue"]

    # If every grid combo produced a non-finite score, best_params/best_tissue
    # never updated. That almost always means a degenerate input (e.g. atlas
    # all-NaN/all-zero so every per-gene correlation is NaN).
    if best_tissue is None or best_params is None:
        raise RuntimeError(
            "optimize_novosparc: no valid scores across the entire grid "
            f"({len(results)} combos). Check that atlas_df has finite "
            "non-constant values for the validation genes "
            f"{validation_genes!r}."
        )

    # Save the best tissue object.
    best_obj_info = {
        "tissue": best_tissue,
        "best_params": best_params,
        "best_score": best_score,
        "best_correlations": best_correlations,
    }
    with open(best_obj_file, "wb") as f:
        pickle.dump(best_obj_info, f)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "alpha_values": alpha_values.tolist(),
        "epsilon_values": epsilon_values.tolist(),
        "t_values": t_values,
        "s_values": s_values,
        "best_correlations": best_correlations,
        "best_novosparc_obj": best_tissue,
    }
