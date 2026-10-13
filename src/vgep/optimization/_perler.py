"""PERLER parameter optimization via grid search over k_metagenes."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from vgep.evaluation.correlations import per_gene_pearson_r

logger = logging.getLogger(__name__)


def generate_perler(
    single_cell_df,
    atlas_df,
    training_genes,
    k_metagenes,
    geometry,
    validation_genes=None,
):
    """Single PERLER reconstruction with known parameters.

    Parameters
    ----------
    single_cell_df : pd.DataFrame
        Preprocessed single cell expression (NOT AnnData).
    atlas_df : pd.DataFrame
        Atlas gene expression data.
    training_genes : list
        List of training gene names.
    k_metagenes : int
        Number of metagenes.
    geometry : pd.DataFrame
        Spatial coordinates.
    validation_genes : list, optional
        If provided, compute per-gene Pearson correlations.
        If ``None``, skip validation.

    Returns
    -------
    dict
        ``{"expression": DataFrame (raw result_with_location),
        "plr": PERLER, "correlations": list | None}``
    """
    from perler import PERLER

    plr = PERLER(
        data=single_cell_df,
        reference=atlas_df[training_genes],
        print_iter=False,
        n_metagenes=k_metagenes,
        log_reference=False,
    )
    plr.em_algorithm(optimize_pi=False)
    plr.calc_dist()
    plr.loocv()
    plr.grid_search()
    plr.spatial_reconstruction(location=geometry)

    correlations = None
    if validation_genes is not None:
        correlations = []
        for gene in validation_genes:
            reconstructed = plr.result_with_location.loc[
                ~plr.result_with_location.index.isin(geometry.index), gene
            ]
            reconstructed = pd.DataFrame(
                reconstructed.values, index=reconstructed.index, columns=[gene]
            ).values

            original = pd.DataFrame(
                atlas_df[gene].values, index=atlas_df.index, columns=[gene]
            ).values

            correlations.append(float(per_gene_pearson_r(reconstructed.T, original.T)[0]))

    return {
        "expression": plr.result_with_location,
        "plr": plr,
        "correlations": correlations,
    }


def optimize_perler(
    single_cell,
    atlas_df,
    training_genes,
    validation_genes,
    geometry,
    output_dir="optimisation_perler_output",
    k_range=None,
):
    """Optimize PERLER parameters and save/load results.

    Searches over the number of metagenes (``k``), saves results
    continuously, and persists the best PERLER object.

    Parameters
    ----------
    single_cell : AnnData
        Single cell data.
    atlas_df : pd.DataFrame
        Atlas gene expression data.
    training_genes : list
        List of training gene names.
    validation_genes : list
        List of validation gene names.
    geometry : pd.DataFrame
        Spatial coordinates.
    output_dir : str or Path, optional
        Output directory for results and best object.
    k_range : list of int, optional
        Values of ``k_metagenes`` to search. Defaults to ``range(5, 23)``.

    Returns
    -------
    dict
        Dictionary with keys ``best_params``, ``best_score``,
        ``k_values``, ``best_correlations``, ``best_perler_obj``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "optimization_results.csv"
    best_obj_file = output_dir / "best_perler.pkl"

    if k_range is None:
        k_metagenes = list(range(5, 23))
    else:
        k_metagenes = list(k_range)

    # If best PERLER object already exists, load and return it.
    if best_obj_file.exists():
        with open(best_obj_file, "rb") as f:
            best_obj_info = pickle.load(f)
        if results_file.exists():
            results_df = pd.read_csv(results_file)
            best_row = results_df.loc[results_df["score"].idxmax()]
            best_params = {"k_metagenes": int(best_row["k_metagenes"])}
            best_score = best_row["score"]
            best_correlations = [
                best_row[f"corr_{gene}"] for gene in validation_genes
            ]
        else:
            best_params, best_score, best_correlations = None, None, None
        return {
            "best_params": best_params,
            "best_score": best_score,
            "k_values": k_metagenes,
            "best_correlations": best_correlations,
            "best_perler_obj": best_obj_info["plr"],
        }

    if results_file.exists():
        results_df = pd.read_csv(results_file)
        results = results_df.to_dict(orient="records")
        computed_k = {row["k_metagenes"] for row in results}
        logger.debug("Computed k: %s", computed_k)

        if results:
            best_row = results_df.loc[results_df["score"].idxmax()]
            best_score = best_row["score"]
            best_params = {"k_metagenes": int(best_row["k_metagenes"])}
            best_correlations = [
                best_row[f"corr_{gene}"] for gene in validation_genes
            ]
        else:
            best_score = float("-inf")
            best_params = None
            best_correlations = None
    else:
        results = []
        computed_k = set()
        best_score = float("-inf")
        best_params = None
        best_correlations = None

    missing_ks = sorted(set(k_metagenes) - computed_k)
    best_plr = None

    for k in tqdm(missing_ks, desc="Optimizing Perler parameters"):
        gen_result = generate_perler(
            single_cell_df=single_cell,
            atlas_df=atlas_df,
            training_genes=training_genes,
            k_metagenes=k,
            geometry=geometry,
            validation_genes=validation_genes,
        )
        plr = gen_result["plr"]
        correlations = gen_result["correlations"]

        mean_r = float(np.nanmean(correlations))
        median_r = float(np.nanmedian(correlations))
        score = mean_r + median_r

        result = {
            "k_metagenes": k,
            "mean_r": mean_r,
            "median_r": median_r,
            "score": score,
        }
        for gene, corr in zip(validation_genes, correlations):
            result[f"corr_{gene}"] = corr

        results.append(result)
        computed_k.add(k)

        pd.DataFrame(results).to_csv(results_file, index=False)

        if score > best_score:
            best_score = score
            best_params = {"k_metagenes": k}
            best_plr = plr
            best_correlations = correlations
            logger.info("New best score found for k=%d: %s", k, score)

    # The best k came from an earlier (resumed) run: rebuild its PERLER object.
    if best_plr is None:
        logger.info("Rebuilding the PERLER object for the best k=%d", best_params["k_metagenes"])
        best_plr = generate_perler(
            single_cell_df=single_cell,
            atlas_df=atlas_df,
            training_genes=training_genes,
            k_metagenes=best_params["k_metagenes"],
            geometry=geometry,
        )["plr"]

    best_obj_info = {
        "plr": best_plr,
        "best_params": best_params,
        "best_score": best_score,
        "best_correlations": best_correlations,
    }
    with open(best_obj_file, "wb") as f:
        pickle.dump(best_obj_info, f)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "k_values": k_metagenes,
        "best_correlations": best_correlations,
        "best_perler_obj": best_plr,
    }
