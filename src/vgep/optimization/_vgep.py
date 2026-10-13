"""vGEP cutoff parameter optimization via grid search."""

import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..plotter._recap import plot_recap as plot_recap_func
from ..processing import scale_data
from ..spatial_transcriptome import assess, generate_vgep, sc_data_handler, transform

logger = logging.getLogger(__name__)


def optimize_transcriptome(
    single_cells,
    atlas,
    geometry,
    validation,
    list_cutoffs,
    output_path,
    macrochaetae=None,
    plot_recap=True,
):
    """Optimize the vGEP cutoff parameter via grid search.

    Iterates over a list of cutoff quantiles, reconstructs a spatial
    transcriptome for each, and selects the cutoff that maximizes the
    composite score (mean + median) of validation gene correlations.
    Results are saved to CSV so the search can be resumed if interrupted.

    Parameters
    ----------
    single_cells : pd.DataFrame or AnnData
        Single-cell gene expression data (raw counts).
    atlas : pd.DataFrame
        Spatial atlas of gene expression patterns.
    geometry : pd.DataFrame
        Spatial coordinates with 'x' and 'y' columns.
    validation : pd.DataFrame
        Held-out validation gene expression dataset.
    list_cutoffs : list of float
        Cutoff quantile values to evaluate.
    output_path : str
        Directory for saving results CSV and best transcriptome.
    macrochaetae : pd.DataFrame, optional
        Morphological landmark positions to store in the best
        transcriptome.
    plot_recap : bool, optional
        If True, generates a PDF report for the best cutoff.

    Returns
    -------
    MuData
        Spatial transcriptome reconstructed with the optimal cutoff.
    """

    os.makedirs(output_path, exist_ok=True)

    # Results
    spatial_transcriptome = None
    recap_list = []

    trainingGenes = atlas.columns.tolist()
    validationGenes = validation.columns.tolist()

    sc_data = sc_data_handler(single_cells)

    # Normalized sc datasets
    sc_data_scaled = scale_data(sc_data)

    sc_data_scaled = sc_data_scaled[:, trainingGenes + validationGenes]

    if not os.path.exists(output_path + "/results.csv"):
        # Per-gene r columns follow the summary stats so CDFs can be drawn from this CSV.
        per_gene_cols = list(trainingGenes) + list(validationGenes)
        for cutoff in tqdm(list_cutoffs):

            if spatial_transcriptome is not None:
                spatial_transcriptome = transform(
                    sc_data_scaled,
                    atlas,
                    geometry,
                    cutoff,
                    verbose=False,
                    correlation_raw=spatial_transcriptome["meta"].layers[
                        "correlations_raw"
                    ],
                )

            else:
                spatial_transcriptome = transform(
                    sc_data_scaled, atlas, geometry, cutoff, verbose=False
                )
            spatial_transcriptome = assess(spatial_transcriptome, atlas, validation)

            results = spatial_transcriptome["st"].uns["results_recap"]
            train_corrs = spatial_transcriptome["st"].uns["correlations_genes_training"]
            val_corrs = spatial_transcriptome["st"].uns["correlations_genes_validation"]
            train_lookup = train_corrs.iloc[:, 0].to_dict()
            val_lookup = val_corrs.iloc[:, 0].to_dict()
            per_gene_row = [
                str(round(train_lookup.get(g, val_lookup.get(g, np.nan)), 4))
                for g in per_gene_cols
            ]
            recap_list.append(
                [",".join(trainingGenes), str(cutoff)]
                + [str(round(i, 3)) for i in results.loc["Validation"]]
                + per_gene_row
            )

        recapDf = pd.DataFrame(
            recap_list,
            columns=[
                "genes",
                "cutoff",
                "n genes",
                "mean",
                "min",
                "q25%",
                "median",
                "q75%",
                "max",
                *[f"r_{g}" for g in per_gene_cols],
            ],
        )
        recapDf.to_csv(output_path + "/results.csv", sep=";")
    else:

        recapDf = pd.read_csv(
            os.path.join(output_path, "results.csv"), index_col=0, sep=";"
        )
    scores = recapDf[["mean", "median"]].astype(float).round(2).sum(axis=1).values

    # We do [::-1] to get the most stringent cutoff first and select as few cells as possible
    best_index = len(scores) - np.nanargmax(scores[::-1]) - 1

    logger.info(
        "Best cutoff value: %s  mean: %s  median: %s",
        recapDf.iloc[best_index, 1],
        recapDf.iloc[best_index, 3],
        recapDf.iloc[best_index, 6],
    )

    transcriptome = generate_vgep(
        single_cells,
        atlas,
        geometry,
        float(recapDf.iloc[best_index, 1]),
        validation,
        morphological_marks=macrochaetae,
    )

    # Plotting a recap for the best spatial transcriptome
    if plot_recap:
        plot_recap_func(
            transcriptome,
            atlas,
            validation,
            path=output_path + "/best_transcriptome_report.pdf",
        )

    transcriptome.write(output_path + "/best_transcriptome.h5mu")

    return transcriptome
