"""Spatial transcriptome generation, transformation, and validation."""

import logging
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from .processing import scale_data
from .utils import change_modality, remove_index_prefix, has_prefix, add_index_prefix
from vgep.evaluation.correlations import per_gene_pearson_r

logger = logging.getLogger(__name__)


def generate_vgep(
    sc_data: Union[pd.DataFrame, AnnData],
    atlas: pd.DataFrame,
    geometry: pd.DataFrame,
    cutoff: float,
    validation: pd.DataFrame = None,
    morphological_marks=None,
    specific_genes: list = None,
) -> MuData:
    """Generate a spatial transcriptome from single-cell data and a spatial atlas.

    Normalizes single-cell data, computes correlation-based mapping between
    cells and spatial locations, projects gene expression onto tissue geometry,
    and optionally validates against known spatial patterns.

    Parameters
    ----------
    sc_data : AnnData or pd.DataFrame
        Single-cell gene expression data (raw counts).
    atlas : pd.DataFrame
        Spatial atlas of gene expression patterns. Rows are spatial positions,
        columns are genes.
    geometry : pd.DataFrame
        Spatial coordinates of atlas positions with 'x' and 'y' columns.
    cutoff : float
        Quantile threshold for filtering poorly correlated cells at each
        spatial position.
    validation : pd.DataFrame, optional
        Validation gene expression dataset for assessing reconstruction
        quality.
    morphological_marks : pd.DataFrame, optional
        Morphological landmark positions to store in the spatial
        transcriptome for visualization.
    specific_genes : list, optional
        Restrict mapping to this subset of genes.

    Returns
    -------
    MuData
        Multi-modal dataset with modalities:
        - ``"st"``: spatial transcriptome (projected expression)
        - ``"sc"``: single-cell reference
        - ``"meta"``: correlation and mapping matrices
    """

    sc_data = sc_data_handler(sc_data)

    # Normalized sc datasets
    sc_data = scale_data(sc_data)

    if specific_genes is not None:
        sc_data = sc_data[:, specific_genes]

    spatial_transcriptome = transform(sc_data, atlas, geometry, cutoff, verbose=True)

    # Adding the morphological marks
    if morphological_marks is not None:
        morphological_marks.index = morphological_marks.index.astype(str)
        morphological_marks.columns = morphological_marks.columns.astype(str)
        spatial_transcriptome["st"].uns["morphological_marks"] = morphological_marks

    if validation is not None:
        spatial_transcriptome = assess(spatial_transcriptome, atlas, validation)

    return spatial_transcriptome


def transform(
    sc_data: AnnData,
    atlas: pd.DataFrame,
    geometry: pd.DataFrame,
    cutoff: float,
    verbose: bool = True,
    correlation_raw=None,
) -> MuData:
    """Project single-cell expression onto spatial positions.

    Computes Pearson correlations between atlas and single-cell profiles,
    filters low-quality mappings using a quantile cutoff, and projects
    gene expression onto spatial coordinates.

    Parameters
    ----------
    sc_data : AnnData
        Single-cell data with a ``"scaled_data"`` layer.
    atlas : pd.DataFrame
        Spatial atlas of gene expression patterns.
    geometry : pd.DataFrame
        Spatial coordinates with 'x' and 'y' columns.
    cutoff : float
        Quantile threshold for filtering poorly correlated cells.
    verbose : bool, optional
        Log progress messages.
    correlation_raw : np.ndarray, optional
        Pre-computed correlation matrix. If None, computed from
        atlas and single-cell data.

    Returns
    -------
    MuData
        Multi-modal dataset with ``"st"``, ``"sc"``, and ``"meta"``
        modalities.
    """
    # Convert the anndata to a dataframe and remove the prefix if it exists
    single_cell = sc_data.to_df(layer="scaled_data")
    if has_prefix(single_cell):
        single_cell = change_modality(single_cell, "")

    # Convert single_cell column names to strings
    single_cell.columns = single_cell.columns.astype(str)

    # Convert atlas column names to strings
    atlas.columns = atlas.columns.astype(str)

    training_genes = np.intersect1d(
        atlas.columns.astype(str), single_cell.columns.astype(str)
    )

    if correlation_raw is None:
        correlation_raw = _compute_correlation_matrix(
            atlas[training_genes].values, np.log(single_cell[training_genes] + 1).values
        )

    # Creating the DataFrame containing these values
    df_correlation_raw = pd.DataFrame(
        correlation_raw, index=atlas.index, columns=single_cell.index
    )

    if verbose:
        logger.debug("Correlation between cells and Atlas established")

    # Temporary DataFrame
    df_correlation_mapped = df_correlation_raw.copy()

    # Removing values under quantile value
    quantiles_log = df_correlation_mapped.quantile(cutoff, axis=1)

    mask = df_correlation_mapped.lt(quantiles_log, axis=0)
    df_correlation_mapped[mask] = 0

    # Removing negatives and nan values
    df_correlation_mapped[df_correlation_mapped < 0] = 0
    df_correlation_mapped[df_correlation_mapped.isna()] = 0

    # Boolean matrix containing the cells that are mapped
    df_correlation_mapped_bool = df_correlation_mapped.copy()
    df_correlation_mapped_bool[df_correlation_mapped_bool > 0] = 1

    # Removing cells that are poorly correlated for each location
    if verbose:
        logger.debug("Removing cells that are poorly correlated with every location")

    good_cells_log = np.where(np.sum(df_correlation_mapped, axis=0) != 0)[0]

    if verbose:
        logger.debug(
            "Cells retained: %d, Cells discarded: %d",
            len(good_cells_log),
            df_correlation_raw.shape[1] - len(good_cells_log),
        )

    normalized_correlation_matrix = _normalize_correlation_matrix(df_correlation_mapped.values)
    normalize_correlation_df = pd.DataFrame(normalized_correlation_matrix,
                                            index=df_correlation_mapped.index,
                                            columns=df_correlation_mapped.columns)

    projection_log_scaled = _compute_projection_matrix(
        normalized_correlation_matrix,
        single_cell.loc[df_correlation_mapped.columns].values,
    )

    df_projection_log_scaled = pd.DataFrame(
        projection_log_scaled,
        index=df_correlation_mapped.index,
        columns=single_cell.columns,
    )

    if verbose:
        logger.debug("Projection of expression of location established")
    # Setting class values
    mapping = df_projection_log_scaled.astype(np.float32)

    # Creating the anndata object containing everything
    uns = {"training_genes": training_genes, "cutoff": cutoff}
    obs = {"x": geometry["x"].values.astype(str), "y": geometry["y"].values.astype(str)}

    # Spatial transcriptome we add the prefix to comply with mudata
    mapping.columns = add_index_prefix(mapping.columns, "st")
    adata_spatial_transcriptome = AnnData(X=mapping.values, uns=uns, obs=obs)
    adata_spatial_transcriptome.obs_names = mapping.index.astype(str)
    adata_spatial_transcriptome.var_names = mapping.columns

    # Metadata, we add a prefix as well to comply with mudata
    df_correlation_mapped.columns = add_index_prefix(
        df_correlation_mapped.columns, "meta"
    )
    adata_meta = AnnData(X=df_correlation_mapped.values.astype(np.float32), obs=obs)

    # Adding layers
    adata_meta.layers["correlations_raw"] = correlation_raw.astype(np.float32)
    adata_meta.layers["mapped"] = df_correlation_mapped_bool.values.astype(int)
    adata_meta.layers["normalized_correlation"] = normalize_correlation_df.values.astype(np.float32)
    adata_meta.obs_names = mapping.index.astype(str)
    adata_meta.var_names = df_correlation_raw.columns

    mdata = MuData(
        {"st": adata_spatial_transcriptome, "sc": sc_data, "meta": adata_meta}
    )

    return mdata


def assess(spatial_transcriptome, atlas, validation) -> MuData:
    """Assess reconstruction quality by correlating with validation data.

    Computes per-gene Pearson correlations between the reconstructed
    spatial transcriptome and both the training atlas and the held-out
    validation dataset.

    Parameters
    ----------
    spatial_transcriptome : MuData
        Spatial transcriptome produced by :func:`transform`.
    atlas : pd.DataFrame
        Training atlas used for the reconstruction.
    validation : pd.DataFrame
        Held-out validation gene expression dataset.

    Returns
    -------
    MuData
        Input object with added ``uns`` entries:
        ``"correlations_genes_training"``,
        ``"correlations_genes_validation"``, and ``"results_recap"``.
    """

    mapping = pd.DataFrame(
        data=spatial_transcriptome["st"].X,
        index=spatial_transcriptome["st"].obs_names,
        columns=remove_index_prefix(spatial_transcriptome["st"].var_names),
    )

    training_genes = spatial_transcriptome["st"].uns["training_genes"]
    validation_genes = validation.columns

    # training genes correlations
    correlations_genes_training = _calculate_gene_correlations(
        mapping, atlas, training_genes, reference_name="Training Genes"
    )

    # validation genes correlations
    correlations_genes_validation = _calculate_gene_correlations(
        mapping, validation, validation_genes, reference_name="Validation Genes"
    )

    results_recap = pd.concat(
        (
            _calculate_gene_correlations_stats(correlations_genes_training, "Training"),
            _calculate_gene_correlations_stats(
                correlations_genes_validation, "Validation"
            ),
        )
    )

    spatial_transcriptome["st"].uns["validation_genes"] = validation_genes.tolist()
    spatial_transcriptome["st"].uns[
        "correlations_genes_training"
    ] = correlations_genes_training
    spatial_transcriptome["st"].uns[
        "correlations_genes_validation"
    ] = correlations_genes_validation
    spatial_transcriptome["st"].uns["results_recap"] = results_recap

    return spatial_transcriptome


def sc_data_handler(sc_data: Union[pd.DataFrame, AnnData]):
    """Normalize single-cell input to AnnData format.

    Accepts either a DataFrame or an AnnData object. For AnnData, extracts
    raw counts from the ``"counts"`` layer or ``raw.X``. For DataFrames,
    wraps into AnnData directly.

    Parameters
    ----------
    sc_data : pd.DataFrame or AnnData
        Single-cell gene expression data.

    Returns
    -------
    AnnData
        Standardized AnnData with raw counts in ``.X`` and a ``"counts"``
        layer.
    """

    if isinstance(sc_data, AnnData):
        if "features" in sc_data.var_names:
            sc_data.var_names = sc_data.var["features"]
        if "counts" in sc_data.layers.keys():
            sc_data.X = sc_data.layers["counts"].astype(np.float32)
        else:
            sc_data.X = sc_data.raw.X.toarray().astype(np.float32)
            sc_data.layers["counts"] = sc_data.X

    elif isinstance(sc_data, pd.DataFrame):
        sc_data_df = sc_data.copy().astype(np.float32)
        sc_data = AnnData(
            sc_data_df, sc_data_df.index.to_frame(), sc_data_df.columns.to_frame()
        )
        sc_data.obs = pd.DataFrame(index=sc_data_df.index)
        sc_data.var = pd.DataFrame(index=sc_data_df.columns)
    else:
        raise ValueError("sc_data should be an AnnData or a pd.DataFrame")

    return sc_data


# ---------------------------------------------------------------------------
# Private helpers — correlation, projection, and validation math
# ---------------------------------------------------------------------------


def _compute_correlation_matrix(atlas: np.array, sc_data: np.array) -> np.array:
    """
    Calculates the correlation between rows of 2 Arrays efficiently
    """
    # Normalize the data
    atlas_norm = (atlas - atlas.mean(axis=1)[:, np.newaxis]) / atlas.std(axis=1)[
        :, np.newaxis
    ]
    sc_data_norm = (sc_data - sc_data.mean(axis=1)[:, np.newaxis]) / sc_data.std(
        axis=1
    )[:, np.newaxis]

    # Compute correlation matrix
    correlation_data = np.dot(atlas_norm, sc_data_norm.T) / atlas.shape[1]

    return correlation_data


def _normalize_correlation_matrix(correlation_data: np.array) -> np.array:
    """
    Normalizes the correlation matrix by dividing each row by the sum of the row
    """
    correlation_sums = np.sum(correlation_data, axis=1, keepdims=True)
    normalized_correlation = correlation_data / correlation_sums
    return normalized_correlation


def _compute_projection_matrix(
    normalized_correlation_data: np.array, sc_data: np.array
) -> np.array:
    """
    Calculates the transcriptome at each position efficiently
    """
    projection_matrix = np.dot(normalized_correlation_data, sc_data)
    return projection_matrix


def _calculate_gene_correlations(
    projection: pd.DataFrame,
    reference: pd.DataFrame,
    genes: Union[list, pd.Index],
    reference_name: str = "reference",
):
    """Per-gene Pearson r between projection and reference for the given genes."""
    genes = list(genes)
    r = per_gene_pearson_r(
        projection[genes].to_numpy(dtype=np.float64),
        reference[genes].to_numpy(dtype=np.float64),
        gene_axis=1,
    )
    df_results = pd.DataFrame(r.astype(float), index=genes, columns=[reference_name])

    return df_results.sort_values(reference_name)


def _calculate_gene_correlations_stats(
    gene_correlations, reference_name: str = "reference"
):
    nb_of_genes = len(gene_correlations.index)
    mean = np.nanmean(gene_correlations.values)
    median = np.nanmedian(gene_correlations.values)
    mini = np.nanmin(gene_correlations.values)
    maxi = np.nanmax(gene_correlations.values)
    q25, q75 = (
        gene_correlations[gene_correlations.columns[0]].quantile([0.25, 0.75]).values
    )

    cols = ["n genes", "mean", "min", "q25%", "median", "q75%", "max"]
    row = [reference_name]
    values = [[nb_of_genes, mean, mini, q25, median, q75, maxi]]

    df_stats = pd.DataFrame(values, columns=cols, index=row)

    return df_stats
