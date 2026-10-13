"""Tissue-specific gene scoring using tau statistics."""

import logging

import numpy as np
import tspex

logger = logging.getLogger(__name__)


def compute_tau_scores(clusters, transcriptome_df):
    """
    Compute tau specificity scores from a transcriptome dataframe and cluster assignments.

    Parameters
    ----------
    clusters : pd.Series
        Cluster assignments for each position.
    transcriptome_df : pd.DataFrame
        Spatial transcriptome expression data.

    Returns
    -------
    tuple of (pd.Series, pd.DataFrame)
        - tau_scores: tau specificity scores (pandas Series where indices are gene names)
        - average_expression: the average expression per cluster (DataFrame with genes as rows)
    """
    # Filter to valid coordinates (non-null cluster assignments)
    valid_coords = clusters.dropna().index
    # Use only the coordinates that exist in transcriptome_df to avoid KeyError
    common_coords = transcriptome_df.index.intersection(valid_coords)
    expr_df = transcriptome_df.loc[common_coords]
    clusters = clusters.loc[common_coords]

    # Compute average expression per cluster (genes as rows, clusters as columns)
    average_expression = expr_df.groupby(clusters).mean().T

    # Compute tau scores using tspex
    tissue_spec = tspex.TissueSpecificity(average_expression, "tau")
    tau_scores = tissue_spec.tissue_specificity

    logger.info("Computed tau scores for %d genes across %d clusters",
                len(tau_scores), average_expression.shape[1])

    return tau_scores, average_expression


def filter_tau_by_max_expression(tau_scores, expression_df, valid_columns):
    """
    Filter tau scores to keep only genes whose maximum expression (Home)
    is in one of the valid_columns.

    Parameters
    ----------
    tau_scores : pd.Series
        The Tau scores.
    expression_df : pd.DataFrame
        The combined expression matrix (Genes x Groups).
    valid_columns : list or Index
        List of column names in expression_df that are valid "Homes".

    Returns
    -------
    pd.Series
        Filtered Tau scores.
    """
    # Get the column name with max expression for each gene
    max_col = expression_df.idxmax(axis=1)

    # Check if the max column is in the valid set
    is_valid_home = max_col.isin(valid_columns)

    filtered_tau = tau_scores[is_valid_home]

    logger.info(
        "Filtering by max expression: kept %d/%d genes",
        len(filtered_tau), len(tau_scores),
    )
    return filtered_tau


def compute_pseudobulk_zscores(pseudobulk_df):
    """
    Compute per-gene z-scores using the pseudobulk expression profiles (log1p transformed).
    This normalizes expression across all provided conditions (spatial clusters + cell types).

    Parameters
    ----------
    pseudobulk_df : pd.DataFrame
        DataFrame with genes as rows and conditions (clusters/cell types) as columns.

    Returns
    -------
    pd.DataFrame
        Z-score normalized expression matrix (genes x conditions).
    """
    # Log-transform to stabilize variance
    log_df = np.log1p(pseudobulk_df)

    # Compute Z-scores across columns (axis=1) for each gene
    means = log_df.mean(axis=1)
    stds = log_df.std(axis=1)

    # Avoid division by zero for genes with constant expression
    stds[stds == 0] = 1.0

    # (X - mean) / std
    z_scores = log_df.sub(means, axis=0).div(stds, axis=0)
    return z_scores


def find_specific_genes(tau_scores, quantile=0.9, threshold=None):
    """
    Compute specific genes based on tau values using either a quantile cutoff or a hard threshold.

    By default, selects genes whose tau scores are in the top 10% (quantile=0.9).
    If 'threshold' is provided, it takes precedence over 'quantile' and selects genes
    with tau >= threshold.

    Parameters
    ----------
    tau_scores : pd.Series
        Series of tau scores indexed by gene.
    quantile : float, optional
        Quantile to use for thresholding (default 0.9). Ignored if 'threshold' is provided.
    threshold : float, optional
        Hard cutoff to use instead of a quantile. If provided, overrides 'quantile'.

    Returns
    -------
    tuple of (pd.Series, pd.Series)
        - A boolean series indicating, for each gene, whether it is specific
          (True if its tau score passes the chosen cutoff).
        - A series containing the tau specificity scores.
    """
    # Decide cutoff: threshold overrides quantile
    if threshold is not None:
        cutoff = threshold
    else:
        # Fallback to provided quantile or default 0.9 if None
        q = 0.9 if quantile is None else quantile
        cutoff = tau_scores.quantile(q)

    is_specific = tau_scores >= cutoff
    logger.info("Specificity filter (cutoff=%.4f): %d/%d genes are specific (%.1f%%)",
                cutoff, is_specific.sum(), len(is_specific),
                100 * is_specific.sum() / len(is_specific))
    return is_specific, tau_scores


def argmax_one_hot(df):
    """Return one-hot rows indicating the column of maximum value per row.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric DataFrame.

    Returns
    -------
    np.ndarray
        One-hot encoded array (n_rows x n_cols).
    """
    arr = df.values
    argmax_idx = np.argmax(arr, axis=1)
    return np.eye(arr.shape[1], dtype=int)[argmax_idx]
