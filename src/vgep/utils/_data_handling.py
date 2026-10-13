"""Spatial data format conversion and rasterization."""

from typing import Optional, Union

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData, read_zarr
from mudata import MuData

from ._mudata import remove_index_prefix


def to_matrix_form(
    df_projection: Union[AnnData, MuData, pd.DataFrame, zarr.Group],
    geometry: Optional[pd.DataFrame] = None,
    crop=False,
    color: Union[str, None] = None,
    target_shape: Optional[tuple] = None,
) -> np.array:
    """Convert spatial transcriptome data to a 2D or 3D numpy array on tissue coordinates.

    Maps feature values onto a spatial grid defined by the tissue geometry.

    Parameters
    ----------
    df_projection : AnnData, MuData, pd.DataFrame, or zarr.Group
        Spatial transcriptome data to rasterize.
    geometry : pd.DataFrame, optional
        Spatial coordinates with 'x' and 'y' columns. Extracted automatically
        from AnnData/MuData obs if not provided.
    crop : bool
        If True, crop the output to the bounding box of non-NaN values.
    color : str, optional
        Column name to select (gene name or obs column). If None, all
        columns are included (returns 3D array).
    target_shape : tuple of int, optional
        Target ``(height, width)`` for the output. Pads with NaN or crops
        to match.

    Returns
    -------
    np.ndarray
        2D array ``(x, y)`` if a single feature, or 3D array
        ``(features, x, y)`` if multiple features.
    """

    # Convert zarr.Group to AnnData
    if isinstance(df_projection, zarr.Group):
        df_projection = read_zarr(df_projection)

    # Handling MuData type for df_projection
    if isinstance(df_projection, MuData):
        df_projection = df_projection["st"]

    # Handling AnnData type for df_projection
    if isinstance(df_projection, AnnData):
        if geometry is None:
            geometry = df_projection.obs.loc[:, ["x", "y"]]

        obs = df_projection.obs
        df_projection = df_projection.to_df()
        df_projection.columns = remove_index_prefix(df_projection.columns)
        df_projection = pd.concat([df_projection, obs], axis=1)

    elif isinstance(df_projection, pd.DataFrame):
        if geometry is None:
            if isinstance(df_projection.index, pd.MultiIndex) and set(
                df_projection.index.names
            ) == {"x", "y"}:
                geometry = df_projection.index.to_frame(index=False)
            else:
                raise ValueError(
                    "The geometry of the tissue must be provided if the transcriptome is a pd.DataFrame "
                    "and its index is not a MultiIndex with levels 'x' and 'y'."
                )

    if color is not None:
        df_projection = df_projection[color]

    if isinstance(getattr(df_projection, "dtype", None), pd.CategoricalDtype):
        values, _ = convert_to_numeric(df_projection, replace_nan=False)
    else:
        values = df_projection.values

    x = geometry["x"].values.astype(int)
    y = geometry["y"].values.astype(int)

    # Handle potential negative values in geometry
    x_offset = min(0, np.min(x))
    y_offset = min(0, np.min(y))

    x = x - x_offset
    y = y - y_offset

    nx = np.max(x) + 1
    ny = np.max(y) + 1

    if len(df_projection.shape) == 1:
        expression_matrix = np.empty((nx, ny))
        expression_matrix[:] = np.nan
        expression_matrix[x, y] = values
    else:
        expression_matrix = np.empty([nx, ny, df_projection.shape[1]])
        expression_matrix[:] = np.nan
        expression_matrix[x, y, :] = values
        # We want to first dimension to be the list of genes
        # x y z(genes) ==> y x z(genes)
        expression_matrix = np.swapaxes(expression_matrix, 0, 1)
        # y x z(genes) ==> z(genes) x y
        expression_matrix = np.swapaxes(expression_matrix, 0, 2)

    if crop:
        if len(df_projection.shape) == 1:
            mask = np.isfinite(expression_matrix)
            columns = np.sum(mask, axis=0)
            rows = np.sum(mask, axis=1)
            y_mini, y_maxi = np.where(columns != 0)[0][0], np.where(columns != 0)[0][-1]
            x_mini, x_maxi = np.where(rows != 0)[0][0], np.where(rows != 0)[0][-1]
            expression_matrix = expression_matrix[
                x_mini - 1 : x_maxi + 2, y_mini - 1 : y_maxi + 2
            ]
        else:
            mask = np.isfinite(expression_matrix[0])
            columns = np.sum(mask, axis=0)
            rows = np.sum(mask, axis=1)
            y_mini, y_maxi = np.where(columns != 0)[0][0], np.where(columns != 0)[0][-1]
            x_mini, x_maxi = np.where(rows != 0)[0][0], np.where(rows != 0)[0][-1]
            expression_matrix = expression_matrix[
                :, x_mini - 1 : x_maxi + 2, y_mini - 1 : y_maxi + 2
            ]

    # Adjust matrix to target shape if specified
    if target_shape is not None:
        if len(df_projection.shape) == 1:
            current_shape = expression_matrix.shape
            target_height, target_width = target_shape
            
            # Create a new matrix with target shape filled with NaN
            adjusted_matrix = np.empty(target_shape)
            adjusted_matrix[:] = np.nan
            
            # Copy data from original matrix, limited by the smaller dimensions
            height_to_copy = min(current_shape[0], target_height)
            width_to_copy = min(current_shape[1], target_width)
            
            adjusted_matrix[:height_to_copy, :width_to_copy] = expression_matrix[:height_to_copy, :width_to_copy]
            expression_matrix = adjusted_matrix
        else:
            current_shape = expression_matrix.shape[1:]  # Skip the genes dimension
            target_height, target_width = target_shape
            
            # Create a new matrix with target shape filled with NaN
            adjusted_matrix = np.empty((expression_matrix.shape[0], target_height, target_width))
            adjusted_matrix[:] = np.nan
            
            # Copy data from original matrix, limited by the smaller dimensions
            height_to_copy = min(current_shape[0], target_height)
            width_to_copy = min(current_shape[1], target_width)
            
            adjusted_matrix[:, :height_to_copy, :width_to_copy] = expression_matrix[:, :height_to_copy, :width_to_copy]
            expression_matrix = adjusted_matrix

    return expression_matrix


def convert_to_numeric(serie: pd.Series, replace_nan=False):
    """Convert a categorical Series to numeric values.

    Maps each unique category to a 1-indexed integer. NaN values are
    optionally replaced by zeros.

    Parameters
    ----------
    serie : pd.Series
        Categorical series to convert.
    replace_nan : bool
        If True, replace NaN values with 0.

    Returns
    -------
    pd.Series
        Series with categories replaced by integer codes.
    list
        Ordered list of original category labels.
    """
    labels = []
    conversion = {}
    serie_numeric = serie.copy().astype("object")

    if replace_nan is True:
        serie_numeric[serie_numeric.isna()] = 0

    # Get unique non-NaN values and sort them
    unique_values = [v for v in serie.unique() if pd.notna(v)]
    sorted_values = sorted(unique_values)

    # Create a conversion dictionary
    for i, value in enumerate(sorted_values, 1):
        labels.append(value)
        conversion[value] = i

    serie_numeric.replace(conversion, inplace=True)
    return serie_numeric, labels
