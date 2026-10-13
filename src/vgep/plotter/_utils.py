"""Plotting utility helpers."""

import numpy as np


def get_cropping_boundaries(notum_map: np.array):

    """ Automatically crops the image

    Parameters
    ----------
    notum_map : np.array
        The notum map to crop

    Returns
    -------
    x_min, x_max, y_min, y_max : Tuple[float, float, float, float]
        The boundaries of the cropped image
    """

    mask = np.isfinite(notum_map)
    columns = np.sum(mask, axis=1)
    rows = np.sum(mask, axis=0)

    y_mini, y_maxi = np.where(columns != 0)[0][0] , np.where(columns != 0)[0][-1] 
    x_mini, x_maxi = np.where(rows != 0)[0][0], np.where(rows != 0)[0][-1]

    return x_mini, x_maxi, y_mini, y_maxi

