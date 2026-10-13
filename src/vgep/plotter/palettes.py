"""Colormaps and palettes for spatial transcriptomics visualization."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.pyplot import get_cmap

# ── Continuous colormaps for gene expression ─────────────────────────────────

# Cropped inferno: removes the top 10% of brightest values for better contrast
INFERNO_CROPPED = ListedColormap(get_cmap("inferno")(np.linspace(0, 0.9, 256)))

# White-to-red gradient for atlas vs vGEP comparisons
WHITE_TO_RED = LinearSegmentedColormap.from_list(
    "custom_red",
    [
        "#FEF6F5",  # Very light pink
        "#F4B4A8",  # Peach/salmon
        "#E47058",  # Coral
        "#C93028",  # Red
        "#6B1018",  # Dark burgundy
    ],
)

# ── Colormap conversion ────────────────────────────────────────────────────


def mpl_to_napari(cmap, name=None):
    """Convert a matplotlib colormap to a :class:`napari.utils.Colormap`.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Source matplotlib colormap.
    name : str, optional
        Name for the napari colormap.  Defaults to *cmap.name*.

    Returns
    -------
    napari.utils.Colormap
    """
    from napari.utils import Colormap

    colors = cmap(np.linspace(0, 1, 256))
    return Colormap(colors, name=name or getattr(cmap, "name", "custom"))


# ── Categorical palette ──────────────────────────────────────────────────────

# https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
# orig reference http://epub.wu.ac.at/1692/1/document.pdf
zeileis_28 = [
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#c6dec7",
    "#ead3c6",
    "#f0b98d",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#d5eae7",
    "#f3e1eb",
    "#f6c4e1",
    "#f79cd4",
    # these last ones were added:
    '#7f7f7f',
    "#c7c7c7",
    "#1CE6FF",
    "#336600",
]