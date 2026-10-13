"""MuData modality prefix manipulation."""

import pandas as pd


def remove_index_prefix(index: pd.Index) -> pd.Index:
    """Remove the modality prefix from a MuData index.

    Strips the ``"modality:"`` prefix (e.g. ``"st:gene"`` → ``"gene"``).

    Parameters
    ----------
    index : pd.Index
        Index with ``"modality:name"`` format.

    Returns
    -------
    pd.Index
        Index with prefixes removed.
    """
    index = index.str.split(":", expand=True)
    index = [i[1] for i in index.to_list()]
    index = pd.Index(index)

    return index


def add_index_prefix(index: pd.Index, modality: str) -> pd.Index:
    """Add a modality prefix to an index.

    Prepends ``"modality:"`` to each element (e.g. ``"gene"`` → ``"st:gene"``).

    Parameters
    ----------
    index : pd.Index
        Index to prefix.
    modality : str
        Modality name to prepend.

    Returns
    -------
    pd.Index
        Prefixed index.
    """

    index = [modality + ":" + i for i in index.to_list()]
    index = pd.Index(index)
    return index


def change_index_prefix(index: pd.Index, modality: str) -> pd.Index:
    """Replace the modality prefix on an index.

    Parameters
    ----------
    index : pd.Index
        Index with an existing ``"modality:name"`` prefix.
    modality : str
        New modality name.

    Returns
    -------
    pd.Index
        Index with the new prefix.
    """

    index = remove_index_prefix(index)
    index = add_index_prefix(index, modality)
    return index


def change_modality(dataframe: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Change or remove the modality prefix on a DataFrame index.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame whose index has ``"modality:name"`` format.
    modality : str
        New modality name. Pass ``""`` to remove the prefix entirely.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated index.
    """
    if modality == "":
        dataframe.index = remove_index_prefix(dataframe.index)
    else:
        dataframe.index = change_index_prefix(dataframe.index, modality)
    return dataframe


def has_prefix(dataframe: pd.DataFrame):
    """Check whether a DataFrame index contains a modality prefix.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to check.

    Returns
    -------
    bool
        True if the first index element contains ``":"``.
    """
    index = dataframe.index
    if ":" in index[0]:
        return True
    else:
        return False

