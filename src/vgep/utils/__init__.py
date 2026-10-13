"""Data handling and MuData index utilities."""

from ._mudata import remove_index_prefix, change_index_prefix, change_modality, has_prefix, add_index_prefix
from ._data_handling import to_matrix_form, convert_to_numeric

__all__ = [
    "to_matrix_form", "convert_to_numeric",
    "remove_index_prefix", "change_index_prefix", "change_modality", "has_prefix", "add_index_prefix",
]
