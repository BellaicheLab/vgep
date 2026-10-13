"""Sphinx configuration for vGEP documentation."""

import os
import sys

# vgep uses a src/ layout — point autodoc to the package source
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "vGEP"
author = "Adrien Leroy, Lorette Noiret"
copyright = "2026, Bellaiche Lab — Institut Curie"
release = "0.9.0"
version = release

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # Pull docstrings from Python source
    "sphinx.ext.autosummary",   # Auto-generate module summary pages
    "sphinx.ext.napoleon",      # Parse NumPy/Google-style docstrings
    "sphinx.ext.intersphinx",   # Cross-link to numpy/pandas/scipy docs
    "sphinx.ext.viewcode",      # Add "[source]" links beside functions
    "sphinx.ext.mathjax",       # Render LaTeX math in HTML
    "myst_parser",              # Write pages in Markdown (not just RST)
    "nbsphinx",                 # Render Jupyter notebooks as doc pages
    "sphinx_copybutton",        # Copy button on code blocks
    "sphinx_design",            # Cards, tabs, badges, grid layouts
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints",
]
templates_path = ["_templates"]

# -- Mock imports (for ReadTheDocs builds — heavy deps not installed) --------
autodoc_mock_imports = [
    "napari", "zarr", "mudata", "anndata", "scanpy",
    "matplotlib", "seaborn", "scipy", "tqdm",
    "networkx", "igraph", "leidenalg", "tspex",
    "novosparc", "perler", "tangram", "torch", "optuna",
    "statsmodels",
]

# -- Napoleon (docstring parsing) --------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_preprocess_types = True

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_typehints = "description"
autosummary_generate = True

# -- Intersphinx (ecosystem cross-linking) -----------------------------------
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "pandas":     ("https://pandas.pydata.org/docs/", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "anndata":    ("https://anndata.readthedocs.io/en/latest/", None),
    "mudata":     ("https://mudata.readthedocs.io/stable/", None),
    "scanpy":     ("https://scanpy.readthedocs.io/en/stable/", None),
    "networkx":   ("https://networkx.org/documentation/stable/", None),
    "napari":     ("https://napari.org/stable/", None),
    # Cross-repo links (uncomment once each site is deployed on RTD):
    # "morphodomains":      ("https://morphodomains.readthedocs.io/en/latest/", None),
    # "pyanimalprocessing": ("https://pyanimalprocessing.readthedocs.io/en/latest/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Navbar brand: show just the project name; version is rendered as a small
# secondary line below via CSS (.logo__title::after in custom.css).
html_title = project

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/BellaicheLab/vgep",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "show_prev_next": True,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": [],
}

html_context = {
    "github_user": "BellaicheLab",
    "github_repo": "vgep",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Notebooks ---------------------------------------------------------------
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# -- MyST Markdown extensions ------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3
