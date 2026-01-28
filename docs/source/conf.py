
import os
import sys
from pathlib import Path
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'brainprep'
copyright = '2026, Andjela Dimitrijevic'
author = 'Andjela Dimitrijevic'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Make your package/scripts importable for autodoc
ROOT = Path(__file__).resolve().parents[2]   # repo root
sys.path.insert(0, str(ROOT))

project = "brainprep"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # Google/Numpy docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "myst_parser",                  # Markdown support
    "sphinx_autodoc_typehints",     # show type hints
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

# MyST options (so Markdown works nicely)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
]

