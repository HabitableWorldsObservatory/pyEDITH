import os
import sys
import inspect

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyEDITH"
copyright = "2025 - Alei, Eleonora & Currie, Miles et al"
author = "Alei, Eleonora & Currie, Miles et al."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "myst_parser",
]


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "jupyter_notebook",
}

# NBSphinx settings
nbsphinx_execute = "never"  # Use existing outputs from committed notebooks
nbsphinx_allow_errors = False  # Fail if notebooks have errors in outputs

# MyST settings
myst_enable_extensions = [
    "colon_fence",
]
myst_heading_anchors = 3

templates_path = ["_templates"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "DEFAULT_CONFIG",
}

exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/HabitableWorldsObservatory/pyEDITH",
    "path_to_docs": "docs/source",
    "use_issues_button": True,
    "use_repository_button": True,
}
html_logo = "_static/pyEDITH-removebg-preview.png"  # replace with your logo's filename


# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration


def get_attr(obj, name):
    return getattr(obj, name)


def add_filter_to_env(app):
    app.builder.templates.environment.filters["get_attr"] = get_attr


def setup(app):
    app.connect("builder-inited", add_filter_to_env)
    app.add_css_file("custom.css")

    # Copy tutorial notebooks directly to docs/source during build
    import shutil
    from pathlib import Path

    source_dir = Path(__file__).parent
    tutorials_src = source_dir / "../../tutorials"

    # Copy specific notebooks directly to source folder
    notebooks_to_copy = ["imaging_tutorial.ipynb",
                         "spectroscopy_tutorial.ipynb",
                         "exoearth_spectroscopy_tutorial.ipynb"]

    for notebook in notebooks_to_copy:
        src = tutorials_src / notebook
        dst = source_dir / notebook
        if src.exists():
            shutil.copy2(src, dst)
