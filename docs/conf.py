# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# pipescript is pip-installed in the Read the Docs build (see .readthedocs.yml),
# so no sys.path manipulation is needed for autodoc to import it. For a local
# ``make -C docs html`` from a source checkout, an editable install (``pip
# install -e '.[docs]'``) makes ``import pipescript`` resolve just the same.

# -- Project information -----------------------------------------------------

project = "pipescript"
copyright = "2025, Stephen Macke"
author = "Stephen Macke"

# The full version, including alpha/beta/rc tags, and the short X.Y version.
try:
    from pipescript import __version__ as release
except Exception:
    try:
        from pipescript.version import version as release
    except Exception:
        release = ""
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

# ref: https://stackoverflow.com/questions/56336234/build-fail-sphinx-error-contents-rst-not-found
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Link out to the CPython docs (pipescript references the ``ast`` and
# ``functools`` modules).
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Keep autodoc output in source order rather than alphabetized.
autodoc_member_order = "bysource"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
html_title = f"pipescript {version}" if version else "pipescript"

# Read the Docs theme options: keep the full nav tree expanded and deep enough
# to reach the Diátaxis subsections. See
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
