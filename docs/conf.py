# -*- coding: utf-8 -*-
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# ----------------------------------------------------------------------------
#
# ------------------------------- PATH SETUP ---------------------------------
#
# ----------------------------------------------------------------------------

import os
import re
import sys
import pysyd
import pathlib
import datetime
import warnings
import nbsphinx
from importlib import import_module

nbsphinx_allow_errors = True

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.append('/usr/local/lib/python3.7/site-packages/sphinx_panels')

# Get configuration information from setup.cfg
#from configparser import ConfigParser
#conf = ConfigParser()

#docs_root = pathlib.Path(__file__).parent.resolve()
#conf.read([str(docs_root / '..' / 'setup.cfg')])
#setup_cfg = dict(conf.items('metadata'))


# ----------------------------------------------------------------------------
#
# ------------------------- GENERAL CONFIGURATION ----------------------------
#
# ----------------------------------------------------------------------------


# By default, highlight as Python 3.
highlight_language = 'python3'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# Show summaries of the members in each class along with the
# class' docstring
numpydoc_show_class_members = True

# Whether to create cross-references for the parameter types in the
# Parameters, Other Parameters, Returns and Yields sections of the docstring.
numpydoc_xref_param_type = True

autosummary_generate = True

automodapi_toctreedirnm = 'api'

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = 'obj'

# Class documentation should contain *both* the class docstring and
# the __init__ docstring
autoclass_content = "both"

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog = """
"""

# intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/docs/reference/', None),
    'tqdm': ('https://tqdm.github.io/', None)
}

# Show / hide TODO blocks
#todo_include_todos = True


# ----------------------------------------------------------------------------
#
# ------------------------- PROJECT INFORMATION ------------------------------
#
# ----------------------------------------------------------------------------


project = 'pySYD'
copyright = '2022, Ashley Chontos and contributors'
author = 'Ashley Chontos and contributors'

# This does not *have* to match the package name, but typically does
#project = setup_cfg['name']
#author = setup_cfg['author']
#copyright = '{0}, {1}'.format(
#    datetime.datetime.now().year, setup_cfg['author'])

#package_name = 'pysyd'
#import_module(package_name)
#package = sys.modules[package_name]

#from cmastro import cmaps
#plot_formats = [('png', 200), ('pdf', 200)]
#plot_apply_rcparams = True
# NOTE: if you update these, also update docs/tutorials/nb_setup
#plot_rcparams = {
#    'image.cmap': 'cma:hesperia',

    # Fonts:
#    'font.size': 16,
#    'figure.titlesize': 'x-large',
#    'axes.titlesize': 'large',
#    'axes.labelsize': 'large',
#    'xtick.labelsize': 'medium',
#    'ytick.labelsize': 'medium',

    # Axes:
#    'axes.labelcolor': 'k',
#    'axes.axisbelow': True,

    # Ticks
#    'xtick.color': '#333333',
#    'xtick.direction': 'in',
#    'ytick.color': '#333333',
#    'ytick.direction': 'in',
#    'xtick.top': True,
#    'ytick.right': True,

#    'figure.dpi': 300,
#    'savefig.dpi': 300,
#}
#plot_include_source = False

# The short X.Y version.
version = '.'.join(pysyd.__version__.split('.')[:-1])
# The full version, including alpha/beta/rc tags
release = pysyd.__version__


# ----------------------------------------------------------------------------
#
# ------------------------ OPTIONS FOR HTML OUTPUT ---------------------------
#
# ----------------------------------------------------------------------------


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.

html_logo = "figures/misc/pysyd_logo_rtd.png"

html_theme_options = {
    "logo_link": "index",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ashleychontos/pySYD",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/ashleychontos",
            "icon": "fab fa-twitter-square",
        },
    ],
}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.

#html_favicon = "figures/pysyd_logo_favicon.ico"

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = '{0} v{1}'.format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']
html_css_files = [
    "pysyd.css"
]


# ----------------------------------------------------------------------------
#
# ------------------------ OPTIONS FOR LATEX OUTPUT --------------------------
#
# ----------------------------------------------------------------------------

latex_logo = '_static/latex.png'

latex_elements = {
    'sphinxsetup': 'verbatimwithframe=false',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]

# show inherited members for classes
automodsumm_inherited_members = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add nbsphinx
extensions = [
    'nbsphinx',
    'sphinx_panels',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
 ]

# So sphinx-panels does not include its own CSS classes (different format than rest)
# or at least try this idk
#panels_add_bootstrap_css = False

# Bibliography:
bibtex_bibfiles = ['references.bib']
bibtex_reference_style = 'author_year'
