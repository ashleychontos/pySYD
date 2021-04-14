# -*- coding: utf-8 -*-
#
# pySYD documentation buld configuration file, created by
# sphinx-quickstart on 

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

#import mock
import os
import sys
print(sys.path)
sys.path.insert(0, os.path.abspath('.'))
print(sys.path)

#autodoc_mock_imports = ['scipy', 'pandas', 'numpy', 'nbsphinx']
#for mod_name in autodoc_mock_imports:
#    sys.modules[mod_name] = mock.Mock()

import pysyd

nbsphinx_allow_errors = True


# -- Project information -----------------------------------------------------

project = u'pySYD'
copyright = u'2021, Ashley Chontos, Daniel Huber, and Maryum Sayeed'
author = u'Ashley Chontos, Daniel Huber, and Maryum Sayeed'

# The short X.Y version.
version = '.'.join(pysyd.__version__.split('.')[1:])
# The full version, including alpha/beta/rc tags
release = pysyd.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx'
]

# The suffix(es) of source filesnames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'asteroid_sphinx_theme'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "figures/pysyd_logo_inv.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = "figures/pysyd_logo_favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
#htmlhelp_basename = 'pysyddoc'
