"""
``pySYD``

"""

import os
import sys

__all__ = ['cli', 'pipeline', 'models', 'target', 'plots', 'utils']
__author__ = 'ashley <achontos@hawaii.edu>'
__version__ = '4.7.8'

_ROOT = os.path.abspath(os.getcwd())
INFDIR = os.path.join(_ROOT, 'info')
INPDIR = os.path.join(_ROOT, 'data')
OUTDIR = os.path.join(_ROOT, 'results')

import sys

# enforce python version
# (same as check at beginning of setup.py)

__minimum_python_version__ = "3.6"

class PythonNotSupportedError(Exception):
    pass


if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise PythonNotSupportedError(
        f"{__package__} does not support Python < {__minimum_python_version__}")