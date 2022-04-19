import os
import sys 

__all__ = ['cli', 'pipeline', 'models', 'target', 'plots', 'utils']
__author__ = 'ashley <achontos@hawaii.edu>'
from .version import __version__ 

# Directory with personal pysyd data & info
_ROOT = os.path.abspath(os.getcwd())
INFDIR = os.path.join(_ROOT, 'info')
INPDIR = os.path.join(_ROOT, 'data')
OUTDIR = os.path.join(_ROOT, 'results')

# Package directory & data
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
SYDFILE = os.path.join(PACKAGEDIR, 'data', 'syd_results.txt')
PYSYDFILE = os.path.join(PACKAGEDIR, 'data', 'pysyd_results.csv')
MPLSTYLE = os.path.join(PACKAGEDIR, 'data', 'pysyd.mplstyle')


# enforce python version
# (same as check at beginning of setup.py)

__minimum_python_version__ = "3.6"

class PythonNotSupportedError(Exception):
    pass


if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise PythonNotSupportedError(
        f"{__package__} does not support Python < {__minimum_python_version__}")