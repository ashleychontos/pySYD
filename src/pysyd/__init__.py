__bibtex__ = """
@article{pysyd,
       author = {{Chontos}, Ashley and {Huber}, Daniel and {Sayeed}, Maryum and {Yamsiri}, Pavadol},
        title = "{$\texttt{pySYD}$: Automated measurements of global asteroseismic parameters}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2021,
        month = aug,
          eid = {arXiv:2108.00582},
        pages = {arXiv:2108.00582},
archivePrefix = {arXiv},
       eprint = {2108.00582},
 primaryClass = {astro-ph.SR}, 
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210800582C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
__url__ = "https://pysyd.readthedocs.io"
__author__ = "Ashley Chontos<ashleychontos@astro.princeton.edu>"
__license__ = "MIT"
__description__ = "Automated measurements of global asteroseismic parameters"
__version__ = '6.10.5'

__all__ = ['cli','models','pipeline','plots','target','utils']

import os
import sys 

# Directory with personal pysyd data & info
_ROOT = os.path.abspath(os.getcwd())

# Package directory & data
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

# enforce python version
# (same as check at beginning of setup.py)

__minimum_python_version__ = "3.8"

class PythonNotSupportedError(Exception):
    pass

if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise PythonNotSupportedError(
        f"{__package__} does not support Python < {__minimum_python_version__}")