import os

from .functions import *
from .pipeline import *
from .models import *
from .target import *
from .plots import *
from .utils import *

__all__ = ['cli', 'functions', 'pipeline', 'models', 'target', 'plots', 'utils']

__version__ = '0.4.7'

_ROOT = os.path.abspath(os.getcwd())
TODODIR = os.path.join(_ROOT, 'info', 'todo.txt')
INFODIR = os.path.join(_ROOT, 'info', 'star_info.csv')
INPDIR = os.path.join(_ROOT, 'data')
OUTDIR = os.path.join(_ROOT, 'results')