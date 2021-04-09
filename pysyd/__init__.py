import os

__all__ = ['cli', 'functions', 'models', 'plots', 'target', 'utils']

__version__ = '0.0.1'

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_path(path):
    return os.path.join(_ROOT, 'info', path)

TODODIR = get_path('todo.txt')
INFODIR = get_path('star_info.csv')
INPDIR = os.path.join(_ROOT, 'data')