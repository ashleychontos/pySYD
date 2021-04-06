import os
import warnings
warnings.filterwarnings("ignore")

__all__ = ['cli', 'functions', 'models', 'plots', 'target', 'utils']

__version__ = '0.0.1'
print(__name__)
print(__path__)
print(__file__)
#__spec__ = __name__
#__package__ = __path__[0]

print(sys)
print(sys.prefix)

MODULEDIR, filename = os.path.split(__file__)
DATADIR = os.path.join(sys.prefix, 'example_data')
print(DATADIR)
if not os.path.isdir(DATADIR):
    warnings.warn("Could not find example_data directory in {}".format(sys.prefix),
                  ImportWarning)
    trydir = os.path.join(os.environ['HOME'], '.local', 'example_data')
    if os.path.isdir(trydir):
        warnings.warn("Found example_data in ~/.local", ImportWarning)
        DATADIR = trydir
    else:
        warnings.warn("Failed to locate example_data directory. Example setup files will not work.",
                      ImportWarning)