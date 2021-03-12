SYD-PY is the python translation of IDL asteroseismic pipeline SYD ([Huber et al. 2009](https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract)) for automated extraction of global asteroseismic parameters. Please note that SYD-PY is actively being developed, so most of the available code and documentation is currently under construction. Feel free to email me at achontos@hawaii.edu for more details or new ideas for implementations within the package.

### * The main goal is asteroseismology-friendly software for the non-asteroseismology user *

## Overview

A pipeline `Target` class object has two main methods:

1) `find_excess`: attempts to find the power excess due to solar-like oscillations using a collapsed ACF (autocorrelation function) using 3 different `box` sizes
2) `fit_background`: perform a fit to the stellar background contribution (i.e. granulation) and measures the global asteroseismic properties 1) frequency corresponding to maximum power (numax or nu_max) and 2) the large frequency separation (delta_nu or dnu).

## Examples

There are three example stars provided in Files/data/: 1435467 (the least evolved), 2309595 (~SG), and 11618103 (RGB). To run a single star, execute the main script with the following command:

- `python sydpy.py -v -show -target 1435467` (or whichever target you'd like)

By default, both `verbose` and `show` (plots) are set to `False` but are helpful to see how the pipeline processes targets. If no `-target` is provided, it will use the list of stars provided in Files/todo.txt.

To estimate uncertainties in the derived parameters for a given target, set `-mc` to something sufficient for random sampling (e.g. 200).

- `python sydpy.py -v -show -target 1435467 -mciter 200`

In the previous example, `-mciter` was not specified and is 1 by default (for 1 iteration). By changing this value, it will randomize the power spectrum and attempt to recover the parameters for the specified number of iterations. The uncertainties will appear in the verbose output, output csvs, and an additional figure will show the distributions of the parameters.

##

### `Scripts`
- `functions.py` : data manipulation tools (i.e. smoothing functions, binning data)
- `sydpy.py` : main pipeline initialization and command line interface tools 
- `models.py` : frequency domain distributions (i.e. Gaussian, Lorentzian, Harvey, etc.)
- `plots.py` : plotting routines
- `target.py` : main pipeline Target class that is initialized for each target that is processed
- `utils.py` : contains information dictionaries and non-science related functions

### Files/

- todo.txt: File containing IDs of stars to be processed 
- data/: Directory containing data to be processed. File format: ID_LC.txt (lightcurve: days versus fractional flux) and ID_PS.txt (power spectrum: muHz versus ppm^2 muHz^-1). 
- star_info.csv: basic information on stars to be processed. If no estimate of numax is provided, the stellar parameters are used to calculate as estimate
- results/: Directory containing result plots and files for each target

## Command Line Interface (CLI) Options

- `-bg`, `--bg`, `-fitbg`, `--fitbg`, `-background`, `--background` [boolean]

Turn off the background fitting process (although this is not recommended). Asteroseismic estimates are typically unreliable without properly removing stellar contributions from granulation processes. Since this is the money maker, fitbg is set to `True` by default.

- `-ex`, `--ex`, `-findex`, `--findex`, `-excess`, `--excess` [boolean]

Turn off the find excess module. This is only recommended when a list of numaxes or a list of stellar parameters (to estimate the numaxes) are provided. Otherwise the second module, which fits the background will not be able to run properly. Default=`True`

- `-f`, `--f`, `-file`, `--file` [string]

Path to txt file that contains the list of targets to process. Default=`'Files/todo.txt'`

- `-filter`, `--filter`, `-smooth`, `--smooth` [float]

Box filter width in muHz for the power spectrum. The default is `2.5` muHz but will change to `0.5` muHz if the numax derived from `find_excess` or the numax provided in `Files/stars_info.csv` is <= 500 muHz so that it doesn't oversmooth the power spectrum.

- `-kc`, `--kc`, `-keplercorr`, `--keplercorr` [boolean]

Turn on Kepler short-cadence artefact corrections

- `-mc`, `--mc`, `-mciter`, `--mciter` [int]

Number of MC iterations to run to quantify measurement uncertainties. It is recommended to check the results first before implementing this option and therefore, this is set to `1` by default.

- `-show`, `--show`, `-plot`, `--plot`, `-plots`, `--plots` [boolean]

Shows the appropriate output figures in real time. If the findex module is run, this will show one figure at the end of findex. If the fitbg module is run, a figure will appear at the end of the first iteration. If the monte carlo sampling is turned on, this will provide another figure at the end of the MC iterations. Regardless of this option, the figures will be saved to the output directory. If running more than one target, this is not recommended. 

- `-t`, `--t`, `-target`, `--target`, `-targets`, `--targets` [int]

Option to directly specify targets from the command line. This accepts * arguments and appends them to a list stored in `args.target`. If not specified, `args.target` is `None` and the pipeline will default to the Files/todo.txt file.

- `-v`, `--v`, `-verbose`, `--verbose` [boolean]

Turn on verbose output

#### See `main -help` for more details.

## Tutorials 

Follow examples in

- `path_to_notebook.ipynb`
- `boop`

## Attribution

Written by Ashley Chontos. Developed by Ashley Chontos, Daniel Huber, and Maryum Sayeed. 

Please cite the [original publication](https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract) and the following DOI if you make use of this software in your research.
[need to do]

## Documentation

Documentation is available [here]

## Troubleshooting

If you would like to use multiprocessing for a large number of targets and are running MacOSX Mojave, High Sierra, or Catalina, you must do the following before it will execute:

1) Add `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` to the TOP of your bashrc or zshrc file (or bash_profile - whichever you use).
2) Close all terminal windows, reopen and it should work.

If you do not do this, you will see some version of the error for n threads:

objc[4182]: +[__NSPlaceholderDictionary initialize] may have been in progress in another thread when fork() was called. We cannot safely call it or ignore it in the fork() child process. Crashing instead. Set a breakpoint on objc_initializeAfterForkError to debug.
