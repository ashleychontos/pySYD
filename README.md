<div align="center">
<img src="docs/figures/pysyd_logo_inv.png" width="75%">

**Open-source asteroseismology: automated extraction of global asteroseismic properties**

[![PyPI Status](https://badge.fury.io/py/pysyd.svg)](https://badge.fury.io/py/pysyd)
[![Documentation Status](https://readthedocs.org/projects/pysyd/badge/?version=latest)](https://pysyd.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

</div>

--------------------------------------------------------------------------------


`pySYD` is the python translation of IDL asteroseismic pipeline SYD ([Huber et al. 2009](https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract)) for automated extraction of global asteroseismic parameters. Please note that `pySYD` is actively being developed, so most of the available code and documentation is currently under construction. Feel free to email me at achontos@hawaii.edu for more details or new ideas for implementations within the package.

## Overview

A `pySYD` pipeline `Target` class object has two main methods:

1) `find_excess`: attempts to find the power excess due to solar-like oscillations using a collapsed ACF (autocorrelation function) using 3 different `box` sizes
2) `fit_background`: perform a fit to the stellar background contribution (i.e. granulation) and measures the global asteroseismic properties 1) frequency corresponding to maximum power (numax or nu_max) and 2) the large frequency separation (delta_nu or dnu).

## Examples

There are three example stars provided in pysyd/data/: 1435467 (the least evolved), 2309595 (~SG), and 11618103 (RGB). To run a single star, execute the main script with the following command:

- `pysyd -star 1435467 -show -verbose` (or whichever target you'd like)

By default, both `verbose` and `show` (plots) are set to `False` but are helpful to see how the pipeline processes targets. If no `-target` is provided, it will use the list of stars provided in pysyd/info/todo.txt.

To estimate uncertainties in the derived parameters for a given target, set `-mc` to something sufficient for random sampling (e.g. 200).

- `pysyd -star 1435467 -show -verbose -mc 200`

In the previous example, `-mciter` was not specified and is 1 by default (for 1 iteration). By changing this value, it will randomize the power spectrum and attempt to recover the parameters for the specified number of iterations. The uncertainties will appear in the verbose output, output csvs, and an additional figure will show the distributions of the parameters.

##

### `Scripts`
- `functions.py` : data manipulation tools (i.e. smoothing functions, binning data)
- `cli.py` : sets command line interface tools used with the main pipeline initialization
- `models.py` : frequency domain distributions (i.e. Gaussian, Lorentzian, Harvey, etc.)
- `pipeline.py` : main pipeline initialization, including parallelization capabilities
- `plots.py` : plotting routines
- `target.py` : main pipeline Target class that is initialized for each processed star
- `utils.py` : contains information dictionaries and non-science related functions (i.e. load/check input data)

### `Package Data`

- info/todo.txt: Basic text file containing IDs of stars to be processed (one star ID per line)
- data/: Directory containing data to be processed. File format: ID_LC.txt (lightcurve: days versus fractional flux) and ID_PS.txt (power spectrum: muHz versus ppm^2 muHz^-1). NOTE: this is not created and must be handled manually. There is example data included in the `pip install pysyd` package (in pysyd/data), which we suggest to copy over to a more local, accessible directory. This will successfully read in data if there is a 'data/' directory wherever `pysyd` is initialized from.
- info/star_info.csv: individual star information, but is not a requirement for the pipeline to run. If stellar properties are provided, `pySYD` will estimate a value for numax and use that as an initial starting point. Targets in this csv do not need to be in any specific order nor do they need to exactly include all targets in todo.txt. This can be a file that contains thousands of stars, where only a subset is run (i.e. think of this as a master dictionary for stellar information). In order to read information in properly, be sure that `star_info.csv` has exactly the following column heads:
  - "stars" : star IDs that should exactly match the targets provided via command line or in todo.txt
  - "rad" : stellar radius (in solar radii)
  - "teff" : effective temperature (K)
  - "logg" : surface gravity (dex)
  - "numax" : the frequency corresponding to maximum power (in muHz). If no information is provided, we suggest that the user first run findex (that runs by default anyway on a given target), which will find a starting point for numax. However, if the pipeline does not find the proper numax, providing a value in this csv will override the first module's value and use the provided value instead.
  - "lowerx" : lower frequency limit to use in the findex module (in muHz)
  - "upperx" : upper frequency limit to use in the findex module (in muHz)
  - "lowerb" : lower frequency limit to use in the background-fitting module (in muHz)
  - "upperb" : upper frequency limit to use in the background-fitting module (in muHz)
  - "seed" : random seed generated when using the Kepler correction option, which is saved for future reproducibility purposes.
- results/: Directory containing result plots and files for each target. Currently, it will create this directory (if it does not already exist) in the current working directory of the user.

## Citing pySYD

DOI + ASCL

## Command Line Interface (CLI) Options

- `-bg`, `--bg`, `-fitbg`, `--fitbg`, `-background`, `--background` [boolean]

Turn off the background fitting process (although this is not recommended). Asteroseismic estimates are typically unreliable without properly removing stellar contributions from granulation processes. Since this is the money maker, fitbg is set to `True` by default.

- `-ex`, `--ex`, `-findex`, `--findex`, `-excess`, `--excess` [boolean]

Turn off the find excess module. This is only recommended when a list of numaxes or a list of stellar parameters (to estimate the numaxes) are provided. Otherwise the second module, which fits the background will not be able to run properly. Default=`True`

- `-f`, `--f`, `-file`, `--file` [string]

Path to txt file that contains the list of targets to process. Default=`'info/todo.txt'`

- `-filter`, `--filter`, `-smooth`, `--smooth` [float]

Box filter width in muHz for the power spectrum. The default is `2.5` muHz but will change to `0.5` muHz if the numax derived from `find_excess` or the numax provided in `info/stars_info.csv` is <= 500 muHz so that it doesn't oversmooth the power spectrum.

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

#### See `pysyd --help` for more options.

## Tutorials 

Follow examples in

- `path_to_notebook.ipynb`
- `boop`

## Attribution

Written by Ashley Chontos. Developed by Ashley Chontos, Daniel Huber, and Maryum Sayeed. 

Please cite the [original publication](https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract) and the following DOI if you make use of this software in your research.
[need to do]

## Documentation

Documentation is available [here](https://pysyd.readthedocs.io)

## Troubleshooting

If you would like to use multiprocessing for a large number of targets and are running MacOSX Mojave, High Sierra, or Catalina, you must do the following before it will execute:

1) Add `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` to the TOP of your bashrc or zshrc file (or bash_profile - whichever you use).
2) Close all terminal windows, reopen and it should work.

If you do not do this, you will see some version of the error for n threads:

objc[4182]: +[__NSPlaceholderDictionary initialize] may have been in progress in another thread when fork() was called. We cannot safely call it or ignore it in the fork() child process. Crashing instead. Set a breakpoint on objc_initializeAfterForkError to debug.
