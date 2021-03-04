SYD-PYpline is the python translation of IDL asteroseismic pipeline SYD ([Huber et al. 2009](https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract)) for automated extraction of global asteroseismic parameters. Please note that SYD-PY is actively being developed, so most of the available code and documentation is currently under construction. Feel free to email me at achontos@hawaii.edu for more details or new ideas for implementations within the package.

### ** The main goal is asteroseismology-friendly software for the non-asteroseismology user **

## Overview

A pipeline `Target` class object has two main methods:

1) `find_excess`: finds power excess due to solar-like oscillations using a frequency resolved collapsed autocorrelation function
2) `fit_background`: perform a fit to the granulation background and measures the frequency corresponding to maximum power (numax or nu_max) and the large frequency separation (delta_nu or dnu).

### Scripts
- `main.py` : command line interface tools and initiates main pipeline 
- `target.py` : Target class which is initialized for each target given in Files/todo.txt
- `models.py` : any frequency domain distributions (i.e. Gaussian, Lorentzian, Harvey, etc.)
- `functions.py` : models, distributions, ffts, smoothing functions are all in this file
- `utils.py` : mostly contains information dictionaries and non-science related functions
- `scrape_output.py` : takes each individual target's results and concatenates results into a single csv in Files/ for each submodulel (i.e. findex.csv and globalpars.csv). This is automatically called at the end of the main module.

### `Files/`

- `todo.txt`: File containing IDs of stars to be processed 
- `data/`: Directory containing data to be processed. File format: ID_LC.txt (lightcurve: days versus fractional flux) and ID_PS.txt (power spectrum: muHz versus ppm^2 muHz^-1). 
- `star_info.csv`: basic information on stars to be processed. If no estimate of numax is provided, the stellar parameters are used to calculate as estimate
- `results/`: Directory containing result plots and files for each target

Documentation and code under construction.

## Example

To run example code, clone the repository, cd into the repo and then execute the main script:

- `python main.py` 

## CLI Options

`-ex`, `--ex`, `-findex`, `--findex`, `-excess`, `--excess`

Turn off the find excess module. This is only recommended when a list of numaxes or a list of stellar parameters (to estimate the numaxes) are provided. Otherwise the second module, which fits the background will not be able to run properly. Default=True

`-bg`, `--bg`, `-fitbg`, `--fitbg`, `-background`, `--background`

Turn off the background fitting process (although this is not recommended). Asteroseismic estimates are typically unreliable without properly removing stellar contributions from granulation processes. Since this is the money maker, fitbg is set to 'True' by default.

`-filter`, `--filter`, `-smooth`, `--smooth`

Box filter width [muHz] for the power spectrum (Default = 2.5 muHz)

`-kc`, `--kc`, `-keplercorr`, `--keplercorr`

Turn on Kepler short-cadence artefact corrections

`-v`, `--v`, `-verbose`, `--verbose`

Turn on verbose output

`-show`, `--show`, `-plot`, `--plot`, `-plots`, `--plots`,

Shows the appropriate output figures in real time. If the findex module is run, this will show one figure at the end of findex. If the fitbg module is run, a figure will appear at the end of the first iteration. If the monte carlo sampling is turned on, this will provide another figure at the end of the MC iterations. Regardless of this option, the figures will be saved to the output directory. If running more than one target, this is not recommended. 

`-mc`, `--mc`, `-mciter`, `--mciter`

Number of MC iterations to run to quantify measurement uncertainties. It is recommended to check the results first before implementing this option and therefore, this is set to 1 by default.

#### See `python main.py -help` for more details.

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
