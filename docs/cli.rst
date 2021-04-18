.. _cli:

Command Line Interface
======================

There are two main ``pySYD`` subcommands: ``setup`` and ``run``. 

* `-bg`, `--bg`, `-fitbg`, `--fitbg`, `-background`, `--background` [boolean]

Turn off the background fitting process (although this is not recommended). Asteroseismic estimates are typically unreliable without properly removing stellar contributions from granulation processes. Since this is the money maker, fitbg is set to `True` by default.

* `-ex`, `--ex`, `-findex`, `--findex`, `-excess`, `--excess` [boolean]

Turn off the find excess module. This is only recommended when a list of numaxes or a list of stellar parameters (to estimate the numaxes) are provided. Otherwise the second module, which fits the background will not be able to run properly. Default=`True`

* `-f`, `--f`, `-file`, `--file` [string]

Path to txt file that contains the list of targets to process. Default=`'info/todo.txt'`

* `-filter`, `--filter`, `-smooth`, `--smooth` [float]

Box filter width in muHz for the power spectrum. The default is `2.5` muHz but will change to `0.5` muHz if the numax derived from `find_excess` or the numax provided in `info/stars_info.csv` is <= 500 muHz so that it doesn't oversmooth the power spectrum.

* `-kc`, `--kc`, `-keplercorr`, `--keplercorr` [boolean]

Turn on Kepler short-cadence artefact corrections

* `-mc`, `--mc`, `-mciter`, `--mciter` [int]

Number of MC iterations to run to quantify measurement uncertainties. It is recommended to check the results first before implementing this option and therefore, this is set to `1` by default.

* `-show`, `--show`, `-plot`, `--plot`, `-plots`, `--plots` [boolean]

Shows the appropriate output figures in real time. If the findex module is run, this will show one figure at the end of findex. If the fitbg module is run, a figure will appear at the end of the first iteration. If the monte carlo sampling is turned on, this will provide another figure at the end of the MC iterations. Regardless of this option, the figures will be saved to the output directory. If running more than one target, this is not recommended. 

* `-t`, `--t`, `-target`, `--target`, `-targets`, `--targets` [int]

Option to directly specify targets from the command line. This accepts * arguments and appends them to a list stored in `args.target`. If not specified, `args.target` is `None` and the pipeline will default to the Files/todo.txt file.

* `-version`, `--version`

Print ``pysyd`` package version and exit.

* `-v`, `--v`, `-verbose`, `--verbose` [boolean]

Turn on verbose output
