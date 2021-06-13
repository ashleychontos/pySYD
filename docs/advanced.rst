.. _advanced:

Advanced Usage
#################

Initialization of ``pySYD`` via command line will use the following paths:

- ``TODODIR`` : '~/path_to_put_pysyd_stuff/info/todo.txt'
- ``INFODIR`` : '~/path_to_put_pysyd_stuff/info/star_info.csv'
- ``INPDIR`` : '~/path_to_put_pysyd_stuff/data'
- ``OUTDIR`` : '~/path_to_put_pysyd_stuff/results'

which by default, is the absolute path of the current working directory (or however you choose to set it up). All of these paths should be ready to go
if you followed the suggestions in :ref:`structure` or used our ``setup`` feature.

Modes
======

There are currently three modes that ``pySYD`` can operate in via command line: 

#. ``setup`` : Initializes ``pysyd.pipeline.setup`` for quick and easy setup of directories, files and examples. This mode only
   inherits higher level functionality and has limited CLI (see :ref:`parent parser<parentparse>` below).

#. ``run`` : The main pySYD pipeline function is initialized through ``pysyd.pipeline.run`` and runs the two core modules 
   (i.e. ``find_excess`` and ``fit_background``) for each star consecutively. This mode operates using most CLI options, inheriting
   both the :ref:`parent<parentparse>` and :ref:`main parser<mainparse>` options.

#. ``parallel`` : Operates the same way as the previous mode, but processes stars simultaneously in parallel. Based on the number of threads
   available, stars are separated into groups (where the number of groups is exactly equal to the number of threads). This mode uses all CLI
   options, including the number of threads to use for parallelization (:ref:`see here<parallel>`).

=======

.. _cli:

Command Line Interface
#########################

.. _parentparse:

Parent Parser
================

Higher level functionality of the software. All three modes inherent the parent parser.

- ``-file``, ``--file``, ``-list``, ``--list``, ``-todo``, ``--todo``
   * dest = ``args.file``
   * help = Path to text file that contains the list of stars to process (convenient for running many stars).
   * type = ``str``
   * default = ``TODODIR``
- ``-in``, ``--in``, ``-input``, ``--input``, ``-inpdir``, ``--inpdir``
   * dest = ``args.inpdir``
   * help = Path to input data
   * type = ``str``
   * default = ``INPDIR``
- ``-info``, ``--info``, ``-information``, ``--information`` 
   * dest = ``args.info``
   * help = Path to the csv containing star information (although not required).
   * type = ``str``
   * default = ``INFODIR``
- ``-out``, ``--out``, ``-output``, ``--output``, ``-outdir``, ``--outdir``
   * dest = ``args.outdir``
   * help = Path that results are saved to
   * type = ``str``
   * default = ``OUTDIR``
- ``-verbose``, ``--verbose``
   * dest = ``args.verbose``
   * help = Turn on verbose output
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``

.. _mainparse:

Main Parser
==============

Accesses all science-related functions and is therefore for both ``run`` and ``parallel`` modes.

- ``-bg``, ``--bg``, ``-fitbg``, ``--fitbg``, ``-background``, ``--background``
   * dest = ``args.background``
   * help = Turn off the background fitting procedure
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``-cad``, ``--cad``, ``-cadence``, ``--cadence``
   * dest = ``args.cadence``
   * help = cadence of time series (used to calculate nyquist frequency), which will automatically be calculated when time series data is available. 
   * type = ``int``
   * default = `0`
   * unit = seconds
- ``-ex``, ``--ex``, ``-findex``, ``--findex``, ``-excess``, ``--excess``
   * dest = ``args.background``
   * help = turn off find excess module
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``-kc``, ``--kc``, ``-kepcorr``, ``--kepcorr``
   * dest = ``args.kepcorr``
   * help = turn on the *Kepler* short-cadence artefact correction module
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-nyq``, ``--nyq``, ``-nyquist``, ``--nyquist``
   * dest = ``args.nyquist``
   * help = nyquist frequency of the power spectrum (relevant for when the time series is not provided) 
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``-ofa``, ``--ofa``, ``-of_actual``, ``--of_actual``
   * dest = ``args.of_actual``
   * help = The oversampling factor of the provided power spectrum. Default is `0`, which means it is calculated from the time series data. Note: This needs to be provided if there is no time series data!
   * type = ``int``
   * default = `0`
- ``-ofn``, ``--ofn``, ``-of_new``, ``--of_new``
   * dest = ``args.of_new``
   * help = The new oversampling factor to use in the first iterations of both modules. Default is `5` (see performance for more details).
   * type = int
   * default = `5`
- ``-save``, ``--save``
   * dest = ``args.save``
   * help = save output files and figures
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``-show``, ``--show`` 
   * dest = ``args.show``
   * help = show output figures (note: this is not recommended if running many stars)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-star``, ``--star``, ``-stars``, ``--stars``
   * dest = ``args.star``
   * help = List of stars to process. Default is ``None``, which will read in the star list from ``args.file``.
   * nargs = '*'
   * default = ``None``
   
Excess Group
***************

- ``-bin``, ``--bin``, ``-binning``, ``--binning``
   * dest = ``args.binning``
   * help = Interval for binning of spectrum in log(muHz) (bins equally in logspace).
   * type = ``float``
   * default = `0.005`
   * unit = log(muHz)
- ``-bm``, ``--bm``, ``-mode``, ``--mode``, ``-bmode``, ``--bmode`` 
   * dest = ``args.mode``
   * help = Which mode to use when binning. Choices are ["mean", "median", "gaussian"]
   * type = ``str``
   * default = ``mean``
- ``-sw``, ``--sw``, ``-smoothwidth``, ``--smoothwidth``
   * dest = ``args.smooth_width``
   * help = Box filter width for smoothing the power spectrum
   * type = ``int``
   * default = `20`
- ``-step``, ``--step``, ``-steps``, ``--steps``
   * dest = ``args.step``
   * help = The step width for the collapsed ACF wrt the fraction of the boxsize
   * type = ``float``
   * default = `0.25`
- ``-trials``, ``--trials``, ``-ntrials``, ``--ntrials``
   * dest = ``args.n_trials``
   * help = Number of trials to estimate numax
   * type = int
   * default = `3`
- ``-lx``, ``--lx``, ``-lowerx``, ``--upperx``
   * dest = ``args.lower_ex``
   * help = Lower limit of power spectrum to use in findex module
   * nargs = '*'
   * type = ``float``
   * default = `10.0`
   * unit = muHz
- ``-ux``, ``--ux``, ``-upperx``, ``--upperx``
   * dest = ``args.upper_ex``
   * help = Upper limit of power spectrum to use in findex module
   * nargs = '*'
   * type = ``float``
   * default = `4000.0`
   * unit = muHz

Background Group
******************

- ``-bf``, ``--bf``, ``-box``, ``--box``, ``-boxfilter``, ``--boxfilter``
   * dest = ``args.box_filter``
   * help = Box filter width for plotting the power spectrum
   * type = ``float``
   * default = `1.0`
   * unit = muHz
- ``-dnu``, ``--dnu``
   * dest = ``args.dnu``
   * help = Brute force method to provide value for dnu
   * nargs = '*'
   * type = ``float``
   * default = ``None``
- ``-iw``, ``--iw``, ``-width``, ``--width``, ``-indwidth``, ``--indwidth``
   * dest = ``args.ind_width``
   * help = Width * resolution to use for binning of power spectrum in muHz (default=100*res)
   * type = ``float``
   * default = `100.0`
- ``-numax``, ``--numax``
   * dest = ``args.numax``
   * help = Brute force method to bypass findex and provide value for numax. Please note: len(args.numax) == len(args.targets) for this to work! This is mostly intended for single star runs.
   * nargs = '*'
   * type = ``float``
   * default = ``None``
- ``-lb``, ``--lb``, ``-lowerb``, ``--upperb``
   * dest = ``args.lower_bg``
   * help = Lower limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``-ub``, ``--ub``, ``-upperb``, ``--upperb``
   * dest = ``args.upper_bg``
   * help = Upper limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``-mc``, ``--mc``, ``-iter``, ``--iter``, ``-mciter``, ``--mciter``
   * dest = ``args.mc_iter``
   * help = Number of Monte-Carlo iterations
   * type = ``int``
   * default = `1`
- ``-peak``, ``--peak``, ``-peaks``, ``--peaks``, ``-npeaks``, ``--npeaks``
   * dest = ``args.n_peaks``
   * help = Number of peaks to fit in the ACF
   * type = ``int``
   * default = `5`
- ``-rms``, ``--rms``, ``-nrms``, ``--nrms``
   * dest = ``args.n_rms``
   * help = Number of points used to estimate amplitudes of individual background components (this should rarely need to be touched)
   * type = int
   * default = `20`
- ``-slope``, ``--slope`` 
   * dest = ``args.slope``
   * help = When true, this will correct for residual slope in a smoothed power spectrum before estimating numax
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-sp``, ``--sp``, ``-smoothps``, ``--smoothps``
   * dest = ``args.smooth_ps``
   * help = Box filter width for smoothing of the power spectrum. The default is 2.5, but will switch to 0.5 for more evolved stars (numax < 500 muHz).
   * type = ``float``
   * default = `2.5`
   * unit = muHz
- ``-samples``, ``--samples`` 
   * dest = ``args.samples``
   * help = Save samples from Monte-Carlo sampling
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-ce``, ``--ce``, ``-clipech``, ``--clipech`` 
   * dest = ``args.clip_ech``
   * help = Disable the automatic clipping of high peaks in the echelle diagram
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``-cv``, ``--cv``, ``-value``, ``--value``
   * dest = ``args.clip_value``
   * help = Clip value for echelle diagram (i.e. if ``args.clip_ech`` is ``True``). If none is provided, it will cut at 3x the median value of the folded power spectrum.
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``-ie``, ``--ie``, ``-interpech``, ``--interpech`` 
   * dest = ``args.interp_ech``
   * help = Turn on the bilinear interpolation for the echelle diagram
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-se``, ``--se``, ``-smoothech``, ``--smoothech``
   * dest = ``args.smooth_ech``
   * help = Option to smooth the echelle diagram output using a box filter
   * type = ``float``
   * default = ``None``
   * unit = muHz
   
.. _parallel:

Parallel Parser
===================

- ``-nt``, ``--nt``, ``-nthread``, ``--nthread``, ``-nthreads``, ``--nthreads`` 
   * dest = ``args.n_threads``
   * help = Number of processes to run in parallel. If nothing is provided, the software will use the ``multiprocessing`` package to determine the number of CPUs on the operating system and then adjust accordingly.
   * type = int
   * default = `0`
