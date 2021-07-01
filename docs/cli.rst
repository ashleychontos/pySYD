.. _cli:

Command Line Interface
************************

.. _parentparse:


Parent Parser
===============

Higher level functionality of the software. All four modes inherent the parent parser.

- ``--file``, ``--list``, ``--todo``
   * dest = ``args.file``
   * help = Path to text file that contains the list of stars to process (convenient for running many stars).
   * type = ``str``
   * default = ``TODODIR``
- ``--in``, ``--input``, ``--inpdir``
   * dest = ``args.inpdir``
   * help = Path to input data
   * type = ``str``
   * default = ``INPDIR``
- ``--info``, ``--information`` 
   * dest = ``args.info``
   * help = Path to the csv containing star information (although not required).
   * type = ``str``
   * default = ``INFODIR``
- ``--out``, ``--output``, ``--outdir``
   * dest = ``args.outdir``
   * help = Path that results are saved to
   * type = ``str``
   * default = ``OUTDIR``
- ``--verbose``, ``-v``
   * dest = ``args.verbose``
   * help = Turn on verbose output
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``


.. _mainparse:


Main Parser
=============

Accesses dataa and all science-related functions and is therefore relevant for the ``load``, ``run`` and ``parallel`` modes.

-  ``--bg``, ``--fitbg``, ``--background``, ``-b``
   * dest = ``args.background``
   * help = Turn off the background fitting procedure
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``--ex``, ``--findex``, ``--excess``, ``-x``
   * dest = ``args.background``
   * help = turn off find excess module
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``--kc``, ``--kepcorr``, ``-k``
   * dest = ``args.kepcorr``
   * help = turn on the *Kepler* short-cadence artefact correction module
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--ofa``, ``--of_actual``
   * dest = ``args.of_actual``
   * help = The oversampling factor of the provided power spectrum. Default is `0`, which means it is calculated from the time series data. Note: This needs to be provided if there is no time series data!
   * type = ``int``
   * default = `0`
- ``--ofn``, ``--of_new``
   * dest = ``args.of_new``
   * help = The new oversampling factor to use in the first iterations of both modules. Default is `5` (see performance for more details).
   * type = int
   * default = `5`
- ``--save``
   * dest = ``args.save``
   * help = save output files and figures
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``--show``, ``s`` 
   * dest = ``args.show``
   * help = show output figures (note: this is not recommended if running many stars)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--star``, ``--stars``
   * dest = ``args.star``
   * help = List of stars to process. Default is ``None``, which will read in the star list from ``args.file``.
   * nargs = '*'
   * default = ``None``


Excess Group
++++++++++++++

All CLI options relevant to the first (find excess) module:

- ``--bin``, ``--binning``
   * dest = ``args.binning``
   * help = Interval for binning of spectrum in log(muHz) (bins equally in logspace).
   * type = ``float``
   * default = `0.005`
   * unit = log(muHz)
- ``--bm``, ``--mode``, ``--bmode`` 
   * dest = ``args.mode``
   * help = Which mode to use when binning. Choices are ["mean", "median", "gaussian"]
   * type = ``str``
   * default = ``mean``
- ``--sw``, ``--smoothwidth``
   * dest = ``args.smooth_width``
   * help = Box filter width (in muHz) for smoothing the power spectrum
   * type = ``float``
   * default = `50.0`
- ``--step``, ``--steps``
   * dest = ``args.step``
   * help = The step width for the collapsed ACF wrt the fraction of the boxsize
   * type = ``float``
   * default = `0.25`
- ``--trials``, ``--ntrials``
   * dest = ``args.n_trials``
   * help = Number of trials to estimate numax
   * type = int
   * default = `3`
- ``--lx``, ``--lowerx``
   * dest = ``args.lower_ex``
   * help = Lower limit of power spectrum to use in findex module
   * nargs = '*'
   * type = ``float``
   * default = `10.0`
   * unit = muHz
- ``--ux``, ``--upperx``
   * dest = ``args.upper_ex``
   * help = Upper limit of power spectrum to use in findex module
   * nargs = '*'
   * type = ``float``
   * default = `4000.0`
   * unit = muHz


Background-related
+++++++++++++++++++++

All CLI options relevant to the background-fitting process:

- ``--lb``, ``--upperb``
   * dest = ``args.lower_bg``
   * help = Lower limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``--ub``,  ``--upperb``
   * dest = ``args.upper_bg``
   * help = Upper limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``--iw``, ``--indwidth``
   * dest = ``args.ind_width``
   * help = Width of binning for power spectrum (in muHz)
   * type = ``float``
   * default = `20.0`
- ``--bf``, ``--box``, ``--boxfilter``
   * dest = ``args.box_filter``
   * help = Box filter width (in muHz) for plotting the power spectrum
   * type = ``float``
   * default = `1.0`
   * unit = muHz
- ``--rms``, ``--nrms``
   * dest = ``args.n_rms``
   * help = Number of points used to estimate amplitudes of individual background components (this should rarely need to be touched)
   * type = int
   * default = `20`
- ``--laws``, ``--nlaws``
   * dest = ``args.n_laws``
   * help = Force the number of red-noise component(s)
   * type = int
   * default = `None`
- ``--use``
   * dest = ``args.use``
   * help = Which model metric to use for the best-fit background model, choices~['bic','aic']
   * type = str
   * default = `'bic'`

   
Numax-related
++++++++++++++++++

All CLI options relevant to estimating numax:

- ``--sm``, ``--smpar``
   * dest = ``args.sm_par``
   * help = Value of smoothing parameter to estimate the smoothed numax (typical values range from `1`-`4`)
   * type = ``float``
   * default = `None`
- ``--numax``
   * dest = ``args.numax``
   * help = Brute force method to bypass findex and provide value for numax. Please note: len(args.numax) == len(args.targets) for this to work! This is mostly intended for single star runs.
   * nargs = '*'
   * type = ``float``
   * default = ``None``
- ``--lp``, ``--lowerp``
   * dest = ``args.lower_ps``
   * help = Lower frequency limit for zoomed in power spectrum (around power excess)
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``--up``,  ``--upperp``
   * dest = ``args.upper_ps``
   * help = Upper frequency limit for zoomed in power spectrum (around power excess)
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``--ew``, ``--exwidth``
   * dest = ``args.width``
   * help = Fractional value of width to use for power excess, where width is computed using a solar scaling relation and then centered on the estimated numax.
   * type = ``float``
   * default = `1.0`


Dnu-related
++++++++++++++++++

All CLI options relevant to determining dnu:

- ``--dnu``
   * dest = ``args.dnu``
   * help = Brute force method to provide value for dnu
   * nargs = '*'
   * type = ``float``
   * default = ``None``
- ``--sp``, ``--smoothps``
   * dest = ``args.smooth_ps``
   * help = Box filter width for smoothing of the power spectrum. The default is 2.5, but will switch to 0.5 for more evolved stars (numax < 500 muHz).
   * type = ``float``
   * default = `2.5`
   * unit = muHz
- ``--peak``, ``--peaks``, ``--npeaks``
   * dest = ``args.n_peaks``
   * help = Number of peaks to fit in the ACF
   * type = ``int``
   * default = `5`
- ``--thresh``, ``--threshold``
   * dest = ``args.threshold``
   * help = Fractional value of the ACF FWHM to use for determining dnu
   * type = ``float``
   * default = ``1.0``
   
Echelle-related
++++++++++++++++++

All CLI options relevant for the echelle diagram output:

- ``--ce``, ``--clipech`` 
   * dest = ``args.clip_ech``
   * help = Disable the automatic clipping of high peaks in the echelle diagram
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``--cv``, ``--value``
   * dest = ``args.clip_value``
   * help = Clip value for echelle diagram (i.e. if ``args.clip_ech`` is ``True``). If none is provided, it will cut at 3x the median value of the folded power spectrum.
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``--le``, ``--uppere``
   * dest = ``args.lower_ech``
   * help = Lower frequency limit of the folded PS to whiten mixed modes before determining the correct dnu
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``--ue``,  ``--uppere``
   * dest = ``args.upper_ech``
   * help = Upper frequency limit of the folded PS to whiten mixed modes before determining the correct dnu
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``--hey``
   * dest = ``args.hey``
   * help = Plugin for Daniel Hey's echelle package (not currently implemented yet)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--ie``, ``--interpech`` 
   * dest = ``args.interp_ech``
   * help = Turn on the bilinear interpolation for the echelle diagram
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--se``, ``--smoothech``
   * dest = ``args.smooth_ech``
   * help = Option to smooth the echelle diagram output using a box filter
   * type = ``float``
   * default = ``None``
   * unit = muHz
   
   
Sampling-related
++++++++++++++++++

All CLI options relevant for the Monte-Carlo sampling:

- ``--mc``, ``--iter``, ``--mciter``
   * dest = ``args.mc_iter``
   * help = Number of Monte-Carlo iterations
   * type = ``int``
   * default = `1`
- ``--samples`` 
   * dest = ``args.samples``
   * help = Save samples from Monte-Carlo sampling
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
   
.. _parallel:

Parallel Parser
================

Additional option for the number of threads to use when running stars in parallel.

- ``-nt``, ``--nt``, ``-nthread``, ``--nthread``, ``-nthreads``, ``--nthreads`` 
   * dest = ``args.n_threads``
   * help = Number of processes to run in parallel. If nothing is provided, the software will use the ``multiprocessing`` package to determine the number of CPUs on the operating system and then adjust accordingly.
   * type = int
   * default = `0`
