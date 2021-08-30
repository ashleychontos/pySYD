.. _cli:

Command Line Interface
************************

In order to maximize the performance of the software, we have included many optional commands to help identify the
best possible asteroseismic parameters in even the lower signal cases. Examples of the optional features are shown 
in :ref:`advanced usage<advanced>`. 

The options are sorted both into :ref:`groups<parentparse>` by relevant science outputs 
and listed by :ref:`input type<inputtype>`. 

.. _parentparse:

Parent Parser
===============

Higher level functionality of the software. All four modes inherent the parent parser.

- ``--cli``, 
   * dest = ``args.cli``
   * help = This option should not be adjusted for current users
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
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

Accesses data and all science-related functions and is therefore relevant for the ``load``, ``run``, ``parallel`` and ``test`` modes.

- ``--bg``, ``--background``, ``-b``
   * dest = ``args.background``
   * help = Turn off the background fitting procedure
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``--globe``, ``--global``, ``-g``
   * dest = ``args.globe``
   * help = Do not estimate global asteroseismic parameters numax and dnu
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``--par``, ``--parallel``, ``-p``
   * dest = ``args.parallel``
   * help = Run pySYD in parallel mode
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--over``, ``--overwrite``, ``-o``
   * dest = ``args.overwrite``
   * help = Overwrite existing files with the same name/path
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
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
- ``--save``, ``-d``
   * dest = ``args.save``
   * help = save output files and figures
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``--show``, ``-s`` 
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
- ``--test``, ``-t``
   * dest = ``args.test``
   * help = Extra verbose output for testing functionality (not currently implemented)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--ex``, ``--excess``, ``-x``
   * dest = ``args.background``
   * help = turn off find excess module
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``


Estimating numax
++++++++++++++++++++

All options relevant for the first (optional) module to estimate numax:

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
   * default = `20.0`
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
   * type = ``float``
   * default = `1.0`
   * unit = muHz
- ``--ux``, ``--upperx``
   * dest = ``args.upper_ex``
   * help = Upper limit of power spectrum to use in findex module
   * type = ``float``
   * default = `6000.0`
   * unit = muHz


Background-related
+++++++++++++++++++++

All options relevant to the background-fitting process:

- ``--lb``, ``--lowerb``
   * dest = ``args.lower_bg``
   * help = Lower limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * type = ``float``
   * default = ``1.0``
   * unit = muHz
- ``--ub``,  ``--upperb``
   * dest = ``args.upper_bg``
   * help = Upper limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * type = ``float``
   * default = ``6000.0``
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
- ``--fix``, ``--fixwn``, ``--wn``, ``-f``
   * dest = ``args.fix``
   * help = Fix the white noise level
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--basis``
   * dest = ``args.basis``
   * help = Which basis to use for background fit (i.e. 'a_b', 'pgran_tau', 'tau_sigma'), *** NOT operational yet ***
   * type = str
   * default = `'tau_sigma'`
- ``--metric``
   * dest = ``args.metric``
   * help = Which model metric to use for the best-fit background model, choices~['bic','aic']
   * type = str
   * default = `'bic'`
- ``--include``, ``-i``
   * dest = ``args.include``
   * help = Include metric values in verbose output, default is `False`.
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``


   
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
- ``--method``
   * dest = ``args.method``
   * help = Method to use to determine dnu, choices ~['M', 'A', 'D']
   * type = ``str``
   * default = ``D``
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

All customizable options relevant for the echelle diagram output:

- ``--cv``, ``--value``
   * dest = ``args.clip_value``
   * help = Clip value for echelle diagram (i.e. if ``args.clip_ech`` is ``True``). If none is provided, it will cut at 3x the median value of the folded power spectrum.
   * type = ``float``
   * default = ``3.0``
   * unit = fractional psd
- ``--ce``, ``--cm``, ``--color``
   * dest = ``args.cmap``
   * help = Change colormap of ED, which is `binary` by default.
   * type = ``str``
   * default = ``binary``
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
- ``--notch``, ``-n``
   * dest = ``args.notching``
   * help = Use notching technique to reduce effects from mixed modes (not fully functional, creates weirds effects for higher SNR cases)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--hey``, ``-h``
   * dest = ``args.hey``
   * help = Plugin for Daniel Hey's echelle package (not currently implemented yet)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--ie``, ``--interpech``, ``-e``
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
- ``--nox``, ``--nacross``
   * dest = ``args.nox``
   * help = Resolution for the x-axis of the ED
   * type = ``int``
   * default = `50`
- ``--noy``, ``--ndown``, ``--norders``
   * dest = ``args.noy``
   * help = The number of orders to plot on the ED y-axis
   * type = ``int``
   * default = `0`
   
   
Sampling-related
++++++++++++++++++

All CLI options relevant for the Monte-Carlo sampling:

- ``--mc``, ``--iter``, ``--mciter``
   * dest = ``args.mc_iter``
   * help = Number of Monte-Carlo iterations
   * type = ``int``
   * default = `1`
- ``--samples``, ``-m``
   * dest = ``args.samples``
   * help = Save samples from Monte-Carlo sampling (i.e. parameter posteriors)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
  

Parallel Parser
================

Additional option for the number of threads to use when running stars in parallel.

- ``-nt``, ``--nt``, ``-nthread``, ``--nthread``, ``-nthreads``, ``--nthreads`` 
   * dest = ``args.n_threads``
   * help = Number of processes to run in parallel. If nothing is provided, the software will use the ``multiprocessing`` package to determine the number of CPUs on the operating system and then adjust accordingly.
   * type = int
   * default = `0`


===============================

.. _inputtype:


Input Type
=============

Booleans
++++++++++++++++++

Our boolean flags are sorted alphabetically by the single hash flag, where we have almost enough options
to span the entire English alphabet.

- ``-b``, ``--bg``, ``--background``, 
   * dest = ``args.background``
   * help = Turn off the background fitting procedure
   * default = ``True``
   * action = ``store_false``
- ``-c``, ``--cli``
   * dest = ``args.cli``
   * help = This option should not be adjusted for current users
   * default = ``True``
   * action = ``store_false``
- ``-d``, ``--save``
   * dest = ``args.save``
   * help = Save output files and figures (to disk)
   * default = ``True``
   * action = ``store_false``
- ``-e``, ``--ie``, ``--interpech``
   * dest = ``args.interp_ech``
   * help = Turn on the bilinear interpolation of the plotted echelle diagram
   * default = ``False``
   * action = ``store_true``
- ``-f``, ``--fix``, ``--fixwn``, ``--wn``
   * dest = ``args.fix``
   * help = Fix the white noise level
   * default = ``False``
   * action = ``store_true``
- ``-g``, ``--globe``, ``--global``, 
   * dest = ``args.globe``
   * help = Do not estimate global asteroseismic parameters numax and dnu
   * default = ``True``
   * action = ``store_false``
- ``-h``, ``--hey``
   * dest = ``args.hey``
   * help = Plugin for Daniel Hey's echelle package (not currently implemented yet)
   * default = ``False``
   * action = ``store_true``
- ``-i``, ``--include``
   * dest = ``args.include``
   * help = Include metric values in verbose output, default is `False`.
   * default = ``False``
   * action = ``store_true``
- ``-k``, ``--kc``, ``--kepcorr``
   * dest = ``args.kepcorr``
   * help = turn on the *Kepler* short-cadence artefact correction module
   * default = ``False``
   * action = ``store_true``
- ``-m``, ``--samples``
   * dest = ``args.samples``
   * help = Save samples from Monte-Carlo sampling (i.e. parameter posteriors)
   * default = ``False``
   * action = ``store_true``
- ``-n``, ``--notch``
   * dest = ``args.notching``
   * help = Use notching technique to reduce effects from mixed modes (not fully functional, creates weirds effects for higher SNR cases)
   * default = ``False``
   * action = ``store_true``
- ``-o``, ``--over``, ``--overwrite``
   * dest = ``args.overwrite``
   * help = Overwrite existing files with the same name/path
   * default = ``False``
   * action = ``store_true``
- ``-p``, ``--par``, ``--parallel``
   * dest = ``args.parallel``
   * help = Run pySYD in parallel mode
   * default = ``False``
   * action = ``store_true``
- ``-s`` , ``--show``
   * dest = ``args.show``
   * help = show output figures (note: this is not recommended if running many stars)
   * default = ``False``
   * action = ``store_true``
- ``-t``, ``--test``
   * dest = ``args.test``
   * help = Extra verbose output for testing functionality (not currently implemented)
   * default = ``False``
   * action = ``store_true``
- ``-v``, ``--verbose``
   * dest = ``args.verbose``
   * help = Turn on verbose output
   * default = ``False``
   * action = ``store_true``
- ``-x``, ``--ex``, ``--excess``
   * dest = ``args.background``
   * help = turn off find excess module
   * default = ``True``
   * action = ``store_false``


Integers
++++++++++++++++++

- ``--laws``, ``--nlaws``
   * dest = ``args.n_laws``
   * help = Force the number of red-noise component(s)
   * default = `None`
- ``--mc``, ``--iter``, ``--mciter``
   * dest = ``args.mc_iter``
   * help = Number of Monte-Carlo iterations
   * default = `1`
- ``--nox``, ``--nacross``
   * dest = ``args.nox``
   * help = Resolution for the x-axis of the ED
   * default = `50`
- ``--noy``, ``--ndown``, ``--norders``
   * dest = ``args.noy``
   * help = The number of orders to plot on the ED y-axis
   * default = `0`
- ``-nt``, ``--nt``, ``-nthread``, ``--nthread``, ``-nthreads``, ``--nthreads`` 
   * dest = ``args.n_threads``
   * help = Number of processes to run in parallel. If nothing is provided, the software will use the ``multiprocessing`` package to determine the number of CPUs on the operating system and then adjust accordingly.
   * default = `0`
- ``--ofa``, ``--of_actual``
   * dest = ``args.of_actual``
   * help = The oversampling factor of the provided power spectrum. Default is `0`, which means it is calculated from the time series data. Note: This needs to be provided if there is no time series data!
   * default = `0`
- ``--ofn``, ``--of_new``
   * dest = ``args.of_new``
   * help = The new oversampling factor to use in the first iterations of both modules. Default is `5` (see performance for more details).
   * default = `5`
- ``--peak``, ``--peaks``, ``--npeaks``
   * dest = ``args.n_peaks``
   * help = Number of peaks to fit in the ACF
   * default = `5`
- ``--rms``, ``--nrms``
   * dest = ``args.n_rms``
   * help = Number of points used to estimate amplitudes of individual background components (this should rarely need to be touched)
   * default = `20`
- ``--trials``, ``--ntrials``
   * dest = ``args.n_trials``
   * help = Number of trials to estimate numax
   * default = `3`


Floats
++++++++++++++++++


- ``--bf``, ``--box``, ``--boxfilter``
   * dest = ``args.box_filter``
   * help = Box filter width (in muHz) for plotting the power spectrum
   * default = `1.0`
   * unit = muHz
- ``--bin``, ``--binning``
   * dest = ``args.binning``
   * help = Interval for binning of spectrum in log(muHz) (bins equally in logspace).
   * default = `0.005`
   * unit = log(muHz)
- ``--cv``, ``--value``
   * dest = ``args.clip_value``
   * help = Clip value for echelle diagram (i.e. if ``args.clip_ech`` is ``True``). If none is provided, it will cut at 3x the median value of the folded power spectrum.
   * default = ``3.0``
   * unit = fractional psd
- ``--dnu``
   * dest = ``args.dnu``
   * help = Brute force method to provide value for dnu
   * nargs = '*'
   * default = ``None``
- ``--ew``, ``--exwidth``
   * dest = ``args.width``
   * help = Fractional value of width to use for power excess, where width is computed using a solar scaling relation and then centered on the estimated numax.
   * default = `1.0`
- ``--iw``, ``--indwidth``
   * dest = ``args.ind_width``
   * help = Width of binning for power spectrum (in muHz)
   * default = `20.0`
- ``--lb``, ``--lowerb``
   * dest = ``args.lower_bg``
   * help = Lower limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * default = ``1.0``
   * unit = muHz
- ``--le``, ``--uppere``
   * dest = ``args.lower_ech``
   * help = Lower frequency limit of the folded PS to whiten mixed modes before determining the correct dnu
   * nargs = '*'
   * default = ``None``
   * unit = muHz
- ``--lp``, ``--lowerp``
   * dest = ``args.lower_ps``
   * help = Lower frequency limit for zoomed in power spectrum (around power excess)
   * nargs = '*'
   * default = ``None``
   * unit = muHz
- ``--lx``, ``--lowerx``
   * dest = ``args.lower_ex``
   * help = Lower limit of power spectrum to use in findex module
   * default = `1.0`
   * unit = muHz
- ``--numax``
   * dest = ``args.numax``
   * help = Brute force method to bypass findex and provide value for numax. Please note: len(args.numax) == len(args.targets) for this to work! This is mostly intended for single star runs.
   * nargs = '*'
   * default = ``None``
- ``--se``, ``--smoothech``
   * dest = ``args.smooth_ech``
   * help = Option to smooth the echelle diagram output using a box filter
   * default = ``None``
   * unit = muHz
- ``--sm``, ``--smpar``
   * dest = ``args.sm_par``
   * help = Value of smoothing parameter to estimate the smoothed numax (typical values range from `1`-`4`)
   * default = `None`
- ``--sp``, ``--smoothps``
   * dest = ``args.smooth_ps``
   * help = Box filter width for smoothing of the power spectrum. The default is 2.5, but will switch to 0.5 for more evolved stars (numax < 500 muHz).
   * default = `2.5`
   * unit = muHz
- ``--step``, ``--steps``
   * dest = ``args.step``
   * help = The step width for the collapsed ACF wrt the fraction of the boxsize
   * default = `0.25`
- ``--sw``, ``--smoothwidth``
   * dest = ``args.smooth_width``
   * help = Box filter width (in muHz) for smoothing the power spectrum
   * default = `20.0`
- ``--thresh``, ``--threshold``
   * dest = ``args.threshold``
   * help = Fractional value of the ACF FWHM to use for determining dnu
   * default = ``1.0``
- ``--ub``,  ``--upperb``
   * dest = ``args.upper_bg``
   * help = Upper limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * default = ``6000.0``
   * unit = muHz
- ``--ue``,  ``--uppere``
   * dest = ``args.upper_ech``
   * help = Upper frequency limit of the folded PS to whiten mixed modes before determining the correct dnu
   * nargs = '*'
   * default = ``None``
   * unit = muHz
- ``--up``,  ``--upperp``
   * dest = ``args.upper_ps``
   * help = Upper frequency limit for zoomed in power spectrum (around power excess)
   * nargs = '*'
   * default = ``None``
   * unit = muHz
- ``--ux``, ``--upperx``
   * dest = ``args.upper_ex``
   * help = Upper limit of power spectrum to use in findex module
   * default = `6000.0`
   * unit = muHz



Strings
++++++++++++++++++


- ``--basis``
   * dest = ``args.basis``
   * help = Which basis to use for background fit (i.e. 'a_b', 'pgran_tau', 'tau_sigma'), *** NOT operational yet ***
   * default = `'tau_sigma'`
- ``--bm``, ``--mode``, ``--bmode`` 
   * dest = ``args.mode``
   * help = Which mode to use when binning. Choices are ["mean", "median", "gaussian"]
   * default = ``mean``
- ``--ce``, ``--cm``, ``--color``
   * dest = ``args.cmap``
   * help = Change colormap of ED, which is `binary` by default.
   * default = ``binary``
- ``--file``, ``--list``, ``--todo``
   * dest = ``args.file``
   * help = Path to text file that contains the list of stars to process (convenient for running many stars).
   * default = ``TODODIR``
- ``--in``, ``--input``, ``--inpdir``
   * dest = ``args.inpdir``
   * help = Path to input data
   * default = ``INPDIR``
- ``--info``, ``--information`` 
   * dest = ``args.info``
   * help = Path to the csv containing star information (although not required).
   * default = ``INFODIR``
- ``--method``
   * dest = ``args.method``
   * help = Method to use to determine dnu, choices ~['M', 'A', 'D']
   * default = ``D``
- ``--metric``
   * dest = ``args.metric``
   * help = Which model metric to use for the best-fit background model, choices~['bic','aic']
   * default = `'bic'`
- ``--out``, ``--output``, ``--outdir``
   * dest = ``args.outdir``
   * help = Path that results are saved to
   * default = ``OUTDIR``
- ``--star``, ``--stars``
   * dest = ``args.star``
   * help = List of stars to process. Default is ``None``, which will read in the star list from ``args.file``.
   * nargs = '*'
   * default = ``None``


===============================


Below is a quick, crash course about accessing ``pySYD`` via command line.

.. raw:: html

   <iframe width="800" height="450" src="https://www.youtube.com/embed/c1do_BKtHXk" 
   title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; 
   clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


===============================
