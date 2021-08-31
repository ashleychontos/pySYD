.. _cli:

############################
CLI
############################


In order to maximize the performance of the software, we have included many optional commands to help identify the
best possible asteroseismic parameters in even the lowest signal cases. 


Help
=======

Running the ``pySYD`` help command for the main pipeline execution will display:

.. code-block::

    $ pysyd run --help                     
    usage: pySYD run [-h] [-c] [--file path] [--in path] [--info path]
                     [--out path] [-v] [-b] [-d] [-g] [-k] [--ofa n] [--ofn n]
                     [-o] [-p] [-s] [--star [star [star ...]]] [-t] [-x] [-a]
                     [--bin value] [--bm mode] [--lx freq] [--step value]
                     [--trials n] [--sw value] [--ux freq] [--basis str]
                     [--bf value] [-f] [-i] [--iw value] [--laws n] [--lb freq]
                     [--metric metric] [--rms n] [--ub freq] [--ew value]
                     [--lp [freq [freq ...]]] [--numax [value [value ...]]]
                     [--sm value] [--up [freq [freq ...]]]
                     [--dnu [value [value ...]]] [--method method] [--peak n]
                     [--sp value] [--thresh value] [--ce cmap] [--cv value] [-y]
                     [-e] [--le [freq [freq ...]]] [--notch] [--nox n] [--noy n]
                     [--se value] [--ue [freq [freq ...]]] [--mc n] [-m]

    optional arguments:
      -h, --help            show this help message and exit
      -c, --cli             This option should not be adjusted for current users
      --file path, --list path, --todo path
                            List of stars to process
      --in path, --input path, --inpdir path
                            Input directory
      --info path, --information path
                            Path to star info
      --out path, --outdir path, --output path
                            Output directory
      -v, --verbose         Turn on verbose output
      -b, --bg, --background
                            Turn off the automated background fitting routine
      -d, --show, --display
                            Show output figures
      -g, --globe, --global
                            Do not estimate global asteroseismic parameters (i.e.
                            numax or dnu)
      -k, --kc, --kep_corr  Turn on the Kepler short-cadence artefact correction
                            routine
      --ofa n, --of_actual n
                            The oversampling factor (OF) of the input PS
      --ofn n, --of_new n   The OF to be used for the first iteration
      -o, --over, --overwrite
                            Overwrite existing files with the same name/path
      -p, --par, --parallel
                            Use parallel processing for data analysis
      -s, --save            Do not save output figures and results.
      --star [star [star ...]], --stars [star [star ...]]
                            List of stars to process
      -t, --test            Extra verbose output for testing functionality
      -x, --ex, --excess    Turn off the find excess routine

    Estimate numax:
      -a, --ask             Ask which trial to use
      --bin value, --binning value
                            Binning interval for PS (in muHz)
      --bm mode, --mode mode, --bmode mode
                            Binning mode
      --lx freq, --lowerx freq
                            Lower frequency limit of PS
      --step value, --steps value
      --trials n, --ntrials n
      --sw value, --smoothwidth value
                            Box filter width (in muHz) for smoothing the PS
      --ux freq, --upperx freq
                            Upper frequency limit of PS

    Background Fit:
      --basis str           Which basis to use for background fit (i.e. 'a_b',
                            'pgran_tau', 'tau_sigma'), *** NOT operational yet ***
      --bf value, --box value, --boxfilter value
                            Box filter width [in muHz] for plotting the PS
      -f, --fix, --fixwn    Fix the white noise level
      -i, --include         Include metric values in verbose output, default is
                            `False`.
      --iw value, --indwidth value
                            Width of binning for PS [in muHz]
      --laws n, --nlaws n   Force number of red-noise component(s)
      --lb freq, --lowerb freq
                            Lower frequency limit of PS
      --metric metric       Which model metric to use, choices=['bic','aic']
      --rms n, --nrms n     Number of points to estimate the amplitude of red-
                            noise component(s)
      --ub freq, --upperb freq
                            Upper frequency limit of PS

    Deriving numax:
      --ew value, --exwidth value
                            Fractional value of width to use for power excess,
                            where width is computed using a solar scaling
                            relation.
      --lp [freq [freq ...]], --lowerp [freq [freq ...]]
                            Lower frequency limit for zoomed in PS
      --numax [value [value ...]]
                            Skip find excess module and force numax
      --sm value, --smpar value
                            Value of smoothing parameter to estimate smoothed
                            numax (typically between 1-4).
      --up [freq [freq ...]], --upperp [freq [freq ...]]
                            Upper frequency limit for zoomed in PS

    Deriving dnu:
      --dnu [value [value ...]]
                            Brute force method to provide value for dnu
      --method method       Method to use to determine dnu, ~[M, A, D]
      --peak n, --peaks n, --npeaks n
                            Number of peaks to fit in the ACF
      --sp value, --smoothps value
                            Box filter width [in muHz] of PS for ACF
      --thresh value, --threshold value
                            Fractional value of FWHM to use for ACF

    Echelle diagram:
      --ce cmap, --cm cmap, --color cmap
                            Change colormap of ED, which is `binary` by default.
      --cv value, --value value
                            Clip value multiplier to use for echelle diagram (ED).
                            Default is 3x the median, where clip_value == `3`.
      -y, --hey             Use Daniel Hey's plugin for echelle
      -e, --ie, -interpech, --interpech
                            Turn on the interpolation of the output ED
      --le [freq [freq ...]], --lowere [freq [freq ...]]
                            Lower frequency limit of folded PS to whiten mixed
                            modes
      --notch               Use notching technique to reduce effects from mixed
                            modes (not fully functional, creates weirds effects
                            for higher SNR cases)
      --nox n, --nacross n  Resolution for the x-axis of the ED
      --noy n, --ndown n, --norders n
                            The number of orders to plot on the ED y-axis
      --se value, --smoothech value
                            Smooth ED using a box filter [in muHz]
      --ue [freq [freq ...]], --uppere [freq [freq ...]]
                            Upper frequency limit of folded PS to whiten mixed
                            modes

    Sampling:
      --mc n, --iter n, --mciter n
                            Number of Monte-Carlo iterations
      -m, --samples         Save samples from the Monte-Carlo sampling


which shows a very long but very healthy list of available options. We tried to make this
easier on the eyes by separating the commands into related groups, but do not fret! We realize
this is a lot of information, which is why we have dedicated an entire page to describing these
features.

Additionally, we have examples of some put to use in :ref:`advanced usage<advanced>` 
and also have included a brief :ref:`tutorial` below that describes some of these commands.



Features
===========

Due to the large number of options, we have them sorted into :ref:`groups<groups>` by relevant science outputs 
and also listed by :ref:`input type<inputtype>`. 


.. note::

    Our features were developed using principles from Unix-like operating systems, 
    where a single hyphen can be followed by multiple single-character flags (i.e.
    mostly boolean flags that do not require additional output). 
    
    An example is ``-dvoi``, which is far more convenient than writing ``--display --verbose 
    --overwrite --include``. Together, these commands tell ``pySYD`` to:
     1. Display the output figures (``-d``, ``--show``, ``--display``),
     2. Turn on the verbose output (``-v``, ``--verbose``),
     3. Overwrite existing files with the same name (``-o``, ``--overwrite``), and
     4. Include the model metrics and values with the verbose output (``-i``, ``--include``).
     

.. _groups:

Groups
********

High-level functionality
```````````````````````````

All ``pySYD`` modes inherent the parent parser, which includes the properties 
enumerated below. With the exception of the ``verbose`` command, most of these
features are related to the initial (setup) paths and directories and should be
used very sparingly. 

- ``--cli``, ``-c``
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


Data analyses
```````````````````````````

The following features are primarily related to the initial and final treatment of
data products, including information about the input data, how to process and save
the data as well as which modules to use.

- ``-b``, ``--bg``, ``--background`` 
   * dest = ``args.background``
   * help = Turn off the background fitting procedure and run ``pySYD`` on raw power spectrum
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``-d``, ``--show``, ``--display``
   * dest = ``args.show``
   * help = show output figures (note: this is not recommended if running many stars)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-g``, ``--globe``, ``--global``
   * dest = ``args.globe``
   * help = Do not estimate global asteroseismic parameters numax and dnu
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``-p``, ``--par``, ``--parallel``
   * dest = ``args.parallel``
   * help = Run pySYD in parallel mode
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-o``, ``--over``, ``--overwrite``
   * dest = ``args.overwrite``
   * help = Overwrite existing files with the same name/path
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-k``, ``--kc``, ``--kepcorr``
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
- ``-s``, ``--save``
   * dest = ``args.save``
   * help = Save output files and figures to disk
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``
- ``--star``, ``--stars``
   * dest = ``args.star``
   * help = List of stars to process. Default is ``None``, which will read in the star list from ``args.file``.
   * nargs = '*'
   * default = ``None``
- ``-t``, ``--test``
   * dest = ``args.test``
   * help = Extra verbose output for testing functionality (not currently implemented)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-x``, ``--ex``, ``--excess``
   * dest = ``args.background``
   * help = turn off find excess module
   * type = ``bool``
   * default = ``True``
   * action = ``store_false``


Estimating numax
```````````````````````````

The following options are relevant for the first, optional module that is designed
to estimate numax if it is not known: 

- ``-a``, ``--ask``
   * dest = ``args.ask``
   * help = Ask which trial (or estimate) to use
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
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


Granulation background
```````````````````````````

Below is a complete list of parameters relevant to the background-fitting routine:

- ``--basis``
   * dest = ``args.basis``
   * help = Which basis to use for background fit (i.e. 'a_b', 'pgran_tau', 'tau_sigma'), *** NOT operational yet ***
   * type = str
   * default = `'tau_sigma'`
- ``--bf``, ``--box``, ``--boxfilter``
   * dest = ``args.box_filter``
   * help = Box filter width (in muHz) for plotting the power spectrum
   * type = ``float``
   * default = `1.0`
   * unit = muHz
- ``-f``, ``--fix``, ``--fixwn``
   * dest = ``args.fix``
   * help = Fix the white noise level
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``-i``, ``--include``
   * dest = ``args.include``
   * help = Include metric values in verbose output, default is `False`.
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--iw``, ``--indwidth``
   * dest = ``args.ind_width``
   * help = Width of binning for power spectrum (in muHz)
   * type = ``float``
   * default = `20.0`
- ``--laws``, ``--nlaws``
   * dest = ``args.n_laws``
   * help = Force the number of red-noise component(s)
   * type = int
   * default = `None`
- ``--lb``, ``--lowerb``
   * dest = ``args.lower_bg``
   * help = Lower limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * type = ``float``
   * default = ``1.0``
   * unit = muHz
- ``--metric``
   * dest = ``args.metric``
   * help = Which model metric to use for the best-fit background model, choices~['bic','aic']
   * type = str
   * default = `'bic'`
- ``--rms``, ``--nrms``
   * dest = ``args.n_rms``
   * help = Number of points used to estimate amplitudes of individual background components (this should rarely need to be touched)
   * type = int
   * default = `20`
- ``--ub``,  ``--upperb``
   * dest = ``args.upper_bg``
   * help = Upper limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * type = ``float``
   * default = ``6000.0``
   * unit = muHz


   
Deriving numax
```````````````````````````

All of the following parameters are related to deriving numax, or the frequency
corresponding to maximum power:

- ``--ew``, ``--exwidth``
   * dest = ``args.width``
   * help = Fractional value of width to use for power excess, where width is computed using a solar scaling relation and then centered on the estimated numax.
   * type = ``float``
   * default = `1.0`
- ``--lp``, ``--lowerp``
   * dest = ``args.lower_ps``
   * help = Lower frequency limit for zoomed in power spectrum (around power excess)
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``--numax``
   * dest = ``args.numax``
   * help = Brute force method to bypass findex and provide value for numax. Please note: len(args.numax) == len(args.targets) for this to work! This is mostly intended for single star runs.
   * nargs = '*'
   * type = ``float``
   * default = ``None``
- ``--sm``, ``--smpar``
   * dest = ``args.sm_par``
   * help = Value of smoothing parameter to estimate the smoothed numax (typical values range from `1`-`4`)
   * type = ``float``
   * default = `None`
- ``--up``,  ``--upperp``
   * dest = ``args.upper_ps``
   * help = Upper frequency limit for zoomed in power spectrum (around power excess)
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz


Deriving dnu
```````````````````````````

Below are all options related to the characteristic frequency spacing (dnu):

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
- ``--peak``, ``--peaks``, ``--npeaks``
   * dest = ``args.n_peaks``
   * help = Number of peaks to fit in the ACF
   * type = ``int``
   * default = `5`
- ``--sp``, ``--smoothps``
   * dest = ``args.smooth_ps``
   * help = Box filter width for smoothing of the power spectrum. The default is 2.5, but will switch to 0.5 for more evolved stars (numax < 500 muHz).
   * type = ``float``
   * default = `2.5`
   * unit = muHz
- ``--thresh``, ``--threshold``
   * dest = ``args.threshold``
   * help = Fractional value of the ACF FWHM to use for determining dnu
   * type = ``float``
   * default = ``1.0``
   

Echelle diagram
```````````````````````````

All customizable options relevant for the echelle diagram output:


- ``--ce``, ``--cm``, ``--color``
   * dest = ``args.cmap``
   * help = Change colormap of ED, which is `binary` by default.
   * type = ``str``
   * default = ``binary``
- ``--cv``, ``--value``
   * dest = ``args.clip_value``
   * help = Clip value for echelle diagram (i.e. if ``args.clip_ech`` is ``True``). If none is provided, it will cut at 3x the median value of the folded power spectrum.
   * type = ``float``
   * default = ``3.0``
   * unit = fractional psd
- ``-e``, ``--ie``, ``--interpech``
   * dest = ``args.interp_ech``
   * help = Turn on the bilinear interpolation for the echelle diagram
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
- ``--le``, ``--uppere``
   * dest = ``args.lower_ech``
   * help = Lower frequency limit of the folded PS to whiten mixed modes before determining the correct dnu
   * nargs = '*'
   * type = ``float``
   * default = ``None``
   * unit = muHz
- ``-n``, ``--notch``
   * dest = ``args.notching``
   * help = Use notching technique to reduce effects from mixed modes (not fully functional, creates weirds effects for higher SNR cases)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
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
- ``--se``, ``--smoothech``
   * dest = ``args.smooth_ech``
   * help = Option to smooth the echelle diagram output using a box filter
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
- ``-y``, ``--hey``
   * dest = ``args.hey``
   * help = Plugin for Daniel Hey's echelle package (not currently implemented yet)
   * type = ``bool``
   * default = ``False``
   * action = ``store_true``
   
   
Estimating uncertainties
```````````````````````````

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
  

Parallel processing
```````````````````````````

Additional option for the number of threads to use when running stars in parallel.

- ``--nt``, ``--nthread``, ``--nthreads`` 
   * dest = ``args.n_threads``
   * help = Number of processes to run in parallel. If nothing is provided, the software will use the ``multiprocessing`` package to determine the number of CPUs on the operating system and then adjust accordingly.
   * type = int
   * default = `0`
   
   
.. warning::

    All parameters are optimized for most star types but some may need adjusting. 
    An example is the smoothing width (``--sw``), which is 20 muHz by default, but 
    may need to be adjusted based on the nyquist frequency and frequency resolution 
    of the input power spectrum.


.. _inputtype:


Types
*********

Boolean
````````````

Our boolean flags are sorted alphabetically by the single hash flag (and span almost the
entire English alphabet):

- ``-a``, ``--ask``
   * dest = ``args.ask``
   * help = Ask which trial (or estimate) of numax to use
   * default = ``False``
   * action = ``store_true``
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
- ``-y``, ``--hey``
   * dest = ``args.hey``
   * help = Plugin for Daniel Hey's echelle package (not currently implemented yet)
   * default = ``False``
   * action = ``store_true``


Integer
````````````

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


Float
````````````


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



String
````````````


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


.. _tutorial::


Tutorial
===========


Below is a quick, crash course demonstrating the easy accessibility of
``pySYD`` via command line.

.. raw:: html

   <iframe width="680" height="382.5" src="https://www.youtube.com/embed/c1do_BKtHXk" 
   title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; 
   clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


===============================
