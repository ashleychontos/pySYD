.. _cli/index:

**********************
As a command-line tool
**********************

.. note::

    ``pySYD`` was developed s.t. it would be primarily used as a command-line tool. However, we 
    acknowledge that this is not convenient for everyone and have therefore provided some tutorials 
    on how to use it in a Jupyter notebook.

.. _cli/help:

Try clicking this :term:`a`

Try clicking this :term:`save`

Try clicking this :term:`bg`

Navigating 
###########

From a terminal window, running the ``pySYD`` help command for the main pipeline execution (i.e. ``pysyd.run``)

.. code-block::

    pysyd run --help

will display an enormous list of options that we have broken up into relevant groups to make 
it easier to digest. There is an overwhelming amount but they are only there to make your
experience with asteroseismology as customizable as possible.

**Jump to:**
 - :ref:`usage + optional arguments <cli/help/usage>`
 - :ref:`estimating numax <cli/help/est>`
 - :ref:`background fit <cli/help/bg>`
 - deriving :ref:`numax <cli/help/numax>` and :ref:`dnu <cli/help/dnu>`
 - :ref:`echelle diagram <cli/help/ech>`
 - :ref:`sampling <cli/help/mc>`
 
-----

.. _cli/help/usage:

Usage & optional arguments
**************************

Below is the first part of the output, which is primarily related to the higher level functionality.
Within the software, these are defined by the parent and main parsers, which are inevitably inherited
by all ``pySYD`` modes that handle the data.

.. code-block::
                   
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

.. _cli/help/est:

Estimating :math:`\nu_{\mathrm{max}}`
*************************************

.. code-block::

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

.. _cli/help/bg:

Background fit
**************

.. code-block::

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

.. _cli/help/numax:

Deriving :math:`\nu_{\mathrm{max}}`
***********************************

.. code-block::

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

.. _cli/help/dnu:

Deriving :math:`\Delta\nu`
**************************

.. code-block::

      --dnu [value [value ...]]
                            Brute force method to provide value for dnu
      --method method       Method to use to determine dnu, ~[M, A, D]
      --peak n, --peaks n, --npeaks n
                            Number of peaks to fit in the ACF
      --sp value, --smoothps value
                            Box filter width [in muHz] of PS for ACF
      --thresh value, --threshold value
                            Fractional value of FWHM to use for ACF

.. _cli/help/ech:



Echelle diagram
***************

.. code-block::

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

.. _cli/help/mc:

Sampling
*********

.. code-block::

      --mc n, --iter n, --mciter n
                            Number of Monte-Carlo iterations
      -m, --samples         Save samples from the Monte-Carlo sampling


which shows a very long but very healthy list of available options. We tried to make this
easier on the eyes by separating the commands into related groups, but do not fret! We realize
this is a lot of information, which is why we have dedicated an entire page to describing these
features.

Additionally, we have examples of some put to use in :ref:`advanced usage<advanced>` 
and also have included a brief :ref:`tutorial` below that describes some of these commands.

.. warning::

    All parameters are optimized for most star types but some may need adjusting. 
    An example is the smoothing width (``--sw``), which is 20 muHz by default, but 
    may need to be adjusted based on the nyquist frequency and frequency resolution 
    of the input power spectrum.

-----

.. _cli/commands:

Option list
###########

Due to the large number of available commands, we have sorted parameters by:

- :ref:`related groups <cli/groups>`
- :ref:`option types <cli/types>`

.. note::

    Our features were developed using principles from Unix-like operating systems, 
    where a single hyphen can be followed by multiple single-character flags (i.e.
    mostly boolean flags that do not require additional input). 
    
    An example is ``-dvoi``, which is far more convenient than writing ``--display --verbose 
    --overwrite --include``. Together, these commands tell ``pySYD`` to:
     1. Display the output figures (``-d``, ``--show``, ``--display``),
     2. Turn on the verbose output (``-v``, ``--verbose``),
     3. Overwrite existing files with the same name (``-o``, ``--overwrite``), and
     4. Include the model metrics and values with the verbose output (``-i``, ``--include``).

-----

.. _cli/groups:

By related topics
*****************

Jump to:

- :ref:`high-level functions <cli/groups/high>`
- :ref:`data analyses <cli/groups/data>`
- :ref:`estimating numax <cli/groups/est>`
- :ref:`granulation background <cli/groups/bg>`
- :ref:`final numax <cli/groups/numax>`
- :ref:`final dnu <cli/groups/dnu>`
- :ref:`echelle diagram <cli/groups/ech>`
- :ref:`estimating uncertainties <cli/groups/mc>`
- :ref:`parallel processing <cli/groups/pp>`


.. _cli/groups/high:

High-level functionality
````````````````````````

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


.. _cli/groups/data:

Data analyses
`````````````

The following features are primarily related to the initial and final treatment of
data products, including information about the input data, how to process and save
the data as well as which modules to run.

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


.. _cli/groups/est:

Estimating numax
````````````````

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


.. _cli/groups/bg:

Granulation background
``````````````````````

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


.. _cli/groups/numax:
   
Deriving numax
``````````````

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


.. _cli/groups/dnu:

Deriving dnu
````````````

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
   

.. _cli/groups/ech:

Echelle diagram
```````````````

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


.. _cli/groups/mc:
   
Estimating uncertainties
````````````````````````

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
  

.. _cli/groups/pp:

Parallel processing
```````````````````

Additional option for the number of threads to use when running stars in parallel.

- ``--nt``, ``--nthread``, ``--nthreads`` 
   * dest = ``args.n_threads``
   * help = Number of processes to run in parallel. If nothing is provided, the software will use the ``multiprocessing`` package to determine the number of CPUs on the operating system and then adjust accordingly.
   * type = int
   * default = `0`

-----

Glossary of options
###################

.. glossary::

    a, ask
        the option to select which trial (or estimate) of numax to use from the first module
        **TODO: this is not yet operational**
         * dest = ``args.ask``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    b, bg, background
        controls the background-fitting procedure -- BUT this should never be touched
        since a majority of the work done in the software happens here and it should 
        not need to be turned off
         * dest = ``args.background``
         * type = ``bool``
         * default = ``True``
         * action = ``store_false``
    
    c, cli
        while in the list of commands, this option should not be tinkered with for current
        users. The purpose of adding this was to extend it to beyond the basic command-line
        usage -- therefore, this triggers to ``False`` when calling functions from a notebook
         * dest = ``args.cli``
         * type = ``bool``
         * default = ``True``
         * action = ``store_false``
        
    d, save
        turn off the automatic saving of output figures and files
         * dest = ``args.save``
         * type = ``bool``
         * default = ``True``
         * action = ``store_false``
          
    e, ie, interpech
        turn on the bilinear interpolation of the plotted echelle diagram
         * dest = ``args.interp_ech``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
           
    f, fix, fixwn, wn
        fix the white noise level in the background fitting **NOT operational yet**
        this still needs to be tested
         * dest = ``args.fix``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
            
    g, globe, global
        do not estimate the global asteroseismic parameter numax and dnu (although
        I'm not sure why you would want to do that because that's exactly what this
        software is intended for)
         * dest = ``args.globe``
         * type = ``bool``
         * default = ``True``
         * action = ``store_false``
    
    i, include
        include metric (i.e. BIC, AIC) values in verbose output during the background
        fitting procedure
         * dest = ``args.include``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    k, kc, kepcorr
        turn on the *Kepler* short-cadence artefact correction module. if you don't
        know what a *Kepler* short-cadence artefact is, chances are you shouldn't mess
        around with this option yet
         * dest = ``args.kepcorr``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    laws, nlaws
        force the number of red-noise component(s). **fun fact:** the older IDL version
        of ``SYD`` fixed this number to ``2`` for the *Kepler* legacy sample -- now we
        have made it customizable all the way down to an individual star!
         * dest = ``args.n_laws``
         * type = ``int``
         * default = `None`
    
    m, samples
        option to save the samples from the Monte-Carlo sampling (i.e. parameter 
        posteriors) in case you'd like to reproduce your own plots, etc.
         * dest = ``args.samples``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    mc, iter, mciter
        number of Monte-Carlo-like iterations. This is `1` by default, since you should
        always check the data and output figures before running the sampling algorithm.
        But for purposes of generating uncertainties, `n=200` is typically sufficient.
         * dest = ``args.mc_iter``
         * type = ``int``
         * default = `1`
    
    n, notch
        use notching technique to reduce effects from mixes modes (pretty sure this is not
        full functional yet, creates weird effects for higher SNR cases)
         * dest = ``args.notching``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    nox, nacross
        specifies the number of bins (i.e. the resolution) to use for the x-axis of the
        echelle diagram -- fixing this number if complicated because it depends on both the
        resolution of the power spectrum as well as the characteristic frequency separation.
        This is another example where, if you don't know what this means, you probably should
        not change it.
         * dest = ``args.nox``
         * type = ``int``
         * default = `50`
    
    noy, ndown, norders
        similar to :term:`nox`, this specifies the number of bins (i.e. orders) to use on the
        y-axis of the echelle diagram. **TODO:** check how it is automatically calculating the
        number of orders since there cannot be `0`.
         * dest = ``args.noy``
         * type = ``int``
         * default = `0`
    
    nt, nthread, nthreads
        the number of processes to run in parallel. If nothing is provided when you run in ``pysyd.parallel``
        mode, the software will use the ``multiprocessing`` package to determine the number of CPUs on the
        operating system and then adjust accordingly. **In short:** this probably does not need to be changed
         * dest = ``args.n_threads``
         * type = ``int``
         * default = `0`
    
    o, over, overwrite
        newer option to overwrite existing files with the same name/path since it will now add extensions
        with numbers to avoid overwriting these files
         * dest = ``args.overwrite``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    ofa, ofactual
        the oversampling factor of the provided power spectrum. Default is `0`, which means it is calculated from
        the time series data. **Note:** this needs to be provided if there is no time series data!
         * dest = ``args.of_actual``
         * type = ``int``
         * default = `0`
    
    ofn, ofnew
        the new oversampling factor to use in the first iteration of both modules ** see performance for more details?
         * dest = ``args.of_new``
         * type = ``int``
         * default = `5`
    
    p, par, parallel
        run ``pySYD`` in parallel mode
         * dest = ``args.parallel``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    s, show
        show output figures, which is not recommended if running many stars
         * dest = ``args.show``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    t, test
        extra verbose output for testing functionality (not currently implemented)
        **NEED TO DO**
         * dest = ``args.test``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    v, verbose
        turn on the verbose output (also not recommended when running many stars, and
        definitely *not* when in parallel mode) **Check** this but I think it will be
        disabled automatically if the parallel mode is `True`
         * dest = ``args.verbose``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    x, ex, excess
        turn off the find excess module -- this will automatically happen if :term:`numax`
        is provided
         * dest = ``args.excess``
         * type = ``bool``
         * default = ``True``
         * action = ``store_false``
    
    y, hey
        plugin for Daniel Hey's interactive echelle package **but is not currently implemented**
        **TODO**
         * dest = ``args.hey``
         * type = ``bool``
         * default = ``False``
         * action = ``store_true``
    
    




- ``--peak``, ``--peaks``, ``--npeaks``
   * dest = ``args.n_peaks``
   * type = ``int``
   * help = Number of peaks to fit in the ACF
   * default = `5`

- ``--rms``, ``--nrms``
   * dest = ``args.n_rms``
   * type = ``int``
   * help = Number of points used to estimate amplitudes of individual background components (this should rarely need to be touched)
   * default = `20`

- ``--trials``, ``--ntrials``
   * dest = ``args.n_trials``
   * type = ``int``
   * help = Number of trials to estimate numax
   * default = `3`

- ``--bf``, ``--box``, ``--boxfilter``
   * dest = ``args.box_filter``
   * type = ``float``
   * help = Box filter width (in muHz) for plotting the power spectrum
   * default = `1.0`
   * unit = muHz

- ``--bin``, ``--binning``
   * dest = ``args.binning``
   * type = ``float``
   * help = Interval for binning of spectrum in log(muHz) (bins equally in logspace).
   * default = `0.005`
   * unit = log(muHz)

- ``--cv``, ``--value``
   * dest = ``args.clip_value``
   * type = ``float``
   * help = Clip value for echelle diagram (i.e. if ``args.clip_ech`` is ``True``). If none is provided, it will cut at 3x the median value of the folded power spectrum.
   * default = ``3.0``
   * unit = fractional psd

- ``--dnu``
   * dest = ``args.dnu``
   * type = ``float``
   * help = Brute force method to provide value for dnu
   * nargs = '*'
   * default = ``None``

- ``--ew``, ``--exwidth``
   * dest = ``args.width``
   * type = ``float``
   * help = Fractional value of width to use for power excess, where width is computed using a solar scaling relation and then centered on the estimated numax.
   * default = `1.0`

- ``--iw``, ``--indwidth``
   * dest = ``args.ind_width``
   * type = ``float``
   * help = Width of binning for power spectrum (in muHz)
   * default = `20.0`

- ``--lb``, ``--lowerb``
   * dest = ``args.lower_bg``
   * type = ``float``
   * help = Lower limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * default = ``1.0``
   * unit = muHz

- ``--le``, ``--uppere``
   * dest = ``args.lower_ech``
   * type = ``float``
   * help = Lower frequency limit of the folded PS to whiten mixed modes before determining the correct dnu
   * nargs = '*'
   * default = ``None``
   * unit = muHz

- ``--lp``, ``--lowerp``
   * dest = ``args.lower_ps``
   * type = ``float``
   * help = Lower frequency limit for zoomed in power spectrum (around power excess)
   * nargs = '*'
   * default = ``None``
   * unit = muHz

- ``--lx``, ``--lowerx``
   * dest = ``args.lower_ex``
   * type = ``float``
   * help = Lower limit of power spectrum to use in findex module
   * default = `1.0`
   * unit = muHz

- ``--numax``
   * dest = ``args.numax``
   * type = ``float``
   * help = Brute force method to bypass findex and provide value for numax. Please note: len(args.numax) == len(args.targets) for this to work! This is mostly intended for single star runs.
   * nargs = '*'
   * default = ``None``

- ``--se``, ``--smoothech``
   * dest = ``args.smooth_ech``
   * type = ``float``
   * help = Option to smooth the echelle diagram output using a box filter
   * default = ``None``
   * unit = muHz

- ``--sm``, ``--smpar``
   * dest = ``args.sm_par``
   * type = ``float``
   * help = Value of smoothing parameter to estimate the smoothed numax (typical values range from `1`-`4`)
   * default = `None`

- ``--sp``, ``--smoothps``
   * dest = ``args.smooth_ps``
   * type = ``float``
   * help = Box filter width for smoothing of the power spectrum. The default is 2.5, but will switch to 0.5 for more evolved stars (numax < 500 muHz).
   * default = `2.5`
   * unit = muHz

- ``--step``, ``--steps``
   * dest = ``args.step``
   * type = ``float``
   * help = The step width for the collapsed ACF wrt the fraction of the boxsize
   * default = `0.25`

- ``--sw``, ``--smoothwidth``
   * dest = ``args.smooth_width``
   * type = ``float``
   * help = Box filter width (in muHz) for smoothing the power spectrum
   * default = `20.0`

- ``--thresh``, ``--threshold``
   * dest = ``args.threshold``
   * type = ``float``
   * help = Fractional value of the ACF FWHM to use for determining dnu
   * default = ``1.0``

- ``--ub``,  ``--upperb``
   * dest = ``args.upper_bg``
   * type = ``float``
   * help = Upper limit of power spectrum to use in fitbg module. Please note: unless numax is known, it is not suggested to fix this beforehand.
   * nargs = '*'
   * default = ``6000.0``
   * unit = muHz

- ``--ue``,  ``--uppere``
   * dest = ``args.upper_ech``
   * type = ``float``
   * help = Upper frequency limit of the folded PS to whiten mixed modes before determining the correct dnu
   * nargs = '*'
   * default = ``None``
   * unit = muHz

- ``--up``,  ``--upperp``
   * dest = ``args.upper_ps``
   * type = ``float``
   * help = Upper frequency limit for zoomed in power spectrum (around power excess)
   * nargs = '*'
   * default = ``None``
   * unit = muHz

- ``--ux``, ``--upperx``
   * dest = ``args.upper_ex``
   * type = ``float``
   * help = Upper limit of power spectrum to use in findex module
   * default = `6000.0`
   * unit = muHz

- ``--basis``
   * dest = ``args.basis``
   * type = ``str``
   * help = Which basis to use for background fit (i.e. 'a_b', 'pgran_tau', 'tau_sigma'), *** NOT operational yet ***
   * default = `'tau_sigma'`

- ``--bm``, ``--mode``, ``--bmode`` 
   * dest = ``args.mode``
   * type = ``str``
   * help = Which mode to use when binning. Choices are ["mean", "median", "gaussian"]
   * default = ``mean``

- ``--ce``, ``--cm``, ``--color``
   * dest = ``args.cmap``
   * type = ``str``
   * help = Change colormap of ED, which is `binary` by default.
   * default = ``binary``

- ``--file``, ``--list``, ``--todo``
   * dest = ``args.file``
   * type = ``str``
   * help = Path to text file that contains the list of stars to process (convenient for running many stars).
   * default = ``TODODIR``

- ``--in``, ``--input``, ``--inpdir``
   * dest = ``args.inpdir``
   * type = ``str``
   * help = Path to input data
   * default = ``INPDIR``

- ``--info``, ``--information`` 
   * dest = ``args.info``
   * type = ``str``
   * help = Path to the csv containing star information (although not required).
   * default = ``INFODIR``

- ``--method``
   * dest = ``args.method``
   * type = ``str``
   * help = Method to use to determine dnu, choices ~['M', 'A', 'D']
   * default = ``D``

- ``--metric``
   * dest = ``args.metric``
   * type = ``str``
   * help = Which model metric to use for the best-fit background model, choices~['bic','aic']
   * default = `'bic'`

- ``--out``, ``--output``, ``--outdir``
   * dest = ``args.outdir``
   * type = ``str``
   * help = Path that results are saved to
   * default = ``OUTDIR``

- ``--star``, ``--stars``
   * dest = ``args.star``
   * type = ``str``
   * help = List of stars to process. Default is ``None``, which will read in the star list from ``args.file``.
   * nargs = '*'
   * default = ``None``


-----

.. _cli/examples::

Examples
#########

.. role:: bash(code)
   :language: bash


Below are examples of how to use specific ``pySYD`` command-line features, including before and after figures
to better demonstrate the difference. 


``--dnu``: force dnu
********************

+-------------------------------------------------+---------------------------------------------------------+
| Before                                          | After                                                   |
+=================================================+=========================================================+
| Fix the dnu value if the desired dnu is not automatically selected by `pySYD`.                            |
+-------------------------------------------------+---------------------------------------------------------+
|:bash:`pysyd run --star 9512063 --numax 843`     |:bash:`pysyd run --star 9512063 --numax 843 --dnu 49.54` |
+-------------------------------------------------+---------------------------------------------------------+
| .. figure:: ../figures/advanced/9512063_before.png | .. figure:: ../figures/advanced/9512063_after.png          |
|    :width: 600                                  |    :width: 600                                          |
+-------------------------------------------------+---------------------------------------------------------+


``--ew``: excess width
***********************

+------------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                           | After                                                            |
+==================================================================+==================================================================+
| Changed the excess width in the background corrected power spectrum used to calculate the ACF (and hence dnu).                      |
+------------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run --star 9542776 --numax 900`                     | :bash:`pysyd run --star 9542776 --numax 900 --ew 1.5`            |
+------------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: ../figures/advanced/9542776_before.png                  | .. figure:: ../figures/advanced/9542776_after.png                   |
|    :width: 600                                                   |    :width: 600                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+


``--ie``: smooth echelle
************************

+------------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                           | After                                                            |
+==================================================================+==================================================================+
| Smooth echelle diagram by turning on the interpolation, in order to distinguish the modes for low SNR cases.                        |
+------------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run 3112889 --numax 871.52 --dnu 53.2`              | :bash:`pysyd run --star 3112889 --numax 871.52 --dnu 53.2 --ie`  |
+------------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: ../figures/advanced/3112889_before.png                  | .. figure:: ../figures/advanced/3112889_after.png                   |
|    :width: 600                                                   |    :width: 600                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+


``--kc``: Kepler correction
***************************

+------------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                           | After                                                            |
+==================================================================+==================================================================+
| Remove *Kepler* artefacts from the power spectrum for an accurate numax estimate.                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run --star 8045442 --numax 550`                     | :bash:`pysyd run --star 8045442 --numax 550 --kc`                |
+------------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: ../figures/advanced/8045442_before.png                  | .. figure:: ../figures/advanced/8045442_after.png                   |
|    :width: 600                                                   |    :width: 600                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+


``--lp``: lower frequency of power excess
*****************************************

+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Before                                                                   | After                                                                    |
+==========================================================================+==========================================================================+
| Set the lower frequency limit in zoomed in power spectrum; useful when an artefact is present close to the excess and cannot be removed otherwise.  |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| :bash:`pysyd run --star 10731424 --numax 750`                            | :bash:`pysyd run --star 10731424 --numax 750 --lp 490`                   |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| .. figure:: ../figures/advanced/10731424_before.png                         | .. figure:: ../figures/advanced/10731424_after.png                          |
|    :width: 600                                                           |    :width: 600                                                           |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+


``--npeaks``: number of peaks
*****************************

+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Before                                                                   | After                                                                    |
+==========================================================================+==========================================================================+
| Change the number of peaks chosen in ACF; useful in low SNR cases where the spectrum is noisy and ACF has many peaks close to the expected dnu.     |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| :bash:`pysyd run --star 9455860`                                         | :bash:`pysyd run --star 9455860 --npeaks 10`                             |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| .. figure:: ../figures/advanced/9455860_before.png                          | .. figure:: ../figures/advanced/9455860_after.png                           |
|    :width: 600                                                           |    :width: 600                                                           |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+


``--numax``
************

+-------------------------------------------------------+-------------------------------------------------------+
| Before                                                | After                                                 |
+=======================================================+=======================================================+
| Set the numax value if pySYD chooses the wrong excess in the power spectrum.                                  |
+-------------------------------------------------------+-------------------------------------------------------+
| :bash:`pysyd run --star 5791521`                      | :bash:`pysyd run --star 5791521 --numax 670`          |
+-------------------------------------------------------+-------------------------------------------------------+
| .. figure:: ../figures/advanced/5791521_before.png       | .. figure:: ../figures/advanced/5791521_after.png        |
|    :width: 600                                        |    :width: 600                                        |
+-------------------------------------------------------+-------------------------------------------------------+


``--ux``: upper frequency of PS used in the first module
********************************************************

+--------------------------------------------------+-------------------------------------------------------+
| Before                                             | After                                                     |
+==================================================+=======================================================+
| Set the upper frequency limit in power spectrum; useful when `pySYD` latches on to an artefact.                |
+--------------------------------------------------+-------------------------------------------------------+
| :bash:`pysyd run --star 11769801`                   | :bash:`pysyd run --star 11769801 -ux 3500`               |
+--------------------------------------------------+-------------------------------------------------------+
| .. figure:: ../figures/advanced/11769801_before.png | .. figure:: ../figures/advanced/11769801_after.png       |
|    :width: 600                                      |    :width: 600                                           |
+--------------------------------------------------+-------------------------------------------------------+


Below is a quick, crash course demonstrating the easy accessibility of
``pySYD`` via command line.

.. raw:: html

   <iframe width="680" height="382.5" src="https://www.youtube.com/embed/c1do_BKtHXk" 
   title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; 
   clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


-----
