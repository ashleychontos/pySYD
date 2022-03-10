.. _cli/index:

**********************
As a command-line tool
**********************

.. note::

    ``pySYD`` was developed s.t. it would be primarily used as a command-line tool. However, we 
    acknowledge that this is not convenient for everyone and have therefore provided some tutorials 
    on how to use it in a Jupyter notebook.

.. _cli/help:

CLI Help 
###########

From terminal, the following help command for the main pipeline execution (via ``pysyd.pipeline.run``): 

.. code-block::

    pysyd run --help
    
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

will display an enormous list of options but do not fret, this is for good reason and we have
broken them down into relevant groups to make it easier to digest. It may seem like an overwhelming 
amount but they are only there to make your asteroseismic experience as customizable as possible.

**Jump to:**
 - :ref:`high-level functions <cli/help/high>`
 - :ref:`data analyses <cli/help/data>`
 - :ref:`estimating numax <cli/help/est>`
 - :ref:`granulation background <cli/help/bg>`
 - :ref:`final numax value <cli/help/numax>`
 - :ref:`final dnu value <cli/help/dnu>`
 - :ref:`echelle diagram <cli/help/ech>`
 - :ref:`sampling <cli/help/mc>`
 - :ref:`parallel processing <cli/help/pp>`

As you are navigating this webpage, keep in mind that we have a special :ref:`cli/glossary` for our
CLI options.

.. _cli/help/high:

High-level functions
********************

Below is the first part of the output, which is primarily related to the higher level functionality.
Within the software, these are defined by the parent and main parsers, which are inevitably inherited
by all ``pySYD`` modes that handle the data.

All ``pySYD`` modes inherent the parent parser, which includes the properties 
enumerated below. With the exception of the ``verbose`` command, most of these
features are related to the initial (setup) paths and directories and should be
used very sparingly. 

.. _cli/help/data:

Initial data analyses
*********************

The following features are primarily related to the initial and final treatment of
data products, including information about the input data, how to process and save
the data as well as which modules to run.

.. code-block::

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

The following options are relevant for the first, optional module that is designed
to estimate numax if it is not known: 

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

Below is a complete list of parameters relevant to the background-fitting routine:

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

All of the following parameters are related to deriving numax, or the frequency
corresponding to maximum power:

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

Below are all options related to the characteristic frequency spacing (dnu):

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

All customizable options relevant for the echelle diagram output:

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

All CLI options relevant for the Monte-Carlo sampling:

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

.. _cli/glossary:

Glossary of options
###################

.. glossary::

    ``-a``
    ``--ask``
        the option to select which trial (or estimate) of numax to use from the first module
        **TODO: this is not yet operational**
         * dest = ``args.ask``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``-b``
    ``--bg``
    ``--background``
        controls the background-fitting procedure -- BUT this should never be touched
        since a majority of the work done in the software happens here and it should 
        not need to be turned off
         * dest = ``args.background``
         * type = `bool`
         * default = `True`
         * action = ``store_false``
    
    ``--bf``
    ``--box``
    ``--boxfilter``
        box filter width for plotting the power spectrum **TODO:** make sure this does
        not affect any actual measurements and this is just an aesthetic
         * dest = ``args.box_filter``
         * type = `float`
         * default = `1.0`
         * unit = :math:`\mu \mathrm{Hz}`
         
    ``--bin``
    ``--binning``
        interval for the binning of spectrum in :math:`\mathrm{log(}\mu\mathrm{Hz)}`
        *this bins equally in logspace*
         * dest = ``args.binning``
         * type = `float`
         * default = `0.005`
         * unit = log(:math:`\mu \mathrm{Hz}`)
    
    ``--cv``
    ``--value``
        the clip value to use for the output echelle diagram if and only if ``args.clip_ech`` is
        ``True``. If none is provided, it will use a value that is 3x the median value of the folded
        power spectrum
         * dest = ``args.clip_value``
         * type = `float`
         * default = `3.0`
         * unit = fractional psd
    
    ``-c``
    ``--cli``
        while in the list of commands, this option should not be tinkered with for current
        users. The purpose of adding this was to extend it to beyond the basic command-line
        usage -- therefore, this triggers to ``False`` when calling functions from a notebook
         * dest = ``args.cli``
         * type = `bool`
         * default = `True`
         * action = ``store_false``

    ``-d``
    ``--show``
    ``--display``
        show output figures, which is not recommended if running many stars
         * dest = ``args.show``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--dnu``
        brute force method to provide value for dnu. **Note:** if using the ``pysyd.utils.whiten_mixed`` 
        modes module, this will need to be provided along with :term:`--le` and :term:`--ue`.
         * dest = ``args.dnu``
         * type = `float`
         * nargs = `'*'`
         * default = `None`
    
    ``-e``
    ``--ie``
    ``--interpech``
        turn on the bilinear interpolation of the plotted echelle diagram
         * dest = ``args.interp_ech``
         * type = `bool`
         * default = `False`
         * action = ``store_true``

    ``--ew``
    ``--exwidth``
        the fractional value of the width to use surrounding the power excess, which is computed using a solar
        scaling relation (and then centered on the estimated :math:`\nu_{\mathrm{max}}`) **SEE ALSO:** :term:`--lp`, :term:`--up`
         * dest = ``args.width``
         * type = `float`
         * default = `1.0`
         * unit = fractional :math:`\mu \mathrm{Hz}`
           
    ``-f``
    ``--fix``
    ``--fixwn``
    ``--wn``
        fix the white noise level in the background fitting **NOT operational yet**
        this still needs to be tested
         * dest = ``args.fix``
         * type = `bool`
         * default = `False`
         * action = ``store_true``

    ``-g``
    ``--globe``
    ``--global``
        do not estimate the global asteroseismic parameter numax and dnu (although
        I'm not sure why you would want to do that because that's exactly what this
        software is intended for)
         * dest = ``args.globe``
         * type = `bool`
         * default = `True`
         * action = ``store_false``
    
    ``-i``
    ``--include``
        include metric (i.e. BIC, AIC) values in verbose output during the background
        fitting procedure
         * dest = ``args.include``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--iw``
    ``--indwidth``
        width of binning for the power spectrum used in the first module **TODO: CHECK THIS** 
         * dest = ``args.ind_width``
         * type = `float`
         * default = `20.0`
         * unit = :math:`\mu \mathrm{Hz}`
    
    ``-k``
    ``--kc``
    ``--kepcorr``
        turn on the *Kepler* short-cadence artefact correction module. if you don't
        know what a *Kepler* short-cadence artefact is, chances are you shouldn't mess
        around with this option yet
         * dest = ``args.kepcorr``
         * type = ``bool``
         * default = `False`
         * action = ``store_true``
    
    ``--laws``
    ``--nlaws``
        force the number of red-noise component(s). **fun fact:** the older IDL version
        of ``SYD`` fixed this number to ``2`` for the *Kepler* legacy sample -- now we
        have made it customizable all the way down to an individual star!
         * dest = ``args.n_laws``
         * type = `int`
         * default = `None`

    ``--lb``
    ``--lowerb``
        the lower frequency limit of the power spectrum to use in the background-fitting
        routine. **Please note:** unless :math:`\nu_{\mathrm{max}}` is known, it is highly 
        recommended that you do *not* fix this beforehand
         * dest = ``args.lower_bg``
         * type = `float`
         * nargs = `'*'`
         * default = `1.0`
         * unit = :math:`\mu \mathrm{Hz}`
         
    ``--le``
    ``--uppere``
        the lower frequency limit of the folded power spectrum to "whiten" mixed modes before
        estimating the final value for dnu **this must be used with** :term:`--dnu` and 
        :term:`--ue`` **in order to work properly**
         * dest = ``args.lower_ech``
         * type = `float`
         * nargs = `'*'`  
         * default = `None`
         * unit = :math:`\mu \mathrm{Hz}`
         
    ``--lp``
    ``--lowerp``
        to change the lower frequency limit of the zoomed in power spectrum (i.e. the region with the supposed
        power excess due to oscillations). Similar to :term:`--ew` but instead of a fractional value w.r.t. the 
        scaled solar value, you can provide hard boundaries in this case **TODO** check if it requires and upper
        bound -- pretty sure it doesn't but should check 
         * dest = ``args.lower_ps``
         * type = `float`
         * nargs = `'*'`
         * default = `None`
         * unit = :math:`\mu \mathrm{Hz}`
         
    ``--lx``
    ``--lowerx``
        the lower limit of the power spectrum to use in the first module (to estimate numax)
         * dest = ``args.lower_ex``
         * type = `float`
         * default = `1.0`
         * unit = :math:`\mu \mathrm{Hz}`
         
    ``-m``
    ``--samples``
        option to save the samples from the Monte-Carlo sampling (i.e. parameter 
        posteriors) in case you'd like to reproduce your own plots, etc.
         * dest = ``args.samples``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--mc``
    ``--iter``
    ``--mciter``
        number of Monte-Carlo-like iterations. This is `1` by default, since you should
        always check the data and output figures before running the sampling algorithm.
        But for purposes of generating uncertainties, `n=200` is typically sufficient.
         * dest = ``args.mc_iter``
         * type = `int`
         * default = `1`
    
    ``-n``
    ``--notch``
        use notching technique to reduce effects from mixes modes (pretty sure this is not
        full functional yet, creates weird effects for higher SNR cases)
         * dest = ``args.notching``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--nox``
    ``--nacross``
        specifies the number of bins (i.e. the resolution) to use for the x-axis of the
        echelle diagram -- fixing this number if complicated because it depends on both the
        resolution of the power spectrum as well as the characteristic frequency separation.
        This is another example where, if you don't know what this means, you probably should
        not change it.
         * dest = ``args.nox``
         * type = `int`
         * default = `50`
    
    ``--noy``
    ``--ndown``
    ``--norders``
        similar to :term:`nox`, this specifies the number of bins (i.e. orders) to use on the
        y-axis of the echelle diagram. **TODO:** check how it is automatically calculating the
        number of orders since there cannot be `0`.
         * dest = ``args.noy``
         * type = `int`
         * default = `0`
    
    ``--nt``
    ``--nthread``
    ``--nthreads``
        the number of processes to run in parallel. If nothing is provided when you run in ``pysyd.parallel``
        mode, the software will use the ``multiprocessing`` package to determine the number of CPUs on the
        operating system and then adjust accordingly. **In short:** this probably does not need to be changed
         * dest = ``args.n_threads``
         * type = `int`
         * default = `0`
         
    ``--numax``
        brute force method to bypass the first module and provide an initial starting value for :math:`\rm \nu_{max}`
        ``Asserts len(args.numax) == len(args.targets)``
        * dest = ``args.numax``
        * type = `float`
        * nargs = `'*'`
        * default = `None`
        * unit = :math:`\mu \mathrm{Hz}`
    
    ``-o``
    ``--over``
    ``--overwrite``
        newer option to overwrite existing files with the same name/path since it will now add extensions
        with numbers to avoid overwriting these files
         * dest = ``args.overwrite``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--ofa``
    ``--ofactual``
        the oversampling factor of the provided power spectrum. Default is `0`, which means it is calculated from
        the time series data. **Note:** this needs to be provided if there is no time series data!
         * dest = ``args.of_actual``
         * type = `int`
         * default = `0`
    
    ``--ofn``
    ``--ofnew``
        the new oversampling factor to use in the first iteration of both modules ** see performance for more details?
         * dest = ``args.of_new``
         * type = `int`
         * default = `5`
    
    ``-p``
    ``--par``
    ``--parallel``
        run ``pySYD`` in parallel mode
         * dest = ``args.parallel``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--peak``
    ``--peaks``
    ``--npeaks``
        the number of peaks to identify in the autocorrelation function
         * dest = ``args.n_peaks``
         * type = `int`
         * default = `5`
    
    ``--rms``
    ``--nrms``
        the number of points used to estimate the amplitudes of individual background (red-noise) components
        *Note: this should only rarely need to be touched*
         * dest = ``args.n_rms``
         * type = `int`
         * default = `20`
    
    ``-s``
    ``--save``
        turn off the automatic saving of output figures and files
         * dest = ``args.save``
         * type = `bool`
         * default = `True`
         * action = ``store_false``

    ``--se``
    ``--smoothech``
   * dest = ``args.smooth_ech``
   * type = `float`
   * help = Option to smooth the echelle diagram output using a box filter
   * default = `None`
   * unit = muHz

    ``--sm``
    ``--smpar``
   * dest = ``args.sm_par``
   * type = `float`
   * help = Value of smoothing parameter to estimate the smoothed numax (typical values range from `1`-`4`)
   * default = `None`

    ``--sp``
    ``--smoothps``
        the box filter width used for smoothing of the power spectrum. The default is `2.5` but will switch to
        `0.5` for more evolved stars (if :math:`\rm \nu_{max}` < 500 :math:`\mu \mathrm{Hz}`)
         * dest = ``args.smooth_ps``
         * type = `float`
         * default = `2.5`
         * unit = :math:`\mu \mathrm{Hz}`

    ``--step``
    ``--steps``
        the step width for the collapsed autocorrelation function w.r.t. the fraction of the
        boxsize. **Please note:** this should not be adjusted
         * dest = ``args.step``
         * type = `float`
         * default = `0.25`
         * unit = fractional :math:`\mu \mathrm{Hz}`

    ``--sw``
    ``--smoothwidth``
        the width of the box filter that is used to smooth the power spectrum
         * dest = ``args.smooth_width``
         * type = `float`
         * default = `20.0`
         * unit = :math:`\mu \mathrm{Hz}`
         
.. warning::

    All parameters are optimized for most star types but some may need adjusting. 
    An example is the smoothing width (``--sw``), which is 20 muHz by default, but 
    may need to be adjusted based on the nyquist frequency and frequency resolution 
    of the input power spectrum.

.. glossary::
    
    ``-t``
    ``--test``
        extra verbose output for testing functionality (not currently implemented)
        **NEED TO DO**
         * dest = ``args.test``
         * type = `bool`
         * default = `False`
         * action = ``store_true``

    ``--thresh``
    ``--threshold``
        the fractional value of the autocorrelation function's full width at half
        maximum (which is important in this scenario because it is used to determine :math:`\Delta\nu`)
         * dest = ``args.threshold``
         * type = `float`
         * default = `1.0`
         * unit = fractional :math:`\mu \mathrm{Hz}`
    
    ``--trials``
    ``--ntrials``
        the number of trials used to estimate numax in the first module -- can be bypassed if :term:`--numax`
        is provided.
         * dest = ``args.n_trials``
         * type = `int`
         * default = `3`
    
    ``-v``
    ``--verbose``
        turn on the verbose output (also not recommended when running many stars, and
        definitely *not* when in parallel mode) **Check** this but I think it will be
        disabled automatically if the parallel mode is `True`
         * dest = ``args.verbose``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``-x``
    ``--ex``
    ``--excess``
        turn off the find excess module -- this will automatically happen if :term:`numax`
        is provided
         * dest = ``args.excess``
         * type = `bool`
         * default = `True`
         * action = ``store_false``
    
    ``-y``
    ``--hey``
        plugin for Daniel Hey's interactive echelle package **but is not currently implemented**
        **TODO**
         * dest = ``args.hey``
         * type = `bool`
         * default = `False`
         * action = ``store_true``



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
