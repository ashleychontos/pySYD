.. role:: underlined
   :class: underlined

.. _user-guide-intro:

************
Introduction
************

As we have alluded to throughout the documentation, `pySYD` was intended to be used through 
its command-line interface (CLI) -- which means that the software is specifically optimized 
for this usage and therefore most options probably have the best defaults already
set. Here, "best" just means that the defaults work *best* for most stars. 

However, that does not necessarily mean that your star(s) or setting(s) are expected to 
conform or adhere to these settings. In fact, we recommend playing around with some of the 
settings to see how it affects the results, which might help build your intuition for seismic 
analyses. 

.. note:: 

   Please keep in mind that, while we have extensively tested a majority of our options, we are 
   continuously adding new ones which ultimately might break something. If this happens, we 
   encourage you to submit an issue `here <https://github.com/ashleychontos/pySYD/issues/new?assignees=&labels=&template=bug_report.md>`_ 
   and thank you in advance for helping make `pySYD` even better!


.. _user-guide-help:

CLI help
########

To give you a glimpse into the insanely large amount of available options, open up a terminal
window and enter the help command for the main pipeline execution (`run` aka :mod:`pysyd.pipeline.run`), 
since this mode inherits all command-line parsers. 

.. code-block::

    $ pysyd run --help
    
    usage: pySYD run [-h] [--in str] [--infdir str] [--out str] [-s] [-o] [-v]
                     [--cli] [--notebook] [--star [str [str ...]]] [--file str]
                     [--info str] [--gap int] [-x] [--of int] [-k]
                     [--dnu [float [float ...]]] [--le [float [float ...]]]
                     [--ue [float [float ...]]] [-n] [-e] [-j] [--def str]
                     [--sw float] [--bin float] [--bm str] [--step float]
                     [--trials int] [-a] [--lx [float [float ...]]]
                     [--ux [float [float ...]]] [-b] [--basis str] [--bf float]
                     [--iw float] [--rms int] [--laws int] [-w] [--metric str]
                     [--lb [float [float ...]]] [--ub [float [float ...]]] [-g]
                     [--numax [float [float ...]]] [--lp [float [float ...]]]
                     [--up [float [float ...]]] [--ew float] [--sm float]
                     [--sp float] [-f] [--thresh float] [--peak int] [--mc int]
                     [-m] [--all] [-d] [--cm str] [--cv float] [-y] [-i]
                     [--nox int] [--noy str] [--npb int] [--se float]

    optional arguments:
      -h, --help            show this help message and exit

This was actually just a teaser! 

If you ran it from your end, you probably noticed an output that was a factor of ~5-10 longer! 
It may seem like an overwhelming amount but do not fret, this is for good reason -- and that's 
to make your asteroseismic experience as customized as possible.

Currently `pySYD` has four parsers: the `parent_parser` for high-level functionality, the
`data_parser` for anything related to data loading and manipulation, the `main_parser` for
everything related to the core analyses, and the `plot_parser` for (yes, you guessed it!)
plotting. In fact, the main parser is so large that comprises four subgroups, each related to
the corresponding steps in the main pipeline execution. **BTW** see :ref:`here <library-pipeline-modes>` 
for more information on which parsers a given pipeline mode inherits.

 - :ref:`parent parser <user-guide-intro-parent>`
 - :ref:`data parser <user-guide-intro-data>`
 - :ref:`main parser <user-guide-intro-main>`
    - :ref:`search & estimate <user-guide-intro-est>`
    - :ref:`background fit <user-guide-intro-bg>`
    - :ref:`global fit <user-guide-intro-globe>`
    - :ref:`estimate uncertainties <user-guide-intro-mc>`
 - :ref:`plotting parser <user-guide-intro-plot>`

**Note:** as you are navigating this page, keep in mind that we also have a special 
:ref:`glossary <user-guide-glossary>` for all our command-line options. This includes everything
from the variable type, default value and relevant units to how it's stored within the 
software itself. There are glossary links at the bottom of every section for each of the parameters 
discussed within that subsection.

-----

.. _user-guide-intro-parent:

High-level functionality
########################

c/o the parent parser
*********************

**for all your high-level functionality needs**

All `pySYD` modes inherent the `parent_parser` and therefore, mostly pertains to paths and
how you choose to run the software (i.e. save files and if so, whether or not to overwrite 
old files with the same extension, etc.) 

.. code-block::

      --in str, --input str, --inpdir str
                            Input directory
      --infdir str          Path to relevant pySYD information
      --out str, --outdir str, --output str
                            Output directory
      -s, --save            Do not save output figures and results.
      -o, --overwrite       Overwrite existing files with the same name/path
      -v, --verbose         turn on verbose output
      --cli                 Running from command line (this should not be touched)
      --notebook            Running from a jupyter notebook (this should not be
                            touched)

:underlined:`Glossary terms` (in alphabetical order): 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:term:`--cli<--cli>`, 
:term:`--file<--file, --list, --todo>`, 
:term:`--in<--in, --input, --inpdir>`, 
:term:`--info<--info, --information>`, 
:term:`--information<--info, --information>`, 
:term:`--inpdir<--in, --input, --inpdir>`, 
:term:`--input<--in, --input, --inpdir>`, 
:term:`--list<--file, --list, --todo>`, 
:term:`--notebook<--notebook>`, 
:term:`-o<-o, --overwrite>`, 
:term:`--out<--out, --output, --outdir>`, 
:term:`--overwrite<-o, --overwrite>`, 
:term:`-s<-s, --save>`, 
:term:`--save<-s, --save>`,
:term:`--outdir<--out, --output, --outdir>`, 
:term:`--output<--out, --output, --outdir>`, 
:term:`--todo<--file, --list, --todo>`, 
:term:`-v<-v, --verbose>`, 
:term:`--verbose<-v, --verbose>`

-----

.. _user-guide-intro-data:

Data analyses
#############

aka `data_parser`
*****************

**for anything and everything related to input data and manipulation**

The following features are primarily related to the input data and when applicable, what 
tools to apply to the data. All data manipulation relevant to this step happens *prior*
to any pipeline analyses. **Currently this is mostly frequency-domain tools but we are 
working on implementing time-domain tools as well!**

.. code-block::

      --star [str [str ...]], --stars [str [str ...]]
                            list of stars to process
      --file str, --list str, --todo str
                            list of stars to process
      --info str, --information str
                            list of stellar parameters and options
      --gap int, --gaps int
                            What constitutes a time series 'gap' (i.e. n x the
                            cadence)
      -x, --stitch, --stitching
                            Correct for large gaps in time series data by
                            'stitching' the light curve
      --of int, --over int, --oversample int
                            The oversampling factor (OF) of the input power
                            spectrum
      -k, --kc, --kepcorr   Turn on the Kepler short-cadence artefact correction
                            routine
      --dnu [float [float ...]]
                            spacing to fold PS for mitigating mixed modes
      --le [float [float ...]], --lowere [float [float ...]]
                            lower frequency limit of folded PS to whiten mixed
                            modes
      --ue [float [float ...]], --uppere [float [float ...]]
                            upper frequency limit of folded PS to whiten mixed
                            modes
      -n, --notch           another technique to mitigate effects from mixed modes
                            (not fully functional, creates weirds effects for
                            higher SNR cases??)


:underlined:`Glossary terms` (in alphabetical order): 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:term:`--dnu`
:term:`-k<-k, --kc, --kepcorr>`, 
:term:`--le<--le, --lowere>`, 
:term:`--lowere<--le, --lowere>`,
:term:`--kc<-k, --kc, --kepcorr>`, 
:term:`--kepcorr<-k, --kc, --kepcorr>`, 
:term:`--of<--of, --over, --oversample>`, 
:term:`--over<--of, --over, --oversample>`, 
:term:`--oversample<--of, --over, --oversample>`,  
:term:`--star<--star, --stars>`, 
:term:`--stars<--star, --stars>`, 
:term:`--stitch<-x, --stitch, --stitching>`, 
:term:`--stitching<-x, --stitch, --stitching>`, 
:term:`--ue<--ue, --uppere>`, 
:term:`--uppere<--ue, --uppere>`, 
:term:`-x<-x, --stitch, --stitching>`

-----

.. _user-guide-intro-main:

Main parser
###########

**for the core asteroseismic analyses**

The main parser holds a majority of the parameters that are relevant to core functions of
the software. Since it is so large, it is broken down into four different "groups" which
are related to their application.

.. _user-guide-intro-est:

Search & estimate
*****************

The following options are relevant for the first, optional module that is designed to search
for power excess due to solar-like oscillations and estimate rough starting points for its
main properties.

.. code-block::

      -e, --est, --estimate
                            Turn off the optional module that estimates numax
      -j, --adjust          Adjusts default parameters based on region of
                            oscillations
      --def str, --defaults str
                            Adjust defaults for low vs. high numax values (e.g.,
                            smoothing filters)
      --sw float, --smoothwidth float
                            Box filter width (in muHz) for smoothing the PS
      --bin float, --binning float
                            Binning interval for PS (in muHz)
      --bm str, --mode str, --bmode str
                            Binning mode
      --step float, --steps float
      --trials int, --ntrials int
      -a, --ask             Ask which trial to use
      --lx [float [float ...]], --lowerx [float [float ...]]
                            Lower frequency limit of PS
      --ux [float [float ...]], --upperx [float [float ...]]
                            Upper frequency limit of PS
 
                           
**Glossary terms** (alphabetical order): 
:term:`-a<-a, --ask>`, 
:term:`--ask<-a, --ask>`, 
:term:`--bin<--bin, --binning>`, 
:term:`--binning<--bin, --binning>`, 
:term:`--bm<--bm, --mode, --bmode>`, 
:term:`--bmode<--bm, --mode, --bmode>`, 
:term:`-e<-e, --est, --estimate>`, 
:term:`--est<-e, --est, --estimate>`, 
:term:`--estimate<-e, --est, --estimate>`,
:term:`--lowerx<--lx, --lowerx>`, 
:term:`--lx<--lx, --lowerx>`, 
:term:`--mode<--bm, --mode, --bmode>`, 
:term:`--ntrials<--trials, --ntrials>`, 
:term:`--step<--step, --steps>`, 
:term:`--steps<--step, --steps>`, 
:term:`--sw<--sw, --smoothwidth>`, 
:term:`--smoothwidth<--sw, --smoothwidth>`, 
:term:`--trials<--trials, --ntrials>`, 
:term:`--upperx<--ux, --upperx>`, 
:term:`--ux<--ux, --upperx>`


.. _user-guide-intro-bg:

Background fit
**************

Below is a complete list of parameters relevant to the background-fitting routine:

.. code-block::

      -b, --bg, --background
                            Turn off the routine that determines the stellar
                            background contribution
      --basis str           Which basis to use for background fit (i.e. 'a_b',
                            'pgran_tau', 'tau_sigma'), *** NOT operational yet ***
      --bf float, --box float, --boxfilter float
                            Box filter width [in muHz] for plotting the PS
      --iw float, --indwidth float
                            Width of binning for PS [in muHz]
      --rms int, --nrms int
                            Number of points to estimate the amplitude of red-
                            noise component(s)
      --laws int, --nlaws int
                            Force number of red-noise component(s)
      -w, --wn, --fixwn     Fix the white noise level
      --metric str          Which model metric to use, choices=['bic','aic']
      --lb [float [float ...]], --lowerb [float [float ...]]
                            Lower frequency limit of PS
      --ub [float [float ...]], --upperb [float [float ...]]
                            Upper frequency limit of PS


**Glossary terms** (alphabetical order):  
:term:`-b<-b, --bg, --background>`, 
:term:`--background<-b, --bg, --background>`, 
:term:`--basis`,
:term:`--bf<--bf, --box, --boxfilter>`,
:term:`--bg<-b, --bg, --background>`,   
:term:`--box<--bf, --box, --boxfilter>`, 
:term:`--boxfilter<--bf, --box, --boxfilter>`, 
:term:`--fixwn<-w, --wn, --fixwn>`, 
:term:`--iw<--iw, --indwidth>`, 
:term:`--indwidth<--iw, --indwidth>`, 
:term:`--laws<--laws, --nlaws>`, 
:term:`--lb<--lb, --lowerb>`, 
:term:`--lowerb<--lb, --lowerb>`, 
:term:`--metric`, 
:term:`--nrms<--rms, --nrms>`, 
:term:`--rms<--rms, --nrms>`, 
:term:`--nlaws<--laws, --nlaws>`, 
:term:`--ub<--ub, --upperb>`, 
:term:`--upperb<--ub, --upperb>`, 
:term:`-w<-w, --wn, --fixwn>`, 
:term:`--wn<-w, --wn, --fixwn>`


.. _user-guide-intro-globe:

Global fit
**********

All of the following are related to deriving global asteroseismic parameters, :term:`numax`
(:math:`\rm \nu_{max}`) and :term:`dnu` (:math:`\Delta\nu`). 

.. code-block::

      -g, --globe, --global
                            Disable the main global-fitting routine
      --numax [float [float ...]]
                            initial estimate for numax to bypass the forst module
      --lp [float [float ...]], --lowerp [float [float ...]]
                            lower frequency limit for the envelope of oscillations
      --up [float [float ...]], --upperp [float [float ...]]
                            upper frequency limit for the envelope of oscillations
      --ew float, --exwidth float
                            fractional value of width to use for power excess,
                            where width is computed using a solar scaling
                            relation.
      --sm float, --smpar float
                            smoothing parameter used to estimate the smoothed
                            numax (typically before 1-4 through experience --
                            **development purposes only**)
      --sp float, --smoothps float
                            box filter width [in muHz] of PS for ACF
      -f, --fft             Use :mod:`numpy.correlate` instead of fast fourier
                            transforms to compute the ACF
      --thresh float, --threshold float
                            fractional value of FWHM to use for ACF
      --peak int, --peaks int, --npeaks int
                            number of peaks to fit in the ACF


**Glossary terms** (alphabetical order): 
:term:`--ew<--ew, --exwidth>`, 
:term:`--exwidth<--ew, --exwidth>`, 
:term:`-g<-g, --globe, --global>`, 
:term:`--global<-g, --globe, --global>`, 
:term:`--globe<-g, --globe, --global>`, 
:term:`--lp<--lp, --lowerp>`, 
:term:`--lowerp<--lp, --lowerp>`, 
:term:`--npeaks<--peak, --peaks, --npeaks>`, 
:term:`--numax`, 
:term:`--peak<--peak, --peaks, --npeaks>`, 
:term:`--peaks<--peak, --peaks, --npeaks>`, 
:term:`--sm<--sm, --smpar>`, 
:term:`--smpar<--sm, --smpar>`, 
:term:`--up<--up, --upperp>`, 
:term:`--upperp<--up, --upperp>` :term:`--dnu`,  
:term:`--sp<--sp, --smoothps>`, 
:term:`--smoothps<--sp, --smoothps>`, 
:term:`--thresh<--thresh, --threshold>`


.. _user-guide-intro-mc:

Estimating uncertainties
************************

All CLI options relevant for the Monte-Carlo sampling in order to estimate uncertainties:

.. code-block::

      --mc int, --iter int, --mciter int
                            number of Monte-Carlo iterations to run for estimating
                            uncertainties (typically 200 is sufficient)
      -m, --samples         save samples from the Monte-Carlo sampling


:underlined:`Glossary terms` (in alphabetical order): 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:term:`--iter<--mc, --iter, --mciter>`, 
:term:`-m<-m, --samples>`, 
:term:`--mc<--mc, --iter, --mciter>`, 
:term:`--mciter<--mc, --iter, --mciter>`, 
:term:`--samples<-m, --samples>`

-----

.. _user-guide-intro-plot:

Plotting
########

aka `plot_parser`
*****************

**for anything related to output plots**

Anything related to the plotting of results for *any* of the modules is in this parser. Its 
currently a little heavy on the :term:`echelle diagram` end because this part of the plot is
harder to hack, so we tried to make it as easily customizable as possible.

.. code-block::

      --all, --showall      plot background comparison figure
      -d, --show, --display
                            show output figures
      --cm str, --color str
                            Change colormap of ED, which is `binary` by default
      --cv float, --value float
                            Clip value multiplier to use for echelle diagram (ED).
                            Default is 3x the median, where clip_value == `3`.
      -y, --hey             plugin for Daniel Hey's echelle package **not
                            currently implemented**
      -i, --ie, --interpech
                            turn on the interpolation of the output ED
      --nox int, --nacross int
                            number of bins to use on the x-axis of the ED
                            (currently being tested)
      --noy str, --ndown str, --norders str
                            NEW!! Number of orders to plot pm how many orders to
                            shift (if ED is not centered)
      --npb int             NEW!! npb == "number per bin", which is option instead
                            of nox that uses the frequency resolution and spacing
                            to compute an appropriate bin size for the ED
      --se float, --smoothech float
                            Smooth ED using a box filter [in muHz]


:underlined:`Glossary terms` (in alphabetical order): 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:term:`--ce<--ce, --cm, --color>`, 
:term:`--cm<--ce, --cm, --color>`, 
:term:`--color<--ce, --cm, --color>`, 
:term:`--cv<--cv, --value>`, 
:term:`-d<-d, --show, --display>`, 
:term:`--display<-d, --show, --display>`, 
:term:`--hey<-y, --hey>`, 
:term:`-i<-i, --ie, --interpech>`, 
:term:`--ie<-i, --ie, --interpech>`, 
:term:`--interpech<-i, --ie, --interpech>`, 
:term:`--nox<--nox, --nacross>`, 
:term:`--nacross<--nox, --nacross>`, 
:term:`--ndown<--noy, --ndown, --norders>`, 
:term:`--norders<--noy, --ndown, --norders>`, 
:term:`--noy<--noy, --ndown, --norders>`, 
:term:`--npb`, 
:term:`--se<--se, --smoothech>`, 
:term:`--show<-d, --show, --display>`, 
:term:`--smoothech<--se, --smoothech>`, 
:term:`--value<--cv, --value>`, 
:term:`-y<-y, --hey>`

-----

On the next page, we will show applications for some of these options in command-line examples. 

We also have our :ref:`advanced usage<advanced>` page, which is specifically designed to 
show these in action by providing before and after references. You can also find
descriptions of certain commands available in the notebook tutorials. 
