.. _user-guide-cli-help:

*******************
Introduction & help
*******************

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

-----

CLI help
########

To give you a glimpse into the insanely large amount of available options, open up a terminal
window and enter the help command for the main pipeline execution (`run` aka :mod:`pysyd.pipeline.run`), 
since this mode inherits all command-line parsers. 

.. code-block::

    $ pysyd run --help
    
    usage: pySYD run [-h] [--file str] [--in str] [--infdir str] [--info str]
                     [--out str] [--gap int] [-k] [--le [float [float ...]]] [-n]
                     [--of int] [-o] [-s] [--star [str [str ...]]]
                     [--ue [float [float ...]]] [-x] [-a] [--bin float] [--bm str]
                     [-e] [--lx [float [float ...]]] [--step float] [--trials int]
                     [--sw float] [--ux [float [float ...]]] [--all] [--basis str] 
                     [-b] [--bf float] [-f] [--iw float] [--laws int]
                     [--lb [float [float ...]]] [--metric str] [--rms int]
                     [--ub [float [float ...]]] [--dnu [float [float ...]]]
                     [--ew float] [-g] [--lp [float [float ...]]] [--method str]
                     [--numax [float [float ...]]] [--peak int] [--sm float]
                     [--sp float] [--thresh float] [--up [float [float ...]]]
                     [--ce str] [--cv float] [-y] [-i] [--nox int] [--noy int]
                     [--se float] [--mc int] [-m] [--cli] [-d] [-v]
   
    optional arguments:
      -h, --help            show this help message and exit

This was actually just a teaser! 

If you ran it from your end, you probably noticed an output that was a factor of ~5-10 longer! 
It may seem like an overwhelming amount but do not fret, this is for good reason -- and that's 
to make your asteroseismic experience as customized as possible.

Currently `pySYD` has four parsers: the `parent_parser` for high-level functionality, the
`data_parser` for anything related to data loading and manipulating, the `main_parser` for
everything related to the core analyses, and the `plot_parser` for (yes, you guessed it!)
plotting. In fact, the main parser is so large that comprises four subgroups, each related to
the corresponding steps in the main pipeline execution. **BTW** see :ref:`here <library-pipeline-modes>` 
for more information on which parsers a given pipeline mode inherits.

 - :ref:`parent parser <user-guide-intro-high>`
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

.. _user-guide-intro-high:

Parent parser
#############

**for all your high-level functionality needs**

All `pySYD` modes inherent the `parent_parser` and therefore, mostly pertains to paths and
how you choose to run the software (i.e. save files and if so, whether or not to overwrite 
old files with the same extension, etc.) 

.. code-block::

   High-level functions:
     --in str, --input str, --inpdir str
                           Input directory
     --infdir str          Path to relevant pySYD information
     --out str, --outdir str, --output str
                           Output directory
     -o, --overwrite       Overwrite existing files with the same name/path
     -s, --save            Do not save output figures and results
     --cli                 Running from command line (this should not be touched)
     -v, --verbose         Turn off verbose output

**Glossary terms:** 
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

Data parser
###########

The following features are primarily related to the input data and when applicable, what 
tools to apply to the data. All data manipulation relevant in this step happens *prior*
to any pipeline analysis. **Currently this is mostly frequency-domain tools but we are 
working on implementing some time-domain tools too!**

.. code-block::

   Data analyses:
     --file str, --list str, --todo str
                           List of stars to process
     --info str, --information str
                           List of stellar parameters and options
     --star [str [str ...]], --stars [str [str ...]]
                           List of stars to process
     --gap int, --gaps int
                           What constitutes a time series 'gap' (i.e. n x the
                           cadence)
     -x, --stitch, --stitching
                           Correct for large gaps in time series data by
                           'stitching' the light curve
     -k, --kc, --kepcorr   Turn on the Kepler short-cadence artefact correction
                           routine
     -n, --notch           Use notching technique to reduce effects from mixed
                           modes (not fully functional, creates weirds effects
                           for higher SNR cases)
     --of int, --over int, --oversample int
                           The oversampling factor (OF) of the input power
                           spectrum
     --dnu [float [float ...]]
                           Brute force method to provide value for dnu
     --le [float [float ...]], --lowere [float [float ...]]
                           Lower frequency limit of folded PS to whiten mixed
                           modes
     --ue [float [float ...]], --uppere [float [float ...]]
                           Upper frequency limit of folded PS to whiten mixed
                           modes

**Glossary terms** (alphabetical order): 
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

.. _user-guide-intro-est:

Core functions aka `main_parser`
################################

aka `main_parser`

Identify & estimate
###################

The following options are relevant for the first, optional module that is designed to search
for power excess due to solar-like oscillations and estimate rough starting points for its
main properties.

.. code-block::

   Estimate parameters:
     -a, --ask             Ask which trial to use
     --bin float, --binning float
                           Binning interval for PS (in muHz)
     --bm str, --mode str, --bmode str
                           Binning mode
     -e, --est, --estimate Turn off the optional module that estimates numax
     --lx [float [float ...]], --lowerx [float [float ...]]
                           Lower frequency limit of PS
     --step float, --steps float
     --trials int, --ntrials int
     --sw float, --smoothwidth float
                           Box filter width [in muHz] for smoothing the PS
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

-----

.. _user-guide-intro-bg:

Background fit
##############

Below is a complete list of parameters relevant to the background-fitting routine:

.. code-block::

   Background fits:
     -b, --bg, --background
                           Turn off the routine that determines the stellar
                           background contribution
     --basis str           Which basis to use for background fit (i.e. 'a_b',
                           'pgran_tau', 'tau_sigma'), *** NOT implemented yet ***
     --iw float, --indwidth float
                           Width of binning for PS [in muHz]
     --bf float, --box float, --boxfilter float
                           Box filter width [in muHz] for plotting the PS
     --rms int, --nrms int
                           Number of points to estimate the amplitude of red-
                           noise component(s)
     --laws int, --nlaws int
                           Force number of red-noise component(s)
     --metric str          Which model metric to use, choices=['bic','aic']
     --lb [float [float ...]], --lowerb [float [float ...]]
                           Lower frequency limit of PS
     --ub [float [float ...]], --upperb [float [float ...]]
                           Upper frequency limit of PS
     -w, --wn, --fixwn     Fix the white noise level

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

-----

.. _user-guide-intro-globe:

Global fit
##########

All of the following are related to deriving global asteroseismic parameters, :term:`numax`
(:math:`\rm \nu_{max}`) and :term:`dnu` (:math:`\Delta\nu`). 

.. code-block::

   Global parameters:
     -g, --globe, --global
                           Turn off the main module that estimates global
                           properties
     --numax [float [float ...]]
                           Skip find excess module and force numax
     --lp [float [float ...]], --lowerp [float [float ...]]
                           Lower frequency limit for zoomed in PS
     --up [float [float ...]], --upperp [float [float ...]]
                           Upper frequency limit for zoomed in PS
     --ew float, --exwidth float
                           Fractional value of width to use for power excess,
                           where width is computed using a solar scaling
                           relation
     --sm float, --smpar float
                           Value of smoothing parameter to estimate smoothed
                           numax (typically between 1-4) **developer use only**
     --sp float, --smoothps float
                           Box filter width [in muHz] of PS for ACF
     --peak int, --peaks int, --npeaks int
                           Number of peaks to fit in the ACF
     --thresh float, --threshold float
                           Fractional value of FWHM to use for ACF
     --dnu [value [value ...]]
                           Brute force method to provide value for dnu
     --peak n, --peaks n, --npeaks n
                           Number of peaks to fit in the ACF


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

-----

.. _user-guide-intro-mc:

Sampling
########

All CLI options relevant for the Monte-Carlo sampling in order to estimate uncertainties:

.. code-block::

   Estimate uncertainties:
     --mc int, --iter int, --mciter int
                           Number of Monte-Carlo iterations
     -m, --samples         Save samples from the Monte-Carlo sampling

**Glossary terms** (alphabetical order): 
:term:`--iter<--mc, --iter, --mciter>`, 
:term:`-m<-m, --samples>`, 
:term:`--mc<--mc, --iter, --mciter>`, 
:term:`--mciter<--mc, --iter, --mciter>`, 
:term:`--samples<-m, --samples>`

-----

.. _user-guide-intro-plot:

Plot parser
###########

aka `plot_parser`

Anything related to the plotting of results for *any* of the modules is in this parser. Its 
currently a little heavy on the :term:`echelle diagram` end because this part of the plot is
harder to hack, so we tried to make it as easily customizable as possible.

.. code-block::

   Plotting:
     -d, --show, --display
                           display output figures in real time
     --all, --showall      make background comparison figure
     --ce str, --cm str, --color str
                           colormap of echelle diagram (default=`binary`)
     --cv float, --value float
                           clip value multiplier to use for ED, which is currently 3x the median
     -y, --hey             use Daniel Hey's plugin for echelle **not currently implemented**
     -i, --ie, --interpech
                           Turn on the interpolation of the output ED
     --nox int, --nacross int
                           number of bins to use on the x-axis of the ED
     --noy int, --ndown int, --norders int
                           number of orders to plot on the ED y-axis
     --npb int             related to `--nox` but uses information about the spacing and resolution 
                           to estimate a reasonable number of bins to use for the x-axis of the ED
     --se float, --smoothech float
                           smooth ED using a box filter [in muHz]

**Glossary terms** (alphabetical order): 
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

In the next topic, we will show some examples using these options.

We have additional examples for some of these options in action to in :ref:`advanced usage<advanced>` 
and also have included a brief :ref:`tutorial` below that describes some of these commands.
