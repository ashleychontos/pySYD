
.. _cli/help:

Running help 
############

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
    optional arguments:
      -h, --help            show this help message and exit

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

As you are navigating this webpage, keep in mind that we have a special :ref:`glossary<cli/glossary>` for 
all our command-line options.

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

.. code-block::

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


.. _cli/help/data:

Initial data analyses
*********************

The following features are primarily related to the initial and final treatment of
data products, including information about the input data, how to process and save
the data as well as which modules to run.

.. code-block::

      -b, --bg, --background
                            Turn off the automated background fitting routine
      -d, --show, --display
                            Show output figures
      -g, --globe, --global
                            Do not estimate global asteroseismic parameters (i.e.
                            numax or dnu)
      -k, --kc, --kep_corr  Turn on the Kepler short-cadence artefact correction
                            routine
      --ofa n, --ofactual n
                            The oversampling factor (OF) of the input PS
      --ofn n, --ofnew n   The OF to be used for the first iteration
      -o, --over, --overwrite
                            Overwrite existing files with the same name/path
      -p, --par, --parallel
                            Use parallel processing for data analysis
      -s, --save            Do not save output figures and results.
      --star [star [star ...]], --stars [star [star ...]]
                            List of stars to process
      -t, --test            Extra verbose output for testing functionality
      -x, --ex, --excess    Turn off the find excess routine

**Glossary terms:** :term:`-s<-s, --save>`, :term:`--save<-s, --save>`

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
                            
**Glossary terms:** :term:`-a<-a, --ask>`, :term:`--ask<-a, --ask>`

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
