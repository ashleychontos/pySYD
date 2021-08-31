.. _quickstart:

##################
Getting Started
##################

Jump down to :ref:`summary` to get asteroseismic parameters for a star in less than one minute!

.. _installation:

Installation
***************

Install ``pysyd`` using pip:

.. code-block::

    $ pip install pysyd

The ``pysyd`` binary should have been automatically placed in your system's path by the
``pip`` command. If your system can not find the ``pysyd`` executable, ``cd`` into the 
top-level ``pysyd`` directory and try running the following command:

.. code-block::

    $ python setup.py install

You may test your installation by using ``pysyd --help`` to see the available commands or modes:

.. code-block::
		
    $ pysyd --help
    usage: pySYD [-h] [-version] {load,parallel,run,setup,test} ...

    pySYD: Automated Extraction of Global Asteroseismic Parameters

    optional arguments:
      -h, --help            show this help message and exit
      -version, --version   Print version number and exit.

    pySYD modes:
      {load,parallel,run,setup,test}
        load                Load in data for a given target
        parallel            Run pySYD in parallel
        run                 Run the main pySYD pipeline
        setup               Easy setup of relevant directories and files
        test                Test different utilities (currently under development)
	
	
A majority of the data analyses and subsequent tools are used in the most common mode, ``run``,
and therefore, using the following command:

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


will display a very long but very healthy list of available options.


Setting Up
**********

The easiest way to start using the ``pySYD`` software is by running our setup feature
from a convenient directory:

.. code-block::

    $ pysyd setup

This command will create `data`, `info`, and `results` directories in the current working 
directory, if they don't already exist. Setup will also download two information files: 
**info/todo.txt** and **info/star_info.csv**. See :ref:`overview` for more information on 
what purposes these files serve. Additionally, three example stars 
from the `source code <https://github.com/ashleychontos/pySYD>`_ are included (see :ref:`examples`).

The optional verbose command can be called with the setup feature:

.. code-block::

    $ pysyd setup --verbose

This will print the absolute paths of all directories that are created during setup.


Example Fit
***********

If you ran the setup feature, there are data for three example stars provided: 1435467 (the least evolved), 
2309595 (~SG), and 11618103 (RGB). To run a single star, execute the main script with the following command:


.. code-block::

    $ pysyd run --star 1435467 -dv
    
    
    ------------------------------------------------------
    Target: 1435467
    ------------------------------------------------------
    # LIGHT CURVE: 37919 lines of data read
    # Time series cadence: 59 seconds
    # POWER SPECTRUM: 99518 lines of data read
    # PS is oversampled by a factor of 5
    # PS resolution: 0.426868 muHz
    ------------------------------------------------------
    Estimating numax:
    PS binned to 189 datapoints
    Numax estimate 1: 1430.02 +/- 72.61
    S/N: 2.43
    Numax estimate 2: 1479.46 +/- 60.64
    S/N: 4.87
    Numax estimate 3: 1447.42 +/- 93.31
    S/N: 13.72
    Selecting model 3
    ------------------------------------------------------
    Determining background model:
    PS binned to 419 data points
    Comparing 6 different models:
    Model 0: 0 Harvey-like component(s) + white noise fixed
    Model 1: 0 Harvey-like component(s) + white noise term
    Model 2: 1 Harvey-like component(s) + white noise fixed
    Model 3: 1 Harvey-like component(s) + white noise term
    Model 4: 2 Harvey-like component(s) + white noise fixed
    Model 5: 2 Harvey-like component(s) + white noise term
    Based on BIC statistic: model 2
     **background-corrected PS saved**
    ------------------------------------------------------
    Output parameters:
    tau_1: 233.71 s
    sigma_1: 87.45 ppm
    numax_smooth: 1299.56 muHz
    A_smooth: 1.75 ppm^2/muHz
    numax_gauss: 1345.03 muHz
    A_gauss: 1.49 ppm^2/muHz
    FWHM: 291.32 muHz
    dnu: 70.63 muHz
    ------------------------------------------------------
     - displaying figures
     - press RETURN to exit
     - combining results into single csv file
    ------------------------------------------------------
    
    

Here ``-dv`` means options ``-d`` and ``-v``, which stand for display (figures) and verbose output, 
respectively. Since ``pySYD`` is optimized for running multiple stars, both of these are ``False`` 
by default. We recommend using them for the example, since they are 
helpful to see how the pipeline processes targets. 


The above command should have yielded the following output figures:


.. image:: figures/quickstart/1435467_numax.png
  :width: 680
  :alt: Estimate of numax for KIC 1435467


from the first optional module that estimates numax (using 3 different trials).
All parameter derivations are done in the global fit, the results which are 
encapsulated in this figure:


.. image:: figures/quickstart/1435467_global.png
  :width: 680
  :alt: Global fit for KIC 1435467


To estimate uncertainties in the derived parameters, set ``--mc`` to a number sufficient for bootstrap sampling. In the previous 
example, ``--mc`` was not specified and is 1 by default (for 1 iteration). Below shows the same example with the
sampling enabled, including the verbose output you should see if your software was installed successfully.


.. code-block::

    $ pysyd run -star 1435467 -dv --mc 200
        
    
    ------------------------------------------------------
    Target: 1435467
    ------------------------------------------------------
    # LIGHT CURVE: 37919 lines of data read
    # Time series cadence: 59 seconds
    # POWER SPECTRUM: 99518 lines of data read
    # PS is oversampled by a factor of 5
    # PS resolution: 0.426868 muHz
    ------------------------------------------------------
    Estimating numax:
    PS binned to 189 datapoints
    Numax estimate 1: 1430.02 +/- 72.61
    S/N: 2.43
    Numax estimate 2: 1479.46 +/- 60.64
    S/N: 4.87
    Numax estimate 3: 1447.42 +/- 93.31
    S/N: 13.72
    Selecting model 3
    ------------------------------------------------------
    Determining background model:
    PS binned to 419 data points
    Comparing 6 different models:
    Model 0: 0 Harvey-like component(s) + white noise fixed
    Model 1: 0 Harvey-like component(s) + white noise term
    Model 2: 1 Harvey-like component(s) + white noise fixed
    Model 3: 1 Harvey-like component(s) + white noise term
    Model 4: 2 Harvey-like component(s) + white noise fixed
    Model 5: 2 Harvey-like component(s) + white noise term
    Based on BIC statistic: model 2
     **background-corrected PS saved**
    ------------------------------------------------------
    Running sampling routine:
    100%|█████████████████████████████████████████████████████████████████| 200/200 [00:17<00:00, 11.13it/s]
    
    Output parameters:
    tau_1: 233.71 +/- 20.50 s
    sigma_1: 87.45 +/- 3.18 ppm
    numax_smooth: 1299.56 +/- 56.64 muHz
    A_smooth: 1.75 +/- 0.24 ppm^2/muHz
    numax_gauss: 1345.03 +/- 40.66 muHz
    A_gauss: 1.49 +/- 0.28 ppm^2/muHz
    FWHM: 291.32 +/- 63.62 muHz
    dnu: 70.63 +/- 0.74 muHz
    ------------------------------------------------------
     - displaying figures
     - press RETURN to exit
     - combining results into single csv file
    ------------------------------------------------------
    
    
An additional output figure is created with the sampling routine,
displaying the posteriors for the fitted parameters:


.. image:: figures/quickstart/1435467_samples.png
  :width: 680
  :alt: Posteriors for KIC 1435467


Please visit :ref:`this page<examples>` for additional examples, including how to interpret the results
as well as descriptions about what the plots are showing.


.. _summary:

Quickstart
************

.. _compound::

To determine asteroseismic parameters for a single star in roughly sixty seconds, execute 
the following commands:

.. code-block::

    $ mkdir ~/path_to_put_pysyd_stuff
    $ cd ~/path_to_put_pysyd_stuff
    $ pip install pysyd
    $ pysyd setup
    $ pysyd run --star 1435467 -dv --mc 200
        
... and if you weren't one already, you are now an asteroseismologist!
    
