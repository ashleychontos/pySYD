.. _examples:


Examples
########

If ``pysyd setup`` was successfully executed, there should now be light curves and power spectra 
for three KIC stars in the **data/** directory. If so, then you are ready to test out the software!

====================

High SNR Examples
===================

Below are examples of medium to high signal-to-noise (SNR) detections in three stars of different evolutionary states. The first example includes a brief description of the output plots.

KIC 1435467
*************

KIC 1435467 is our least evolved example star, with numax ~1400 muHz. The following command::


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


runs KIC 1435467 using the default method, which first runs ``find_excess`` followed by ``fit_global``.

Additional commands used in this example (and what they each mean):
 - ``--ux 5000`` is the upper frequency bound of the power spectrum used during the first module 
   (i.e. 'x' for excess, ``--lx`` would be the same but the lower bound for this module). These bounds  
   are used strictly for computational purposes and do not alter or change the power spectrum in any way.
 - ``--ie`` turns the bicubic interpolation on when plotting the \'echelle diagram. This is 
   particularly helpful for lower SNR examples like this. 
 - ``-dv`` == `-d` + `-v` -> single hashes are reserved for boolean arguments, which correspond to 
   ``display`` and ``verbose``, respectively. Since ``pySYD`` is optimized for many stars, both of these
   options are ``False`` by default.
   
As you can read in the text output, the example started with n=2 Harvey-like components but reduced to 1 
based on the BIC statistic. 

The first, optional routine that estimates numax creates the following output figure:

.. image:: figures/examples/1435467_numax.png
  :width: 680
  :alt: Numax estimates for KIC 1435467

The derived parameters from the global fit are summarized below:

.. image:: figures/examples/1435467_global.png
  :width: 680
  :alt: Global fit of KIC 1435467


.. note::

    For a breakdown of what each panel in each figure means, please see ref for more details.
  
  
The derived parameters are saved to an output csv file but also printed at the end of the verbose output.
To quantify uncertainties in these parameters, we need to turn on the Monte Carlo sampling option (``--mc``) with::


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
    
 

where the first 2/3 of the output is (and should be) identical to the first example. By default, 
``--mc == 1`` since you should always check your results first before running ``pySYD`` for
several iterations! The method used to derive the uncertainties is similar to a 
bootstrapping technique and typically n=200 is more than sufficient. You may also use the ``--samples``
option if you would like to save the posteriors of the parameters for later use.

The Monte Carlo ``sampling`` results:

.. image:: figures/examples/1435467_samples.png
  :width: 680
  :alt: Parameter posteriors for KIC 1435467.

====================

KIC 2309595
*************

KIC 2309595 is a subgiant, with numax ~650 muHz.

``find_excess`` results:

.. image:: figures/examples/2309595_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 2309595.

``fit_background`` results:

.. image:: figures/examples/2309595_background.png
  :width: 600
  :alt: Fit background output plot for KIC 2309595.

``sampling`` results:

.. image:: figures/examples/2309595_samples.png
  :width: 600
  :alt: Distributions of Monte-Carlo samples for KIC 2309595.

====================

KIC 11618103
***************

KIC 11618103 is an evolved RGB star, with numax of ~100 muHz.

``find_excess`` results:

.. image:: figures/examples/11618103_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 11618103.

``fit_background`` results:

.. image:: figures/examples/11618103_background.png
  :width: 600
  :alt: Fit background output plot for KIC 11618103.

``sampling`` results:

.. image:: figures/examples/11618103_samples.png
  :width: 600
  :alt: Distributions of Monte-Carlo samples for KIC 11618103.


====================

Low SNR Examples
=================

KIC 8801316
**************

KIC 8801316 is a subgiant, with a numax ~1100 muHz. Although the data has low signal-to-noise ratio, this would be classified as a detection due to the following reasons:

- there is a clear power excess as seen in panel 3
- the power excess has a Gaussian shape as seen in panel 5 corresponding to the solar-like oscillations
- the autocorrelation function (ACF) in panel 6 show periodic peaks
- the echelle diagram in panel 8 shows the ridges, albeit faintly


``find_excess`` results:

.. image:: figures/examples/8801316_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 8801316.

``fit_background`` results:

.. image:: figures/examples/8801316_background.png
  :width: 600
  :alt: Fit background output plot for KIC 8801316.

``sampling`` results:

.. image:: figures/examples/8801316_samples.png
  :width: 600
  :alt: Distributions of Monte-Carlo samples for KIC 8801316.


====================

Non-detections
================

KIC 6278992
*************

KIC 6278992 is a main-sequence star with no solar-like oscillations.

``find_excess`` results:

.. image:: figures/examples/6278992_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 6278992.

``fit_background`` results:

.. image:: figures/examples/6278992_background.png
  :width: 600
  :alt: Fit background output plot for KIC 6278992.

``sampling`` results:

.. image:: figures/examples/6278992_samples.png
  :width: 600
  :alt: Distributions of Monte-Carlo samples for KIC 6278992.


====================

Figure Descriptions
====================

Estimating numax:
******************

| **Top left:** Original time series.  
| **Top middle:** Original power spectrum (white) and heavily smoothed power spectrum (green). The latter is used as an initial (crude) background fit to search for oscillations.  
| **Top right:** Power spectrum after correcting the crude background fit.  
| **Bottom left:** Frequency-resolved, collapsed autocorrelation function of the background-corrected power spectrum using a small step size. This step size is optimized for low-frequency oscillators. The green line is a Gaussian fit to the data, which provides the initial numax estimate.  
| **Bottom middle:** Same as bottom left but for the medium step size (optimized for subgiant stars).  
| **Bottom right:** Same as bottom left but for the large step size (optimized for main-sequence stars).
|

Global fit:
**************

| **Top left:** Original time series. 
| **Top middle:** Original power spectrum (white), lightly smoothed power spectrum (red), and binned power spectrum (green). Blue lines show initial guesses of the fit to the granulation background. The grey region is excluded from the background fit based on the numax estimate provided to the module.
| **Top right:** Same as top middle but now showing the best fit background model (blue) and a heavily smoothed version of the power spectrum (yellow)
| **Center left:** Background corrected, heavily smoothed power spectrum (white). The blue line shows a Gaussian fit to the data (used to calculate numax_gaussian) and the red square is the peak of the smoothed, background corrected power excess (numax_smoothed).
| **Center:** Lightly smoothed, background corrected power spectrum centered on numax. 
| **Center right:** Autocorrelation function of the data in the center panel. The red dotted line shows the estimate Dnu value given the input numax value, and the red region shows the extracted ACF peak that will be used to measure Dnu. The yellow line shows the Gaussian weighting function used to define the red region.
| **Bottom left:** ACF peak extracted in the center right panel (white) and a Gaussian fit to that peak (green). The center of the Gaussian is the estimate of Dnu.
| **Bottom middle:** Echelle diagram of the background corrected power spectrum using the measured Dnu value.
| **Bottom right:** Echelle diagram collapsed along the frequency direction.
|

Sampling:
***********

Each panel shows the samples of parameter estimates from Monte-Carlo simulations. Reported uncertainties on each parameter are calculated by taking the robust standard deviation of each distribution.


Multiple Stars
=================

There is a parallel processing option included in the software, which is helpful for
running many stars. This can be accessed through the following command:

.. code-block::

    $ pysyd parallel (-nthreads 15 -list path_to_star_list.txt)

For parallel processing, ``pySYD`` will divide and group the list of stars based on the number of threads available. 
By default, ``args.n_threads = 0`` but can be specified by using the command line option. If parallelization is preferred
but the ``-nthreads`` option is not used, ``pySYD`` will use ``multiprocessing.cpu_count()`` to determine the number of
cpus available for the local operating system and set the number of threads to ``mulitprocessing.cpu_count()-1``.

.. note::

    Remember that by default, the stars to be processed (i.e. todo) will read in from **info/todo.txt**
    if no ``-list`` or ``-todo`` paths are provided.
