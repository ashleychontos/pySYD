*************
Basic example
*************

If you ran the ``pysyd setup`` was successfully executed, there should now be light curves and power spectra 
for three KIC stars in the **data/** directory. If so, then you are ready to test out the software!

We will start with some of the easiest examples, which have very high signal-to-noise (SNR) detections and then ease into
the more difficult detections and/or situations. 

Below are three examples of medium to high signal-to-noise (SNR) detections for stars of different evolutionary states. See the 
:ref:`examples/description` for more details about each panel in the output figures.

KIC 1435467
###########

KIC 1435467 is our least evolved example star, with numax ~1400 muHz. The following command:

::

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


runs KIC 1435467 using the default method, which first estimates numax before deriving any parameters. 
The reason for this is that the frequency region with power excess is masked out to better estimate 
background parameters.

Additional commands used in this example (and what they each mean):

 - ``--ux 5000`` is the upper frequency bound of the power spectrum used during the first module 
   (i.e. 'x' for excess, ``--lx`` would be the same but the lower bound for this module). These bounds  
   are used strictly for computational purposes and do not alter or change the power spectrum in any way.
 - ``--ie`` turns the bicubic interpolation on when plotting the \'echelle diagram. This is 
   particularly helpful for lower SNR examples like this. 
 - ``-dv`` == ``-d`` + ``-v`` -> single hashes are reserved for boolean arguments, which correspond to 
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


**For a breakdown of what each panel in each figure means, please see :ref:<examples/description> for more details.**
  
  
The derived parameters are saved to an output csv file but also printed at the end of the verbose output.
To quantify uncertainties in these parameters, we need to turn on the Monte Carlo sampling option (``--mc``) with::

::

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
bootstrapping technique and typically n=200 is more than sufficient. 

Parameter posteriors:

.. image:: figures/examples/1435467_samples.png
  :width: 680
  :alt: Parameter posteriors for KIC 1435467
  
  
.. note::

    The sampling results can be saved by using the boolean flag ``-m`` or ``--samples``,
    which will save the posteriors of the fitted parameters for later use. 
    
And just like that, you are now an asteroseismologist (if you were not one before)!
   
