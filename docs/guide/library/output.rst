*******
Outputs
*******

Required
########

So I meannnn technically no outputs are required but we hope that you would like to, at the very
least, see some of the figures. Those are hands down the coolest (but also, lots of information
to unpack)!

Optional
########




Output
********

Results/
++++++++++

Subdirectories are automatically created for each individually processed star.
Results for each of the two main ``pySYD`` modules (``find_excess`` and ``fit_background``) 
will be concatenated into a single csv in the upper-level results directory, which is
helpful when running many stars.

A single star will yield one summary figure (png) and one data product (csv) for each of the two
main modules. Additionally, the background-corrected (divided) power spectrum is saved as a basic
text file, for a total of 5 output files. If the monte-carlo sampling is used to calculate 
uncertainties, an additional figure will plot the posterior distributions for the estimated 
parameters. An optional feature (i.e. ``--samples``) is available to save the samples if desired. 
See :ref:`examples` for a guide on what the output plots are showing.



*****************
Plotting routines
*****************

Introduction
############

Imports
#######

Usage
#####

Examples
########

``pysyd.plots`` API
###################

.. automodule:: pysyd.plots
   :members:

    

====================

KIC 2309595
*************

KIC 2309595 is a subgiant, with numax ~650 muHz.

Estimating numax:

.. image:: figures/examples/2309595_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 2309595.

The global fit:

.. image:: figures/examples/2309595_background.png
  :width: 600
  :alt: Fit background output plot for KIC 2309595.

Sampling results:

.. image:: figures/examples/2309595_samples.png
  :width: 600
  :alt: Distributions of Monte-Carlo samples for KIC 2309595.

====================


KIC 11618103
***************

KIC 11618103 is our most evolved example, an RGB star with numax of ~100 muHz.

Estimate for numax:

.. image:: figures/examples/11618103_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 11618103.

Global fit:

.. image:: figures/examples/11618103_background.png
  :width: 600
  :alt: Fit background output plot for KIC 11618103.

Sampling results:

.. image:: figures/examples/11618103_samples.png
  :width: 600
  :alt: Distributions of Monte-Carlo samples for KIC 11618103.

====================

.. _examples/medium:


Low SNR Examples
=================

KIC 8801316
**************

KIC 8801316 is a subgiant with a numax ~1100 muHz, shown in the figures below. 

Numax estimate:

.. image:: figures/examples/8801316_excess.png
  :width: 680
  :alt: Numax estimate KIC 8801316.

Derived parameters:

.. image:: figures/examples/8801316_background.png
  :width: 680
  :alt: Global fit for KIC 8801316.

This would be classified as a detection despite the low SNR due to the following reasons:

- there is a clear power excess as seen in panel 3
- the power excess has a Gaussian shape as seen in panel 5 corresponding to the solar-like oscillations
- the autocorrelation function (ACF) in panel 6 show periodic peaks
- the echelle diagram in panel 8 shows the ridges, albeit faintly


====================

.. _examples/hard:

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
.. _examples/description:


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



