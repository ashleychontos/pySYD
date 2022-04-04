****************
Pipeline results
****************

As we've said many times before, the software is optimized for running an ensemble of stars. 
Therefore, the utility function ``pysyd.utils.scrape_output`` will automatically concatenate the 
results for each of the main modules into a single csv in the parent results directory so that
it's easy to find and compare.

Introduction
############

Although it is optional, subdirectories are automatically created for each star that is processed.
A single star will yield one summary figure (png) and one data product (csv) for each of the two
main modules, i.e. the estimation and derivation of parameters. 

Additionally, the background-corrected power spectrum is saved as a basic text file for later use.
 for a total of 5 output files. If the monte-carlo sampling is used to calculate 
uncertainties, an additional figure will plot the posterior distributions for the estimated 
parameters. An optional feature (i.e. ``--samples``) is available to save the samples if desired. 
See :ref:`examples` for a guide on what the output plots are showing.

Imports
#######

Usage
#####

Examples
########

Output
######

While this is not yet implemented, our goal is to have a config file that will print the 
verbose output to a file so that everything is reproducible. This will include which modules
were ran, if any random seed was used and what the results were.

Printed
*******

Files
*****

Required
++++++++

Optional
++++++++

Figures
#######

The are three main figures 

Figure descriptions
*******************

Initial estimates
+++++++++++++++++

| **Top left:** Original time series.  
| **Top middle:** Original power spectrum (white) and heavily smoothed power spectrum (green). The latter is used as an initial (crude) background fit to search for oscillations.  
| **Top right:** Power spectrum after correcting the crude background fit.  
| **Bottom left:** Frequency-resolved, collapsed autocorrelation function of the background-corrected power spectrum using a small step size. This step size is optimized for low-frequency oscillators. The green line is a Gaussian fit to the data, which provides the initial numax estimate.  
| **Bottom middle:** Same as bottom left but for the medium step size (optimized for subgiant stars).  
| **Bottom right:** Same as bottom left but for the large step size (optimized for main-sequence stars).


Global fit
++++++++++

| **Top left:** Original time series. 
| **Top middle:** Original power spectrum (white), lightly smoothed power spectrum (red), and binned power spectrum (green). Blue lines show initial guesses of the fit to the granulation background. The grey region is excluded from the background fit based on the numax estimate provided to the module.
| **Top right:** Same as top middle but now showing the best fit background model (blue) and a heavily smoothed version of the power spectrum (yellow)
| **Center left:** Background corrected, heavily smoothed power spectrum (white). The blue line shows a Gaussian fit to the data (used to calculate numax_gaussian) and the red square is the peak of the smoothed, background corrected power excess (numax_smoothed).
| **Center:** Lightly smoothed, background corrected power spectrum centered on numax. 
| **Center right:** Autocorrelation function of the data in the center panel. The red dotted line shows the estimate Dnu value given the input numax value, and the red region shows the extracted ACF peak that will be used to measure Dnu. The yellow line shows the Gaussian weighting function used to define the red region.
| **Bottom left:** ACF peak extracted in the center right panel (white) and a Gaussian fit to that peak (green). The center of the Gaussian is the estimate of Dnu.
| **Bottom middle:** Echelle diagram of the background corrected power spectrum using the measured Dnu value.
| **Bottom right:** Echelle diagram collapsed along the frequency direction.


Parameter posteriors
++++++++++++++++++++

Each panel shows the samples of parameter estimates from Monte-Carlo simulations. Reported uncertainties on each parameter are calculated by taking the robust standard deviation of each distribution.


API
###

.. automodule:: pysyd.plots
   :members:

**********
What next?
**********

You may be asking yourself, well what do I do with this information? That's a perfectly reasonable
question to be asking!






