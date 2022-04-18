.. role:: underlined
   :class: underlined

*************
Saved outputs
*************

While saving output files and figures are totally optional, we wanted to document them 
on this page since there's a lot of information to unpack there. 

Since we have already shown many examples for different stellar types, we will not include 
any additional examples on this page but instead, list and describe each of the output files. 
Therefore we refer the reader to check out :ref:`this page <quickstart>`, the 
:ref:`comand-line examples <user-guide-cli-examples>` or the :ref:`notebook tutorials <user-guide-nb>`
if more examples are desired.

.. _library-output:

Subdirectories are automatically created for each star that is processed. Based on the way
you use ``pySYD``, there are a number of different outputs which are saved by default. Here
we will list and describe them all.

We will reserve this page solely for saved outputs and hence, please see our :ref:`crashteroseismology <quickstart/crash>`
example if you'd like more information about the printed `verbose` output.

.. todo::

    While this is not yet implemented, our goal is to have a config file that will print the 
    verbose output to a file so that everything is reproducible. This will include which modules
    were ran, if any random seed was used and what the results were.

-----

.. _library-output-files:

:underlined:`Files`
###################

Listed are all the possible output files:

 #. :ref:`ID_PS.txt <library-output-files-ps>`
 #. :ref:`ID_bg_corr.txt <library-output-files-bgcorrps>`
 #. :ref:`estimates.csv <library-output-files-estimates>`
 #. :ref:`global.csv <library-output-files-global>`
 #. :ref:`samples.csv <library-output-files-samples>`

which we describe in more detail below, including the frequency and likely scenarios they 
arise from.

.. _library-output-files-ps:

1. `ID_PS.txt`
**************

**(special cases)**

This file is created in the case where *only* the time series data was provided for a target and
`pySYD` computed a power spectrum. This optional, extra step is important to make sure that
the power spectrum used through the analyzes is both normalized correctly and has the proper 
units -- this *ensures* accurate and reliable results. 

**Note:** unlike every other output file, this is instead saved to the data (or input directory)
so that the software can find it in later runs, which will save some time down the road. Of course
you can always copy and paste it to the specific star's result directory if you'd like.

.. important::

    For the saved power spectrum, the frequency array has units of :math:`\rm \mu Hz` and the
    power array is power density, which has units of :math:`\rm ppm^{2} \, \mu Hz^{-1}`. We 
    normalize the power spectrum according to Parseval's Theorem, which loosely means that the 
    fourier transform is unitary. This last bit is incredibly important for two main reasons,
    but both that tie to the noise properties in the power spectrum: 1) different instruments
    (e.g., *Kepler*, TESS) have different systematics and hence, noise properties, and 2) the 
    amplitude of the noise becomes smaller as your time series gets longer. Therefore when we 
    normalize the power spectrum, we can make direct comparisons between power spectra of not
    only different stars, but from different instruments as well!

.. _library-output-files-bgcorrps:

2. `ID_bg_corr.txt`
*******************

**(all cases)**

After the best-fit background model is selected and saved, the model is generated and then
subtracted from the power spectrum to remove all noise components present in a power spectrum.
Therefore, there should be little to no residual slope left in the power spectrum after this
step. This is saved as a basic text file in the star's output directory, where the first column 
is frequency (in :math:`\rm \mu Hz`) and the second column is power density, with units of 
:math:`\rm ppm^{2} \, \mu Hz^{-1}` (i.e. this file has the same units as the power spectrum).

In fact to take a step back, it might be helpful to understand the application and importance of the 
background-corrected power spectrum (:term:`BCPS`). The BCPS is used in subsequent steps such as
computing global parameters (:math:`\rm \nu_{max}` and :math:`\Delta\nu`) and for constructing
the :term:`echelle diagram`. Therefore, we thought it might be useful to have a copy of this!


.. _library-output-files-estimates:

3. `estimates.csv`
******************

**(most cases)**

By default, a module will run to estimate an initial value for the frequency corresponding to 
maximum power, or :math:`\rm \nu_{max}`. The module selects the trial with the highest 
signal-to-noise (SNR) and saves the comma-separated values for three basic variables
associated with the selected trial: :term:`numax`, :term:`dnu`, and the SNR. 

The file is saved to the star's output directory, where both numax and dnu have frequency
units in :math:`\rm \mu Hz` and the SNR is unitless. Remember, these are just estimates
though and adapted results should come from the other csv file called `global.csv`.

This module can be bypassed a few different ways, primarily by directly providing the estimate 
yourself. In the cases where this estimating routine is skipped, this file will not be saved.

**Note:** The numax estimate is *important* for the main fitting routine. 

.. _library-output-files-global:

4. `global.csv`
***************

**(all cases)**

.. _library-output-files-samples:

5. `samples.csv`
****************

**(special cases)**

If the monte-carlo sampling is used to calculate 
uncertainties, an additional figure will plot the posterior distributions for the estimated 
parameters. An optional feature (i.e. ``--samples``) is available to save the samples if desired. 
See :ref:`examples` for a guide on what the output plots are showing.

-----

.. _library-output-figures:

:underlined:`Figures`
#####################

Listed are all possible output figures for running a single star:

 #. :ref:`numax_estimates.png <library-output-figures-estimates>`
 #. :ref:`global_fit.png <library-output-figures-global>`
 #. :ref:`samples.png <library-output-figures-samples>`

and similar to the :ref:`file section <library-output-files>`, we describe each in more detail below.

.. _library-output-figures-estimates:

1. `numax_estimates.png`
************************

| **Top left:** Original time series.  
| **Top middle:** Original power spectrum (white) and heavily smoothed power spectrum (green). The latter is used as an initial (crude) background fit to search for oscillations.  
| **Top right:** Power spectrum after correcting the crude background fit.  
| **Bottom left:** Frequency-resolved, collapsed autocorrelation function of the background-corrected power spectrum using a small step size. This step size is optimized for low-frequency oscillators. The green line is a Gaussian fit to the data, which provides the initial numax estimate.  
| **Bottom middle:** Same as bottom left but for the medium step size (optimized for subgiant stars).  
| **Bottom right:** Same as bottom left but for the large step size (optimized for main-sequence stars).


.. _library-output-figures-global:

2. `global_fit.png`
*******************

| **Top left:** Original time series. 
| **Top middle:** Original power spectrum (white), lightly smoothed power spectrum (red), and binned power spectrum (green). Blue lines show initial guesses of the fit to the granulation background. The grey region is excluded from the background fit based on the numax estimate provided to the module.
| **Top right:** Same as top middle but now showing the best fit background model (blue) and a heavily smoothed version of the power spectrum (yellow)
| **Center left:** Background corrected, heavily smoothed power spectrum (white). The blue line shows a Gaussian fit to the data (used to calculate numax_gaussian) and the red square is the peak of the smoothed, background corrected power excess (numax_smoothed).
| **Center:** Lightly smoothed, background corrected power spectrum centered on numax. 
| **Center right:** Autocorrelation function of the data in the center panel. The red dotted line shows the estimate Dnu value given the input numax value, and the red region shows the extracted ACF peak that will be used to measure Dnu. The yellow line shows the Gaussian weighting function used to define the red region.
| **Bottom left:** ACF peak extracted in the center right panel (white) and a Gaussian fit to that peak (green). The center of the Gaussian is the estimate of Dnu.
| **Bottom middle:** Echelle diagram of the background corrected power spectrum using the measured Dnu value.
| **Bottom right:** Echelle diagram collapsed along the frequency direction.


.. _library-output-figures-samples:

3. `samples.png`
****************

Each panel shows the samples of parameter estimates from Monte-Carlo simulations. Reported uncertainties on each parameter are calculated by taking the robust standard deviation of each distribution.

Takeaway
########

As we've said many times before, the software is optimized for running an ensemble of stars. 
Therefore, the utility function ``pysyd.utils.scrape_output`` will automatically concatenate the 
results for each of the main modules into a single csv in the parent results directory so that
it's easy to find and compare.

.. _library-output-api:

API
###

.. automodule:: pysyd.plots
   :members:

**********
What next?
**********

You may be asking yourself, well what do I do with this information? (and that is a totally
valid question to be asking)





