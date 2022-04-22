.. role:: underlined
   :class: underlined

*************
Saved outputs
*************

We have shown examples applied to stars of various sample sizes, for different stellar types, 
of varying SNR detections, both single star
and many star we will not include 
any additional examples on this page but instead, list and describe each of the output files. 
Therefore we refer the reader to check out :ref:`this page <quickstart>`, the 
:ref:`comand-line examples <user-guide-cli-examples>` or the :ref:`notebook tutorials <user-guide-nb>`
if more examples are desired.

So while saving output files and figures is totally optional, we wanted to document them 
on this page since there's a lot of information to unpack. 

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
 #. :ref:`ID_BSPS.txt <library-output-files-bsps>`
 #. :ref:`ID_BDPS.txt <library-output-files-bdps>`
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

.. _library-output-files-bgcorrps:

2. `ID_BSPS.txt`
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

3. `ID_BDPS.txt`
*******************

**(all cases)**

Since we use both :term:`BCPS`, we figured we'd clear up the muddy waters here (but also
provide both copies to be used for their specific needs).



.. _library-output-files-estimates:

4. `estimates.csv`
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

If the monte-carlo sampling is used to estimate uncertainties, an optional feature is available
(i.e. :term:`--sampling`) to save the samples if desired.

**Note:** there is a *new* feature that saves and sets a random seed any time you are running
a target for the first time and therefore, you should be able to reproduce the samples in the
event that you forget to save the samples.

-----

.. _library-output-figures:

:underlined:`Figures`
#####################

Listed are all possible output figures for a given star (in alphabetical order):

 #. :ref:`background_only.png <library-output-figures-bgonly>`
 #. :ref:`bgmodel_fits.png <library-output-figures-bgfit>`
 #. :ref:`global_fit.png <library-output-figures-global>`
 #. :ref:`power_spectrum.png <library-output-figures-ps>`
 #. :ref:`samples.png <library-output-figures-samples>`
 #. :ref:`search_&_estimate.png <library-output-figures-estimates>`
 #. :ref:`time_series.png <library-output-figures-ts>`

and similar to the :ref:`file section <library-output-files>` above, we describe each in 
more detail below.

.. _library-output-figures-bgonly:

1. `background_only.png`
************************

**(rare cases)**

This figure is produced when the user is interested in determining the stellar background
model *only* and not the global asteroseismic properties. For example, detecting solar-like
oscillations in cool stars is extremely difficult to do but we can still characterize other
properties like their convective time scales, etc.

.. _library-output-figures-bgfit:

2. `bgmodel_fits.png`
*********************

**(optional cases)**

This figure is generated when the :term:`--show`

.. _library-output-figures-global:

3. `global_fit.png`
*******************

**(almost all cases)**

| **Top left:** Original time series. 
| **Top middle:** Original power spectrum (white), lightly smoothed power spectrum (red), and binned power spectrum (green). Blue lines show initial guesses of the fit to the granulation background. The grey region is excluded from the background fit based on the numax estimate provided to the module.
| **Top right:** Same as top middle but now showing the best fit background model (blue) and a heavily smoothed version of the power spectrum (yellow)
| **Center left:** Background corrected, heavily smoothed power spectrum (white). The blue line shows a Gaussian fit to the data (used to calculate numax_gaussian) and the red square is the peak of the smoothed, background corrected power excess (numax_smoothed).
| **Center:** Lightly smoothed, background corrected power spectrum centered on numax. 
| **Center right:** Autocorrelation function of the data in the center panel. The red dotted line shows the estimate Dnu value given the input numax value, and the red region shows the extracted ACF peak that will be used to measure Dnu. The yellow line shows the Gaussian weighting function used to define the red region.
| **Bottom left:** ACF peak extracted in the center right panel (white) and a Gaussian fit to that peak (green). The center of the Gaussian is the estimate of Dnu.
| **Bottom middle:** Echelle diagram of the background corrected power spectrum using the measured Dnu value.
| **Bottom right:** Echelle diagram collapsed along the frequency direction.


.. _library-output-figures-ps:

4. `power_spectrum.png`
***********************

**(special cases)**

This is still in its developmental stage but the idea is that one is supposed to "check"
a target before attempting to process the pipeline on any data. That means checking
the input data for sketchy looking features. For example, *Kepler* short-cadence data
has known artefacts present near the nyquist frequency for *Kepler* long-cadence data
(:math:`\sim 270 \mu \mathrm{Hz}`). In these cases, we have special frequency-domain tools
that are meant to help mitigate such things (e.g., see :mod:`pysyd.target.Target.remove_artefact`)


.. _library-output-figures-samples:

5. `samples.png`
****************

**(many cases)**

Each panel shows the samples of parameter estimates from Monte-Carlo simulations. Reported uncertainties on each parameter are calculated by taking the robust standard deviation of each distribution.


.. _library-output-figures-estimates:

6. `search_&_estimate.png`
**************************

**(most cases)**

| **Top left:** Original time series.  
| **Top middle:** Original power spectrum (white) and heavily smoothed power spectrum (green). The latter is used as an initial (crude) background fit to search for oscillations.  
| **Top right:** Power spectrum after correcting the crude background fit.  
| **Bottom left:** Frequency-resolved, collapsed autocorrelation function of the background-corrected power spectrum using a small step size. This step size is optimized for low-frequency oscillators. The green line is a Gaussian fit to the data, which provides the initial numax estimate.  
| **Bottom middle:** Same as bottom left but for the medium step size (optimized for subgiant stars).  
| **Bottom right:** Same as bottom left but for the large step size (optimized for main-sequence stars).


.. _library-output-figures-ts:

7. `time_series.png`
********************

**(special cases)**


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





