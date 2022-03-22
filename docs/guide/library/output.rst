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


==========================


How It Works
===============

When running the software, initialization of ``pySYD`` via command line will look in the following paths:

- ``TODODIR`` : '~/path_to_put_pysyd_stuff/info/todo.txt'
- ``INFODIR`` : '~/path_to_put_pysyd_stuff/info/star_info.csv'
- ``INPDIR`` : '~/path_to_put_pysyd_stuff/data'
- ``OUTDIR`` : '~/path_to_put_pysyd_stuff/results'

which by default, is the absolute path of the current working directory (or however you choose to set it up). All of these paths should be ready to go
if you followed the suggestions in :ref:`structure` or used our ``setup`` feature.

A ``pySYD`` pipeline ``Target`` class object has two main function calls:

#. The first module :
    * **Summary:** a crude, quick way to identify the power excess due to solar-like oscillations
    * This uses a heavy smoothing filter to divide out the background and then implements a frequency-resolved, collapsed 
      autocorrelation function (ACF) using 3 different ``box`` sizes
    * The main purpose for this first module is to provide a good starting point for the
      second module. The output from this routine provides a rough estimate for numax, which is translated 
      into a frequency range in the power spectrum that is believed to exhibit characteristics of p-mode
      oscillations
#. The second module : 
    * **Summary:** performs a more rigorous analysis to determine both the stellar background contribution
      as well as the global asteroseismic parameters.
    * Given the frequency range determined by the first module, this region is masked out to model 
      the white- and red-noise contributions present in the power spectrum. The fitting procedure will
      test a series of models and select the best-fit stellar background model based on the BIC.
    * The power spectrum is corrected by dividing out this contribution, which also saves as an output text file.
    * Now that the background has been removed, the global parameters can be more accurately estimated. Numax is
      estimated by using a smoothing filter, where the peak of the heavily smoothed, background-corrected power
      spectrum is the first and the second fits a Gaussian to this same power spectrum. The smoothed numax has 
      typically been adopted as the default numax value reported in the literature since it makes no assumptions 
      about the shape of the power excess.
    * Using the masked power spectrum in the region centered around numax, an autocorrelation is computed to determine
      the large frequency spacing.

.. note::

    By default, both modules will run and this is the recommended procedure if no other information 
    is provided. 

    If stellar parameters like the radius, effective temperature and/or surface gravity are provided in the **info/star_info.csv** file, ``pySYD`` 
    can estimate a value for numax using a scaling relation. Therefore the first module can be bypassed,
    and the second module will use the estimated numax as an initial starting point.

    There is also an option to directly provide numax in the **info/star_info.csv** (or via command line, 
    see :ref:`advanced usage<advanced>` for more details), which will override the value found in the first module. This option 
    is recommended if you think that the value found in the first module is inaccurate, or if you have a visual 
    estimate of numax from the power spectrum.
    

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
