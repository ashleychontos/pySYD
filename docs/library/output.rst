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
