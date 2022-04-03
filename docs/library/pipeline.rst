**************
Main execution
**************

When running the software,  `pySYD` will look in the following paths:

- ``INFDIR`` : '~/path/to/local/pysyd/directory/info'
- ``INPDIR`` : '~/path/to/local/pysyd/directory/data'
- ``OUTDIR`` : '~/path/to/local/pysyd/directory/results'

which by default, is the absolute path of the current working directory (or wherever you
ran setup from).

-----

The software generally operates in four main steps:
 #. :ref:`Loads in parameters and data <stepone>`
 #. :ref:`Estimates starting points <steptwo>`
 #. :ref:`Fits global parameters <stepthree>`
 #. :ref:`Estimates uncertainties <stepfour>`

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



Introduction
############

This was initially created to be used with the command-line tool but has been expanded to include
functions compatible with `Python` notebooks as well. This is still a work in progress! 

Imports
#######

Pipeline modes
##############

There are currently four operational ``pySYD`` modes (and two under development): 

#. ``setup`` : Initializes ``pysyd.pipeline.setup`` for quick and easy setup of directories, files and examples. This mode only
   inherits higher level functionality and has limited CLI (see :ref:`parent parser<parentparse>` below). Using this feature will
   set up the paths and files consistent with what is recommended and discussed in more detail below.

#. ``load`` : Loads in data for a single target through ``pysyd.pipeline.load``. Because this does handle data, this has 
   full access to both the :ref:`parent<parentparse>` and :ref:`main parser<mainparse>`.

#. ``run`` : The main pySYD pipeline function is initialized through ``pysyd.pipeline.run`` and runs the two core modules 
   (i.e. ``find_excess`` and ``fit_background``) for each star consecutively. This mode operates using most CLI options, inheriting
   both the :ref:`parent<parentparse>` and :ref:`main parser<mainparse>` options.

#. ``parallel`` : Operates the same way as the previous mode, but processes stars simultaneously in parallel. Based on the number of threads
   available, stars are separated into groups (where the number of groups is exactly equal to the number of threads). This mode uses all CLI
   options, including the number of threads to use for parallelization (:ref:`see here<parallel>`).

#. ``display`` : will primarily be used for development and testing purposes as well, but

#. ``test`` : Currently under development but intended for developers.


Examples
########

``pysyd.pipeline`` API
######################

.. automodule:: pysyd.pipeline
   :members:
