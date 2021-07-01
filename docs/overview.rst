.. _overview:


Modes
******

There are currently four main ``pySYD`` modes: 

#. ``setup`` : Initializes ``pysyd.pipeline.setup`` for quick and easy setup of directories, files and examples. This mode only
   inherits higher level functionality and has limited CLI (see :ref:`parent parser<parentparse>` below). Using this feature will
   set up the paths and files consistent with what is recommended and discussed in more detail below.

#. ``load`` : Initializes ``pysyd.pipeline.load`` to load in data for a single target. Because this does handle data, this has 
   full access to both the :ref:`parent<parentparse>` and :ref:`main parser<mainparse>`.

#. ``run`` : The main pySYD pipeline function is initialized through ``pysyd.pipeline.run`` and runs the two core modules 
   (i.e. ``find_excess`` and ``fit_background``) for each star consecutively. This mode operates using most CLI options, inheriting
   both the :ref:`parent<parentparse>` and :ref:`main parser<mainparse>` options.

#. ``parallel`` : Operates the same way as the previous mode, but processes stars simultaneously in parallel. Based on the number of threads
   available, stars are separated into groups (where the number of groups is exactly equal to the number of threads). This mode uses all CLI
   options, including the number of threads to use for parallelization (:ref:`see here<parallel>`).

=========================

Recommended Structure
***********************

We recommend using the following structure under three main directories, which is discussed 
in more detail below:

#. **info/** : [optional input] directory to provide prior information on the processed stars
#. **data/** : [required input] directory containing the light curves and power spectra
#. **results/** : [optional output] directory for result figures and files

Input
********

Info/
++++++++

There are two main files provided:

#. **info/todo.txt** : text file with one star ID per line, which must match the data ID. If no stars are specified via command line, the star(s) listed in the text file will be processed by default. This is recommended when running a large number of stars.

#. **info/star_info.csv** : contains individual star information. Star IDs are crossmatched with this list and therefore, do not need to be in any particular order. In order to read information in properly, it **must** contain the following columns:

   * "stars" [int] : star IDs that should exactly match the star provided via command line or in todo.txt
   * "radius" [float] : stellar radius (units: solar radii)
   * "teff" [float] : effective temperature (units: K)
   * "logg" [float] : surface gravity (units: dex)
   * "numax" [float] : the frequency corresponding to maximum power (units: muHz)
   * "lower_ex" [float] : lower frequency limit to use in the find_excess module (units: muHz)
   * "upper_ex" [float] : upper frequency limit to use in the find_excess module (units: muHz)
   * "lower_bg" [float] : lower frequency limit to use in the fit_background module (units: muHz)
   * "upper_bg" [float] : upper frequency limit to use in the fit_background module (units: muHz)
   * "seed" [int] : random seed generated when using the Kepler correction option, which is saved for future reproducibility purposes

Data/
++++++++

File format for a given star ID: 

*  **ID_LC.txt** : time series data in units of days
*  **ID_PS.txt** : power spectrum in units of muHz versus power or power density

.. warning::

    Time and frequency in the time series and power spectrum file **must** be in the specified units (days and muHz) in order for the pipeline 
    to properly process the data and provide reliable results. 

Output
********

Results/
++++++++++

Subdirectories are automatically created for each individually processed star.
Results for each of the two main ``pySYD`` modules (``find_excess`` and ``fit_background``) 
will be concatenated into a single csv in the upper-level results directory, which is
helpful when running many stars.

A single star will yield one summary figure (png) and one data product (csv) for each of the two
main modules, for a total of 4 output files. If the monte-carlo sampling is used to calculate uncertainties, an additional
figure will show the posterior distributions for the estimated parameters. See :ref:`examples` 
for a guide on what the output plots are showing.

==========================


Pipeline Overview
++++++++++++++++++++

Initialization of ``pySYD`` via command line will look in the following paths:

- ``TODODIR`` : '~/path_to_put_pysyd_stuff/info/todo.txt'
- ``INFODIR`` : '~/path_to_put_pysyd_stuff/info/star_info.csv'
- ``INPDIR`` : '~/path_to_put_pysyd_stuff/data'
- ``OUTDIR`` : '~/path_to_put_pysyd_stuff/results'

which by default, is the absolute path of the current working directory (or however you choose to set it up). All of these paths should be ready to go
if you followed the suggestions in :ref:`structure` or used our ``setup`` feature.

A ``pySYD`` pipeline ``Target`` class object has two main function calls:

#. The find routine :
    * **Summary:** searches for power excess due to solar-like oscillations by implementing a frequency-resolved, collapsed 
      autocorrelation function (ACF) using 3 different ``box`` sizes
    * The main purpose for this first module is to provide a good starting point for the
      second module. The output from this routine provides a rough estimate for numax, which is translated 
      into a frequency range in the power spectrum that is believed to exhibit characteristics of p-mode
      oscillations
#. The second routine : 
    * **Summary:** performs a fit to the granulation background, corrects 
      the power spectrum using this model, and then fits for the global parameters numax and dnu.
    * Given the frequency range determined by the first module, this region is masked out to model 
      the white and red noise contributions present in the power spectrum. The fitting procedure will
      test a series of models and select the best-fit stellar background model and correct the power spectrum
    * The module will then estimate numax using two different methods: 1) Fitting a Gaussian to the smoothed, background corrected power spectrum and 2) Finding the frequency corresponding to the maximum power of the smoothed, background corrected power spectrum. The second method has typically been adopted as the default numax value reported in the literature since it makes no assumptions about the shape of the power excess.
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
