.. _overview:

Overview
########

A ``pySYD`` pipeline ``Target`` class object has two main methods:

#. ``find_excess`` : searched for power excess due to solar-like oscillations using a collapsed 
   ACF approach with 3 different ``box`` sizes
#. ``fit_background`` : optimizes the global fit by selecting the best-fit stellar background
   model, correcting the power spectrum using this model, and then fitting for the global parameters
   numax and dnu

.. note::

    By default, both of these modules will run and is the recommended procedure if no prior stellar 
    information is provided (i.e in star_info.csv, see below for more details). 

    If stellar parameters (i.e. radius, effective temperature, surface gravity) are provided, ``pySYD`` 
    will estimate a value for numax using a scaling relation. The first module can therefore be bypassed,
    which will use the estimated numax as an initial starting point for the second module.

    There is also an option to directly provide numax in the csv, which will override the value found in
    the first module. Therefore, we suggest using this if you think the value found in the first module
    is not the correct value. This will ensure that the second module will instead center around the 
    desired numax.


=========================

Structure
*********

We recommend using the following structure under three main directories:

#. **info/** : [optional input] directory to provide prior information on the processed stars
#. **data/** : [required input] directory containing the light curves and power spectra
#. **results/** : [optional output] directory for result figures and files


Input
*****

Info/
+++++

There are two main files provided:

#. **info/todo.txt** : this is a basic text file with one star ID per line, which must match the data ID to be loaded in properly. If no stars are specified via command line, the star(s) listed in the text file will be processed by default. This is recommended when running a large number of stars.

#. **info/star_info.csv** : a means for providing individual star information. Star IDs are crossmatched with this list and therefore, do not need to be in any particular order. In order to read information in properly, it **must** contain the following column heads:

   * "stars" [int] : star IDs that should exactly match the star provided via command line or in todo.txt
   * "rad" [float] : stellar radius (units: solar radii)
   * "teff" [float] : effective temperature (units: K)
   * "logg" [float] : surface gravity (units: dex)
   * "numax" [float] : the frequency corresponding to maximum power (units: muHz)
   * "lower_x" [float] : lower frequency limit to use in the findex module (units: muHz)
   * "upper_x" [float] : upper frequency limit to use in the findex module (units: muHz)
   * "lower_b" [float] : lower frequency limit to use in the background-fitting module (units: muHz)
   * "upper_b" [float] : upper frequency limit to use in the background-fitting module (units: muHz)
   * "seed" [int] : random seed generated when using the Kepler correction option, which is saved for future reproducibility purposes


Data/
+++++

File format for a given star ID: 

*  **ID_LC.txt** : lightcurve in units of days versus fractional flux
*  **ID_PS.txt** : power spectrum in units of muHz versus ppm^2 muHz^-1 (normalized power density)


.. warning::

    The power spectrum **must** be in the specified units in order for the pipeline 
    to properly process the data and provide reliable results. 


Output
******

Results/
++++++++

Subdirectories are automatically created for each individually processed star (by their ID).
Results for each of the two main ``pySYD`` modules (i.e. ``find_excess`` and ``fit_background``) 
will be concatenated into a single csv in the upper-level results directory, which is
helpful when running many stars.

A single star will yield one summary figure (png) and one data product (csv) for each of the two
main modules, for a total of 4 output files. If the monte-carlo sampling is used, an additional
figure will show the posterior distributions for the estimated parameters. While not creating
another output, the errors will be reflected in the `background.csv` file. There is also an 
option to save the samples if desired for later use (by adding ``-samples`` to the command line). 
See :ref:`examples` for examples on output plots.
