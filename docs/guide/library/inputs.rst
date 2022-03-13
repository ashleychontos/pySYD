.. _overview/index:

******
Inputs
******

.. note::

    If you used the ``pysyd.pipeline.setup`` aka our setup feature, you will already have
    not only the appropriate files but your directories will all be setup for you -- so you
    can skip over this!

Required
########

Obviously in order to process a star, we need some data!

At the moment, we are implementing time-domain utilities (in that
we would love to calculate the power spectra for you) but it is not
ready yet!

We only realized later that some of these frequency analysis (and even
time-domain analysis) tools are not immediately obvious, especially not
for non-expert users -- which is who we want to use it! **so stay tuned!**

Data [`'./data/'`]
******************

File format for a given star, ID, must be: 

*  `'./data/ID_LC.txt'` : time series data in units of days
*  `'./data/ID_PS.txt'` : power spectrum in units of muHz versus power or power density

.. warning::

    Time and frequency in the time series and power spectrum file **must** be in the specified units (days and muHz) in order for the pipeline 
    to properly process the data and provide reliable results. 

Optional
########

Info [`'./info/'`]
******************

There are two main files provided:

* `'./info/todo.txt'` : text file with one star ID per line, which must match the data ID. If no stars are specified via command line, the star(s) listed in the text file will be processed by default. This is recommended when running a large number of stars.

#. `'./info/star_info.csv'` : contains individual star information. Star IDs are crossmatched with this list and therefore, do not need to be in any particular order. In order to read information in properly, it **must** contain the following columns:

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


