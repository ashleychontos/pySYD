***************
Getting started
***************

In case you somehow missed it -- there is a convenient way for you to get started right
away -- which is by running the ``pySYD`` setup feature. Running the command will provide 
*all* of the relevant files that are discussed in detail on this page. 

To read more details about the setup feature, please visit our :ref:`Installation <..installation>` 
or :ref:`Quickstart<..quickstart>` pages. You can also see how it works directly by visiting
:ref:`the API <pipeline>`. 

Inputs
######

For the initial ``pySYD`` release, we required both the time series data and power 
spectrum since it was initially a translation of ``SYD`` and the pipeline also required 
those two inputs. **FWIW:** the time series data was only used for two purposes: 1) to
determine if the power spectrum was :term:`critically sampled` or :term:`oversampled` 
and 2) to plot as a sanity check. *No data manipulation is currently done to the light curve.*

Since then, we have had requests to drop the requirement for the time series data 
so it is also now possible to now run ``pySYD`` using only a power spectrum. However, this 
still requires the user to compute their own power spectrum.

It was not until recently that we realized the frequency analysis tools and even some of 
the time-domain tools (relevant to asteroseismology) are not immediately obvious, especially 
for the non-expert users.

**Since this is ultimately our target audience, we are currently working on developing and**
**implementing time-domain utilities so stay tuned!**

Required Input
##############

The only thing *required* to successfully run the software and get results is the data 
in which ``pySYD`` will apply the asteroseismic analyses on! 

Data 
****

For a given star, ID, the data : 

*  `'./data/ID_LC.txt'` : time series data in units of days
*  `'./data/ID_PS.txt'` : power spectrum in units of muHz versus power or power density

.. warning::

    It is **critical** that these files are in the proper units in order for ``pySYD`` 
    to work properly. If you are unsure about any of these units, your best bet is to
    provide a light curve (in days) and let us calculate the power spectrum for you! 

Optional Input
##############

There are two main information files which can be provided but both are optional -- whether
or not you choose to use them ultimately depends on how you will run the software. 

Target list
***********

For example

* `'./info/todo.txt'` : text file with one star ID per line, which must match the data ID. If no stars are specified via command line, the star(s) listed in the text file will be processed by default. This is recommended when running a large number of stars.

Star info
*********

As suggested by the name of the file, this contains star information on an individual basis. Similar to
the data, target IDs must *exactly* match the given name in order to be successfully crossmatched -- but
this also means that the information in this file need not be in any particular order. 

.. csv-table:: `'info/star_info.csv'`
   :header: "stars", "rs", "logg", "teff", "numax", "lower_ex", "upper_ex", "lower_bg"
   :widths: 20, 10, 10, 20, 20, 20, 20, 20

   1435467, 1.0, 4.4, 5777.0, 1400.0, 100.0, 5000.0, 100.0
   2309595, 1.0, 4.4, 5777.0, 1400.0, 100.0, 5000.0, 100.0

However in order to ingest the information properly, the columns must match the following and/or 
relevant commands:

   * "stars" [int,str] : star IDs that should exactly match the star provided via command line or in todo.txt
   * "radius" [float] : stellar radius (units: solar radii)
   * "teff" [float] : effective temperature (units: K)
   * "logg" [float] : surface gravity (units: dex)
   * "numax" [float] : the frequency corresponding to maximum power (units: :math:`\rm \mu Hz`)
   * "lower_ex" [float] : lower frequency limit to use in the find_excess module (units: :math:`\rm \mu Hz`)
   * "upper_ex" [float] : upper frequency limit to use in the find_excess module (units: :math:`\rm \mu Hz`)
   * "lower_bg" [float] : lower frequency limit to use in the fit_background module (units: :math:`\rm \mu Hz`)
   * "upper_bg" [float] : upper frequency limit to use in the fit_background module (units: :math:`\rm \mu Hz`)
   * "seed" [int] : random seed generated when using the Kepler correction option, which is saved for future reproducibility purposes

This file is especially helpful for running many stars with different options!

.. TODO::

    Add all the available options (columns) to the csv 
    

