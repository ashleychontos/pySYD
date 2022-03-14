***************
Getting started
***************

**ICYMI:** there is a very convenient way for you to immediately get started -- and that is by
running the ``pySYD`` setup command. Running the command will provide all of the relevant files 
that are discussed in detail on this page. 

Setting up
##########

We ***strongly encourage*** you to run this step regardless of how you intend to 
use the software because it:

- downloads data for three example stars
- provides the example [optional] input files to use with the software *and* 
- sets up the recommended local directory structure

We emphasize the importance of the last bullet because the relative structure
is both straightforward for the user but is also what works best for running the 
software.

Make a local directory
**********************

Before you do that though, we recommend that you create a new, local directory to keep all 
your pysyd-related data, information and results in a single, easy-to-find location. This is 
actually the only reason we didn't include our examples as package data, as it would've put 
them in your root directory and we realize this can be difficult to locate.

The folder or directory can be whatever is most convenient for you:

.. code-block::
    
    mkdir ~/path/to/local/pysyd/directory
    

Run the setup command
*********************

Now all you need to do is change into the new directory, run the command

.. code-block::

    cd ~/path/to/local/pysyd/directory
    pysyd setup

and let ``pySYD`` do the rest of the work for you. 


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

Required
********

The only thing *required* to successfully run the software and get results is the data 
in which ``pySYD`` will apply the asteroseismic analyses on! 

Data 
====

For a given star, ID, the data : 

*  `'./data/ID_LC.txt'` : time series data in units of days
*  `'./data/ID_PS.txt'` : power spectrum in units of muHz versus power or power density

.. warning::

    It is **critical** that these files are in the proper units in order for ``pySYD`` 
    to work properly. 

Optional
********

There are two main information files which can be provided but both are optional -- whether
or not you choose to use them ultimately depends on how you will run the software. 

Target list
============

For example

* `'./info/todo.txt'` : text file with one star ID per line, which must match the data ID. If no stars are specified via command line, the star(s) listed in the text file will be processed by default. This is recommended when running a large number of stars.

Star info
=========

As suggested by the name of the file, this contains star information on an individual basis. Similar to
the data, target IDs must *exactly* match the given name in order to be successfully crossmatched -- but
this also means that the information in this file need not be in any particular order. 

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
    

