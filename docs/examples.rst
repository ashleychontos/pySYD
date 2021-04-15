.. _examples:

Examples
========

If ``pysyd setup`` was successfully run, there should now be light curves and power spectra 
for three KIC stars in the `data/` directory. In addition, there should be a `todo.txt`
saved to `info/` as well as a `star_info.csv`. The `todo.txt` is a basic text file with
one star ID per line, which must match the data ID to be read in properly. If no stars are
specified via command line, the star(s) listed in the text file will be processed by
default. This is convenient when running an ensemble of stars. 

Light curves and power spectra can be added, following a similar format: ID_LC.txt 
for the light curve and ID_PS.txt for the power spectrum. Please note: the units of the 
power spectrum MUST be in ppm^2 muHz^-1 for the software to work properly.

====================

Single Star
+++++++++++

KIC 1435467

pySYD ``find_excess`` results:

.. image:: figures/ex1_x.png

pySYD ``fit_background`` results:

.. image:: figures/ex1_b.png

pySYD ``sampling`` results:

.. image:: figures/ex1_s.png

====================

KIC 2309595

pySYD ``find_excess`` results:

.. image:: figures/ex2_x.png

pySYD ``fit_background`` results:

.. image:: figures/ex2_b.png

pySYD ``sampling`` results:

.. image:: figures/ex2_s.png

====================

KIC 11618103

pySYD ``find_excess`` results:

.. image:: figures/ex3_x.png

pySYD ``fit_background`` results:

.. image:: figures/ex3_b.png

pySYD ``sampling`` results:

.. image:: figures/ex3_s.png


====================


Ensemble of Stars
+++++++++++++++++

If you are running OSX, and want to run an ensemble of stars in parallel, you 
may need to perform some additional installation steps. See ###.
