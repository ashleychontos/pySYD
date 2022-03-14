.. _quickstart/index:

**********
Quickstart
**********

The quick example demonstrated on this page is assuming the user has some
basic `Python` experience.

.. code-block::

    $ python -m pip install pysyd
    
We recommend creating a local directory to keep all your pysyd-related data, 
results and information in a single, easy-to-find location. The only thing
we are missing now is some data to do science!

Fortunately ``pySYD`` comes with a convenient setup feature (accessed via 
``pysyd.pipeline.setup``) that will download data for three example stars 
and set up the local directory structure that best works with the package.
For more information about the ``pySYD`` setup feature, see this (input page) 
or this (actual pysyd.pipeline page).

.. code-block::

    $ mkdir ~/path/to/local/pysyd/directory
    $ cd ~/path/to/local/pysyd/directory
    $ pysyd setup
    
After using these commands, you are now ready to become an asteroseismologist
(that is, if you weren't one already)!

Running your first asteroseismic analysis
#########################################



.. _installation/setup:

Setting up
###########

Ok now that the software has been successfully installed and tested, there's just 
one thing missing before we can do the science...

We need some data to do the science with!

Make a local directory
**********************

While `pip` installed ``pySYD`` to your ``PYTHONPATH``, we recommend that you first 
create a local pysyd directory before running setup. This way you can keep all your 
pysyd-related data, results and information in a single, easy-to-find location. *Note:* 
This is the only reason we didn't include our examples as package data, as it would've put 
them in your root directory and we realize this can be difficult to locate.

The folder or directory can be whatever is most convenient for you, but for demonstration
purposes we'll use:

.. code-block::
    
    mkdir ~/path/to/local/pysyd/directory
    
This way you also don't have to worry about file permissions, restricted access, and
all that other jazz. 

``pySYD`` setup
***************

The ``pySYD`` package comes with a convenient setup feature (accessed via
:ref:`pysyd.pipeline.setup<library/pipeline>`) which can be ran from the command 
line in a single step. 

We ***strongly encourage*** you to run this step regardless of how you intend to 
use the software because it:

- downloads data for three example stars
- provides the example [optional] input files to use with the software *and* 
- sets up the recommended local directory structure

The only thing you need to do from your end is initiate the command -- which now 
that you've created a local pysyd directory -- all you have to do now is jump into 
that directory and run the following command:

.. code-block::

    pysyd setup

and let ``pySYD`` do the rest of the work for you. 

Actually since this step will create a relative directory structure that might be 
useful to know, let's run the above command again but this time with the :term:`verbose output<-v, --verbose>`
so you can see what's being downloaded.

::

    $ pysyd setup --verbose
    
    Downloading relevant data from source directory:
     
     /Users/ashleychontos/Desktop/info
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100    25  100    25    0     0     49      0 --:--:-- --:--:-- --:--:--    49
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100   239  100   239    0     0    508      0 --:--:-- --:--:-- --:--:--   508
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 1518k  100 1518k    0     0  1601k      0 --:--:-- --:--:-- --:--:-- 1601k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 3304k  100 3304k    0     0  2958k      0  0:00:01  0:00:01 --:--:-- 2958k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 1679k  100 1679k    0     0  1630k      0  0:00:01  0:00:01 --:--:-- 1630k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 3523k  100 3523k    0     0  3101k      0  0:00:01  0:00:01 --:--:-- 3099k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 1086k  100 1086k    0     0   943k      0  0:00:01  0:00:01 --:--:--  943k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 2578k  100 2578k    0     0  2391k      0  0:00:01  0:00:01 --:--:-- 2391k
    
    
      - created input file directory: /Users/ashleychontos/Desktop/pysyd/info
      - created data directory at /Users/ashleychontos/Desktop/pysyd/data
      - example data saved
      - results will be saved to /Users/ashleychontos/Desktop/pysyd/results


**Note:** this is another good sanity check to make sure everything is working as intended.





-----

.. _getting_started/overview:

Overview
########

``pySYD`` is a python-based implementation of the IDL-based ``SYD`` pipeline 
`(Huber et al. 2009) <https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract>`_, 
which was extensively used to measure asteroseismic parameters for Kepler stars. 
Papers based on asteroseismic parameters measured using the ``SYD`` pipeline include 
`Huber et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract>`_, 
`Chaplin et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014ApJS..210....1C/abstract>`_, 
`Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_ 
and `Yu et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJS..236...42Y/abstract>`_.

``pySYD`` adapts the well-tested methodology from ``SYD`` while also improving these 
existing analyses and expanding upon numerous new features. Improvements include:

- Automated best-fit background model selection
- Parallel processing
- Easily accessible + command-line friendly interface
- Ability to save samples for further analyses

-----

.. _getting_started/structure:

Structure
*********

We recommend using the following structure under three main directories, which is discussed 
in more detail below:

#. **info/** : [optional input] directory to provide prior information on the processed stars
#. **data/** : [required input] directory containing the light curves and power spectra
#. **results/** : [optional output] directory for result figures and files

Input
=====

Info/
+++++

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
+++++

File format for a given star ID: 

*  **ID_LC.txt** : time series data in units of days
*  **ID_PS.txt** : power spectrum in units of muHz versus power or power density

.. warning::

    Time and frequency in the time series and power spectrum file **must** be in the specified units (days and muHz) in order for the pipeline 
    to properly process the data and provide reliable results. 

Output
======

Results/
++++++++

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

.. _performance/comparison:


``pySYD`` vs ``SYD``
***************************

We ran pySYD on ~100 Kepler legacy stars observed in short-cadence and compared the output to IDL SYD results from `Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_. The same time series and power spectra were used for both analyses.
The resulting values are compared for the two methods below for the frequency of maximum power 
(left) and the large frequency separation (Dnu) on the right. For reference,
``SYD == pySYD`` and ``IDL == SYD``.

.. image:: figures/performance/comparison.png
  :width: 680
  :alt: Comparison of ``pySYD`` and ``SYD``

There residuals show no strong systematics to within <0.5% in Dnu and <~1% in numax, which is smaller than the typical 
random uncertainties. This confirms that the open-source python package pySYD provides consistent results with the legacy 
IDL version that has been used extensively in the literature.

*** NOTE **** Add tutorial or jupyter notebook to reproduce this figure.


.. _performance/speed:

Speed
*******
