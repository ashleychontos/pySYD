.. _quickstart:

Getting Started
###############

Jump down to :ref:`summary` to get asteroseismic parameters for a star in less than a minute!

.. _installation:

Installation
************

Install ``pysyd`` using pip:

.. code-block:: bash

    $ pip install pysyd

The ``pysyd`` binary should have been automatically placed in your system's path by the
``pip`` command. If your system can not find the ``pysyd`` executable, ``cd`` into the 
top-level ``pysyd`` directory and try running the following command:

.. code-block:: bash

    $ python setup.py install

You may test your installation by using ``pysyd --help`` to see available command-line options:

.. code-block:: bash
		
    $ pysyd --help
    usage: pySYD [-h] [-version] {setup,run} ...

    pySYD: Automated Extraction of Global Asteroseismic Parameters

    optional arguments:
      -h, --help           show this help message and exit
      -version, --version  Print version number and exit.

    subcommands:
      {setup,run}


Setting Up
**********

The easiest way to start using the ``pySYD`` software is by running our setup feature
from a convenient directory:

.. code-block:: bash

    $ pysyd setup

This command will create `data`, `info`, and `results` directories in the current working 
directory, if they don't already exist. Setup will also download two information files: 
**info/todo.txt** and **info/star_info.csv**. See :ref:`overview` for more information on 
what purposes these files serve. Additionally, three example stars 
from the `source code <https://github.com/ashleychontos/pySYD>`_ are included (see :ref:`examples`).

The optional verbose command can be called with the setup feature:

.. code-block:: bash

    $ pysyd setup -verbose

This will print the absolute paths of all directories that are created during setup.


Example Fit
***********

If you ran the setup feature, there are three example stars provided: 1435467 (the least evolved), 
2309595 (~SG), and 11618103 (RGB). To run a single star, execute the main script with the following command:

.. code-block:: bash

    $ pysyd run -star 1435467 -show -verbose

``pySYD`` is optimized for running multiple stars and therefore by default, both the ``-verbose`` and ``-show`` 
(i.e. the output plots) options are set to ``False``. We recommend using them for the example, since they are helpful to see how 
the pipeline processes targets.

To estimate uncertainties in the derived parameters, set ``-mc`` to a number sufficient for bootstrap sampling. In the previous 
example, ``-mc`` was not specified and is 1 by default (for 1 iteration). Below shows the same example with the
sampling enabled, including the verbose output you should see if your software was installed successfully.

.. code-block:: bash

    $ pysyd run -star 1435467 -show -verbose -mc 200
    
    -------------------------------------------------
    Target: 1435467
    -------------------------------------------------
    # LIGHT CURVE: 37919 lines of data read
    # POWER SPECTRUM: 99518 lines of data read
    oversampled by a factor of 5
    time series cadence: 58 seconds
    power spectrum resolution: 0.426868 muHz
    -------------------------------------------------
    Running find_excess module:
    PS binned to 338 datapoints
    power excess trial 1: numax = 1459.04 +/- 68.65
    S/N: 1.77
    power excess trial 2: numax = 1449.05 +/- 83.27
    S/N: 2.18
    power excess trial 3: numax = 1442.87 +/- 71.29
    S/N: 6.16
    selecting model 3
    -------------------------------------------------
    Running fit_background module:
    PS binned to 343 data points
    Comparing 4 different models:
    1: one harvey model w/ white noise free parameter
    2: one harvey model w/ white noise fixed
    3: two harvey model w/ white noise free parameter
    4: two harvey model w/ white noise fixed
    Based on reduced chi-squared statistic: model 4
    -------------------------------------------------
    Running sampling routine:
    100%|█████████████████████████████████████████| 200/200 [00:17<00:00, 11.75it/s]

    Output parameters:
    numax (smoothed): 1312.62 +/- 69.30 muHz
    maxamp (smoothed): 1.34 +/- 0.29 ppm^2/muHz. 
    numax (gaussian): 1366.75 +/- 48.85 muHz
    maxamp (gaussian): 1.20 +/- 0.23 ppm^2/muHz
    fwhm (gaussian): 271.63 +/- 67.40 muHz
    dnu: 71.00 +/- 0.83 muHz
    -------------------------------------------------

    Combining results into single csv file.


.. _summary:

Quickstart
**********

.. compound::

    To determine asteroseismic parameters for a single star in roughly sixty seconds, execute 
    the following commands: :: 
    
	$ mkdir ~/path_to_put_pysyd_stuff
	$ cd ~/path_to_put_pysyd_stuff
        $ pip install pysyd
	$ pysyd setup
	$ pysyd run -star 1435467 -show -verbose -mc 200
        
    ... and if you weren't one already, you are now an asteroseismologist!
    
