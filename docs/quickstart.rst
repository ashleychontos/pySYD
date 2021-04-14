.. _quickstart:

Getting Started
===============

.. _installation:



Installation
++++++++++++

Install ``pysyd`` using pip:

.. code-block:: bash

    $ pip install pysyd

The ``pysyd`` binary should have been automatically placed in your system's path by the
``pip`` command (see :ref:`installation`). If your system can not find
the ``pysyd`` executable then try running

.. code-block:: bash

    $ python setup.py install

from within the top-level ``pysyd`` directory.

If you are running OSX, and want to run an ensemble of stars in parallel, you 
may need to perform some additional installation steps. See :ref:`OSX-multiprocessing`.



Setting Up
++++++++++

The easiest way to start using the ``pySYD`` software is by running our setup feature:

.. code-block:: bash

    $ pysyd setup

This command will create `data`, `info`, and `results` directories in the current 
working directory, unless otherwise specified (see :ref: `CLI` for more details). 
Setup will also download two information files (`info/todo.txt` and `info/star_info.csv`) 
as well as three example stars from the `source code <https://github.com/ashleychontos/pySYD>`_.

The verbose option can also be used with the setup feature 

.. code-block:: bash

    $ pysyd setup -verbose

and will provide the absolute paths of the directories that are created.



Example Stars
+++++++++++++

Test your installation by running through one (or all) of the included
examples. We will use the ``pysyd`` command line interface to execute
a multi-planet, multi-instrument fit.



First lets look at ``pysyd --help`` for the available options:

.. code-block:: bash
		
    $ radvel --help
    usage: RadVel [-h] [--version] {fit,plot,mcmc,derive,bic,table,report} ...

    RadVel: The Radial Velocity Toolkit

    optional arguments:
      -h, --help            show this help message and exit
      --version             Print version number and exit.

    subcommands:
      {fit,plot,mcmc,derive,bic,table,report}


Here is an example workflow to
run a simple fit using the included `HD164922.py` example
configuration file. This example configuration file can be found in the ``example_planets``
subdirectory on the `GitHub repository page
<https://github.com/California-Planet-Search/radvel/tree/master/example_planets>`_.

Perform a maximum-likelihood fit. You almost always will need to do this first:

.. code-block:: bash

    $ radvel fit -s /path/to/radvel/example_planets/HD164922.py

   
By default the results will be placed in a directory with the same name as
your planet configuration file (without `.py`, e.g. `HD164922`). You
may also specify an output directory using the ``-o`` flag.

After the maximum-likelihood fit is complete the directory should have been created
and should contain one new file:
`HD164922/HD164922_post_obj.pkl`. This is a ``pickle`` binary file
that is not meant to be human-readable but lets make a plot of the
best-fit solution contained in that file:

.. code-block:: bash

    $ radvel plot -t rv -s /path/to/radvel/example_planets/HD164922.py

This should produce a plot named
`HD164922_rv_multipanel.pdf` that looks something like this.

.. image:: plots/HD164922_rv_multipanel.png

Next lets perform the Markov-Chain Monte Carlo (MCMC) exploration to
assess parameter uncertainties.

.. code-block:: bash

    $ radvel mcmc -s /path/to/radvel/example_planets/HD164922.py

Once the MCMC chains finish running there will be another new file
called `HD164922_mcmc_chains.csv.tar.bz2`. This is a compressed csv
file containing the parameter values and likelihood at each step in
the MCMC chains.


Optional Features
+++++++++++++++++

Combine the measured properties of the RV time-series with
the properties of the host star defined in the setup file to
derive physical parameters for the planetary system. Have a look at the
`epic203771098.py` example setup file to see how to include stellar parameters.

.. code-block:: bash

    $ radvel derive -s /path/to/radvel/example_planets/HD164922.py

Generate a corner plot for the derived parameters. This plot will also be
included in the summary report if available.

.. code-block:: bash

    $ radvel plot -t derived -s /path/to/radvel/example_planets/HD164922.py

Perform a model comparison testing models eliminating different sets of
planets, their eccentricities, and RV trends. If this is run a new table 
will be included in the summary report.

.. code-block:: bash

    $ radvel ic -t nplanets e trend -s /path/to/radvel/example_planets/HD164922.py

Generate and save only the TeX code for any/all of the tables.

.. code-block:: bash

    $ radvel table -t params priors ic_compare derived -s /path/to/radvel/example_planets/HD164922.py

