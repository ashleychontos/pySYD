.. _quickstart:

Getting Started
###############

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

    $ pysyd -star 1435467 -show -verbose

``pySYD`` is optimized for running multiple stars and therefore by default, both the ``-verbose`` and ``-show`` 
(i.e. the output plots) options are set to ``False``. We recommend using them for the example, since they are helpful to see how 
the pipeline processes targets.

To estimate uncertainties in the derived parameters, set ``-mc`` to a number sufficient for bootstrap sampling.

.. code-block:: bash

    $ pysyd -star 1435467 -show -verbose -mc 200

In the previous example, ``-mc`` was not specified and is 1 by default (for 1 iteration). By changing this 
value, it will randomize the power spectrum for the specified number of steps and attempt to recover the parameters. 
The uncertainties will appear in the verbose output, output csvs, and an additional figure will show 
the posterior distributions for the derived parameters.

