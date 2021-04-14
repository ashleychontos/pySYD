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

You may test your installation by using ``pysyd --help`` to see available options:

.. code-block:: bash
		
    $ pysyd --help
    usage: pySYD [-h] [-version] {setup,run} ...

    pySYD: Automated Extraction of Global Asteroseismic Parameters

    optional arguments:
      -h, --help           show this help message and exit
      -version, --version  Print version number and exit.

    subcommands:
      {setup,run}

The two main subcommands ``setup`` and ``run``, the former of which should only be
used once after installation. The rest of the time, you will use ``pysyd run``.



Setting Up
++++++++++

The easiest way to start using the ``pySYD`` software is by running our setup feature
from a convenient directory

.. code-block:: bash

    $ pysyd setup

Please note that this command will create `data`, `info`, and `results` directories in 
the current working directory, unless otherwise specified (see :ref: `CLI` for more details). 
Setup will also download two information files: `info/todo.txt` and `info/star_info.csv`. See 
:ref:`overview` for more information on what purposes these files serve. Additionally, three
example stars from the `source code <https://github.com/ashleychontos/pySYD>`_ are included 
(see :ref:`examples`).

The optional verbose command can also be called with the setup feature 

.. code-block:: bash

    $ pysyd setup -verbose

which will provide the absolute paths of all directories that are created during this step.



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

