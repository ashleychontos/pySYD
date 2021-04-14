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

There are two main subcommands ``setup`` and ``run``, the former of which should only be
executed once after installation. The rest of the time and functionality will require ``pysyd run``.



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



