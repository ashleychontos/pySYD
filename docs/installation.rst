************
Installation
************

With ``pip``
############

Install the latest stable version using pip:

.. code-block::

    pip install pysyd

This is the recommended way to install the package. The ``pysyd`` binary should have been automatically 
placed in your system's path via the ``pip`` command. If your system can not find the ``pysyd`` executable, 
change into the top-level ``pysyd`` directory and try running the following command:

.. code-block::

    python setup.py install
    

.. link-button:: quickstart
    :type: ref
    :text: Jump to Quickstart to get started right away!
    :classes: btn-outline-secondary btn-block
    
With ``conda``
##############

Use conda to create an environment. For this example, I'll call it 'astero'.

.. code-block::
    
    conda create -n astero numpy scipy pandas astropy matplotlib tqdm
    
See our complete list of dependencies (including versions) :ref:`below <installation-dependencies>`. 
Then activate the environment and install ``pySYD``:

.. code-block::

    conda activate astero
    pip install git+https://github.com/ashleychontos/pySYD


With ``git``
############

The latest development version can be cloned from GitHub using git:

.. code-block::

    git clone git://github.com/ashleychontos/pySYD.git

Then to build and install the project use:

.. code-block::

    python -m pip install .

from inside the cloned ``pySYD`` directory.


.. _installation/dependencies:


Package Dependencies
####################

This package has the following dependencies:

* `Python <https://www.python.org>`_ (>=3)
* `Numpy <https://numpy.org>`_
* `pandas <https://pandas.pydata.org>`_ 
* `Matplotlib <https://matplotlib.org/index.html#module-matplotlib>`_
* `Astropy <https://www.astropy.org>`_
* `scipy <https://docs.scipy.org/doc/>`_
* `tqdm <https://tqdm.github.io>`_

Explicit version requirements are specified in the project `requirements.txt <https://github.com/ashleychontos/pySYD/requirements.txt>`_ 
and `setup.cfg <https://github.com/ashleychontos/pySYD/setup.cfg>`_. However, using ``pip`` or 
``conda`` should install and enforce these versions automatically. 

Testing your installation
*************************

You can test your installation by using the help command: 

::

    $ pysyd --help

    usage: pySYD [-h] [-version] {load,parallel,run,setup,test} ...
    
    pySYD: Automated Extraction of Global Asteroseismic Parameters
    
    optional arguments:
      -h, --help            show this help message and exit
      -version, --version   Print version number and exit.
     
    pySYD modes:
      {load,parallel,run,setup,test}
        load                Load in data for a given target
        parallel            Run pySYD in parallel
        run                 Run the main pySYD pipeline
        setup               Easy setup of relevant directories and files
        test                Test different utilities (currently under development)

