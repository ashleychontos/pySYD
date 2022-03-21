************
Installation
************

There are three main ways you can install the software:

#. :ref:`Using pip install <installation/pip>`
#. :ref:`Creating a conda environment <installation/conda>`
#. :ref:`Clone directly from GitHub <installation/git>`


.. _installation/pip:

With `pip`
##########

The ``pySYD`` package is available on the Python Package Index (`PyPI <https://pypi.org/project/pysyd/>`_).
Therefore you can install the latest stable version using `pip`:

.. code-block::

    pip install pysyd

The ``pysyd`` binary should have been automatically placed in your system's path via the ``pip`` command. 
If your system can not find the ``pysyd`` executable, change into the top-level ``pysyd`` directory and try 
running the following command:

.. code-block::

    python setup.py install

.. note::

    **The recommended way to install this package is from PyPI via** `pip`, **since**
    **it will automatically enforce the proper dependencies and versions**

.. _installation/conda:

With `conda`
############

You can also use conda to create an environment. For this example, I'll call it 'astero'.

.. code-block::
    
    conda create -n astero numpy scipy pandas astropy matplotlib tqdm
    
See our complete list of dependencies (including versions) :ref:`below <installation-dependencies>`. 
Then activate the environment and install ``pySYD``:

.. code-block::

    conda activate astero
    pip install git+https://github.com/ashleychontos/pySYD


.. _installation/git:

With `git`
##########

If you are wanting to contribute, you can clone the latest development
version from `GitHub <https://github.com/ashleychontos/pySYD>`_ using `git`.

.. code-block::

    git clone git://github.com/ashleychontos/pySYD.git

The next step is to build and install the project:

.. code-block::

    python -m pip install .

which needs to be executed from the top-level directory inside the 
cloned ``pySYD`` repo.


.. _installation/dependencies:

Dependencies
############

This package has the following dependencies:

* `Python <https://www.python.org>`_ (>=3)
* `Numpy <https://numpy.org>`_
* `pandas <https://pandas.pydata.org>`_ 
* `Matplotlib <https://matplotlib.org/index.html#module-matplotlib>`_
* `Astropy <https://www.astropy.org>`_
* `scipy <https://docs.scipy.org/doc/>`_
* `tqdm <https://tqdm.github.io>`_

Explicit version requirements are specified in the project `requirements.txt <https://github.com/ashleychontos/pySYD/requirements.txt>`_ 
and `setup.cfg <https://github.com/ashleychontos/pySYD/setup.cfg>`_. However, using `pip` or 
`conda` should install and enforce these versions automatically. 


.. _installation/test:

Testing your installation 
#########################

You can simply test your installation by using the help command in a terminal
window, which should display the following output:

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



.. _installation/setup:

Setting up
##########

We ***strongly encourage*** you to run this step regardless of how you intend to 
use the software because it:

- downloads data for three example stars
- provides the example [optional] input files to use with the software *and* 
- sets up the recommended local directory structure

*We emphasize the importance of the last bullet because the relative structure
is both straightforward for the user but is also what works best for running the 
software.*

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
