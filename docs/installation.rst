************
Installation
************

There are three main ways you can install the software:

#. :ref:`Using pip install <installation/pip>`
#. :ref:`Creating a conda environment <installation/conda>`
#. :ref:`Clone directly from GitHub <installation/git>`


.. _installation/pip:

With `pip`
**********

The ``pySYD`` package is available on the Python Package Index (PyPI) (`here <https://pypi.org/project/pysyd/>`_).
Therefore you can install the latest stable version using `pip`:

.. code-block::

    pip install pysyd

**This is the recommended way to install the package.** 

The ``pysyd`` binary should have been automatically placed in your system's path via the ``pip`` command. 
If your system can not find the ``pysyd`` executable, change into the top-level ``pysyd`` directory and try 
running the following command:

.. code-block::

    python setup.py install


.. _installation/conda:

With `conda`
************

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
**********

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

Testing the installation 
########################

You can test your installation by using the help command, which should
display the following output:

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
