********************
Installation & setup
********************

.. link-button:: install/quickstart
    :type: ref
    :text: Jump to quickstart
    :classes: btn-outline-secondary btn-block

-----

Installation
############

There are three main ways you can install the software:

#. :ref:`Install from the Python Package Index <install/pip>`
#. :ref:`Create an environment <install/conda>`
#. :ref:`Clone directly from GitHub <install/git>`

.. note::

    **The recommended way to install this package is from PyPI via** `pip`, **since**
    **it will automatically enforce the proper dependencies and versions**

-----

.. _install/pip:

Install from PyPI
*****************

The ``pySYD`` package is available on the Python Package Index (`PyPI <https://pypi.org/project/pysyd/>`_)
and therefore you can install the latest stable version directly using `pip`:

.. code-block::

    pip install pysyd

The ``pysyd`` binary should have been automatically placed in your system's path via the ``pip`` command. 
If your system can not find the ``pysyd`` executable, change into the top-level ``pysyd`` directory and try 
running the following command:

.. code-block::

    python setup.py install


-----

.. _install/conda:

Create an environment
*********************

You can also use `conda` to create an environment. For this example, I'll call it 'astero'.

.. code-block::
    
    conda create -n astero numpy scipy pandas astropy matplotlib tqdm
    
See our complete list of dependencies (including versions) :ref:`below <installation-dependencies>`. 
Then activate the environment and install ``pySYD``:

.. code-block::

    conda activate astero
    pip install git+https://github.com/ashleychontos/pySYD


-----

.. _install/git:

Clone from GitHub
*****************

If you are wanting to contribute, you can clone the latest development
version from `GitHub <https://github.com/ashleychontos/pySYD>`_ using `git`.

.. code-block::

    git clone git://github.com/ashleychontos/pySYD.git

The next step is to build and install the project:

.. code-block::

    python -m pip install .

which needs to be executed from the top-level directory inside the 
cloned ``pySYD`` repo.

-----

.. _install/dependencies:

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

-----

.. _install/test:

Testing 
#######

You can test your installation by simply using the help command in a terminal
window, which should display the following output:

::

    $ pysyd --help

    usage: pySYD [-h] [--version] {check,load,parallel,run,setup,test} ...

    pySYD: automated measurements of global asteroseismic parameters

    optional arguments:
      -h, --help            show this help message and exit
      --version             Print version number and exit.

    pySYD modes:
      {check,load,parallel,run,setup,test}
        check               Check data for a target or other relevant information
        load                Load in data for a given target
        parallel            Run pySYD in parallel
        run                 Run the main pySYD pipeline
        setup               Easy setup of relevant directories and files
        test                Test different utilities (currently under development)


-----

.. _install/setup:

Setup
#####

The software package comes with a convenient setup feature, which is called through 
:mod:`pysyd.pipeline.setup`. We **strongly encourage** you to run this step 
regardless of how you choose to run ``pySYD`` because it:

- downloads example data for three stars
- provides the properly-formatted [optional] input files *and* 
- sets up the relative local directory structure

We'd like to emphasize this last bullet because it establishes a local, relative directory 
structure that is both straightforward for the pipeline and intuitive to the user.


Make a local directory
**********************

Before you do that though, we recommend that you create a new, local directory to keep all 
your pysyd-related data, information and results in a single, easy-to-find location. This is 
actually the only reason we didn't include our examples as package data, as it would've put 
them in your root directory and we realize this can be difficult to locate.

The folder or directory can be whatever is most convenient for you:

.. code-block::
    
    mkdir ~/path/to/local/pysyd/directory
    

Initialize setup
****************

Now all you need to do is change into that directory, run the following command and let
``pySYD`` do the rest of the work for you!

.. code-block::

    pysyd setup

In setup 'mode', the :term:`verbose<-v, --verbose>` output is `True` by default -- this way you can see what is
being downloaded and where it can be found:

.. code-block::
    
    Downloading relevant data from source directory:
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
    
As shown above, example data and other relevant files were downloaded from the 
`public GitHub repo <https://github.com/ashleychontos/pySYD>`_. 

.. TODO:: add an option to download example data/files as a package in the root directory.

-----

.. _install/quickstart:

Quickstart
##########

Use the following to get up and running right away: 

.. code-block::

    python -m pip install pysyd
    mkdir ~/path/to/local/pysyd/directory
    cd ~/path/to/local/pysyd/directory
    pysyd setup 

The final command which will equip you with example data and files to immediately get 
started using the software. This is essentially the same as all the steps discussed above 
but in a more condensed version.

*You are now ready to become an asteroseismologist!*

-----
