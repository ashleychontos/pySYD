.. link-button:: installation/quickstart
    :type: ref
    :text: Jump down to quickstart to get started right away!
    :classes: btn-outline-primary btn-block

.. _installation/index:

************
Installation
************


With ``pip``
++++++++++++

Install the latest stable version using pip:

.. code-block::

    pip install pysyd

This is the recommended way to install the package. The ``pysyd`` binary should have been automatically 
placed in your system's path via the ``pip`` command. If your system can not find the ``pysyd`` executable, 
change into the top-level ``pysyd`` directory and try running the following command:

.. code-block::

    python setup.py install
    
With ``conda``
++++++++++++++

Use conda to create an environment. For this example, I'll call it 'astero'.

.. code-block::
    
    conda create -n astero numpy scipy pandas astropy matplotlib tqdm
    
See our complete list of dependencies (including versions) :ref:`below <installation-dependencies>`. 
Then activate the environment and install ``pySYD``:

.. code-block::

    conda activate astero
    pip install git+https://github.com/ashleychontos/pySYD


With ``git``
++++++++++++

The latest development version can be cloned from GitHub using git:

.. code-block::

    git clone git://github.com/ashleychontos/pySYD.git

Then to build and install the project use:

.. code-block::

    python -m pip install .

from inside the cloned ``pySYD`` directory.

-----

.. _installation/dependencies:

############
Dependencies
############

This package has the following dependencies:

* `Python <https://www.python.org>`_ (>=3)
* `Numpy <https://numpy.org>`_
* `pandas <https://pandas.pydata.org>`_ 
* `Matplotlib <https://matplotlib.org/index.html#module-matplotlib>`_
* `Astropy <https://www.astropy.org>`_
* `scipy <https://docs.scipy.org/doc/>`_

Explicit version requirements are specified in the project `requirements.txt <https://github.com/ashleychontos/pySYD/requirements.txt>`_ 
and `setup.cfg <https://github.com/ashleychontos/pySYD/setup.cfg>`_. However, using ``pip`` or 
``conda`` should install and enforce these versions automatically. 

Optional
++++++++

If using the sampling feature and you want a progress bar, you'll need to install:

* `tqdm <https://tqdm.github.io>`_

-----

.. _installation/testing:

#######
Testing
#######

You can test your installation by using the help command: 
    
.. dropdown:: pysyd --help
    
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

-----

.. _installation/setup:

#####
Setup
#####

The easiest way to start using the ``pySYD`` package is by running our ``setup`` feature
from a convenient directory:

.. code-block::

    pysyd setup

This command will create `data`, `info`, and `results` directories in the current working 
directory, if they don't already exist. Setup will also download two information files: 
**info/todo.txt** and **info/star_info.csv**. See :ref:`overview` for more information on 
what purposes these files serve. Additionally, three example stars 
from the `source code <https://github.com/ashleychontos/pySYD>`_ are included (see :ref:`examples`).

The optional verbose command can be called with the setup feature:

.. dropdown:: pysyd setup --verbose
    
    | Downloading relevant data from source directory:
    | 
    | /Users/ashleychontos/Desktop/info
    |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
    |                                  Dload  Upload   Total   Spent    Left  Speed
    | 100    25  100    25    0     0     49      0 --:--:-- --:--:-- --:--:--    49
    |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
    |                                  Dload  Upload   Total   Spent    Left  Speed
    | 100   239  100   239    0     0    508      0 --:--:-- --:--:-- --:--:--   508
    |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
    |                                  Dload  Upload   Total   Spent    Left  Speed
    | 100 1518k  100 1518k    0     0  1601k      0 --:--:-- --:--:-- --:--:-- 1601k
    |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
    |                                  Dload  Upload   Total   Spent    Left  Speed
    | 100 3304k  100 3304k    0     0  2958k      0  0:00:01  0:00:01 --:--:-- 2958k
    |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
    |                                  Dload  Upload   Total   Spent    Left  Speed
    | 100 1679k  100 1679k    0     0  1630k      0  0:00:01  0:00:01 --:--:-- 1630k
    |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
    |                                  Dload  Upload   Total   Spent    Left  Speed
    | 100 3523k  100 3523k    0     0  3101k      0  0:00:01  0:00:01 --:--:-- 3099k
    |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
    |                                  Dload  Upload   Total   Spent    Left  Speed
    | 100 1086k  100 1086k    0     0   943k      0  0:00:01  0:00:01 --:--:--  943k
    |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
    |                                  Dload  Upload   Total   Spent    Left  Speed
    | 100 2578k  100 2578k    0     0  2391k      0  0:00:01  0:00:01 --:--:-- 2391k
    | 
    | 
    |  - created input file directory: /Users/ashleychontos/Desktop/pysyd/info 
    |  - created data directory at /Users/ashleychontos/Desktop/pysyd/data 
    |  - example data saved
    |  - results will be saved to /Users/ashleychontos/Desktop/pysyd/results 

which will print the absolute paths of all directories that are created during setup.

-----

.. _installation/quickstart:

##########
Quickstart
##########

To get started right away, use the following commands:

.. code-block::

    mkdir ~/path_to_put_pysyd_stuff
    cd ~/path_to_put_pysyd_stuff
    pip install pysyd
    pysyd setup
    pysyd run --star 1435467 -dv

-----
