.. role:: underlined
   :class: underlined

**********************************
:underlined:`Installation & setup`
**********************************

.. link-button:: install-quickstart
    :type: ref
    :text: Jump to quickstart
    :classes: btn-outline-secondary btn-block

-----

Installation
############

There are three main ways you can install the software:

#. :ref:`Install via PyPI <install-pip>`
#. :ref:`Create an environment <install-conda>`
#. :ref:`Clone directly from GitHub <install-git>`

.. note::

    **The recommended way to install this package is from PyPI via** `pip`, **since**
    **it will automatically enforce the proper dependencies and versions**


.. _install-pip:

Use `pip`
*********

The ``pySYD`` package is available on the Python Package Index (`PyPI <https://pypi.org/project/pysyd/>`_)
and therefore you can install the latest stable version directly using `pip`:

.. code-block::

   $ python -m pip install pysyd

The ``pysyd`` binary should have been automatically placed in your system's path via the 
``pip`` command. To check the command-line installation, you can use the help command in 
a terminal window, which should display something similar to the following output:

::

   $ pysyd --help
   
   usage: pySYD [-h] [--version] {check,fun,load,parallel,plot,run,setup,test} ...
   
   pySYD: automated measurements of global asteroseismic parameters
   
   options:
     -h, --help            show this help message and exit
     --version             Print version number and exit.
   
   pySYD modes:
     {check,fun,load,parallel,plot,run,setup,test}
       check               Check data for a target or other relevant information
       fun                 Print logo and exit
       load                Load in data for a given target
       parallel            Run pySYD in parallel
       plot                Create and show relevant figures
       run                 Run the main pySYD pipeline
       setup               Easy setup of relevant directories and files
       test                Test current installation


If your system can not find the ``pysyd`` executable, change into the top-level ``pysyd`` directory and try 
running the following command:

.. code-block::

   $ python setup.py install

.. _install-conda:

Create an environment
*********************

You can also use `conda` to create an environment. For this example, I'll call it 'astero'.


.. code-block::
    
   $ conda create -n astero numpy scipy pandas astropy matplotlib tqdm


See our complete list of dependencies (including versions) :ref:`below <install-dependencies>`. 
Then activate the environment and install ``pySYD``:


.. code-block::

   $ conda activate astero
   $ pip install git+https://github.com/ashleychontos/pySYD


.. _install-git:

Clone from GitHub
*****************

If you want to contribute, you can clone the latest development
version from `GitHub <https://github.com/ashleychontos/pySYD>`_ using `git`.

.. code-block::

   $ git clone git://github.com/ashleychontos/pySYD.git

The next step is to build and install the project:

.. code-block::

   $ python -m pip install .

which needs to be executed from the top-level directory inside the 
cloned ``pySYD`` repo.


-----

.. _install-dependencies:

Dependencies
############

This package has the following dependencies:

 * `Python <https://www.python.org>`_ (>=3)
 * `Numpy <https://numpy.org>`_
 * `pandas <https://pandas.pydata.org>`_ 
 * `Astropy <https://www.astropy.org>`_
 * `scipy <https://docs.scipy.org/doc/>`_
 * `Matplotlib <https://matplotlib.org/index.html#module-matplotlib>`_
 * `tqdm <https://tqdm.github.io>`_


Explicit version requirements are specified in the project `requirements.txt <https://github.com/ashleychontos/pySYD/requirements.txt>`_ 
and `setup.cfg <https://github.com/ashleychontos/pySYD/setup.cfg>`_. However, using `pip` or 
`conda` should install and enforce these versions automatically. 


-----

.. _install-setup:

Setup
#####

The software package comes with a convenient setup feature which we **strongly 
encourage** you to do because it:

- downloads example data for three stars
- provides the properly-formatted [optional] input files *and* 
- sets up the relative local directory structure

**Note:** this step is helpful *regardless* of how you intend to use the software package.

:underlined:`Make a local directory`
************************************

We recommend to first create a new, local directory to keep all your pysyd-related 
data, information and results in a single, easy-to-find location. The folder or 
directory can be whatever is most convenient for you:

.. code-block::
    
   mkdir pysyd
    

:underlined:`Initialize setup`
******************************

Now all you need to do is change into that directory, run the following command and let
``pySYD`` do the rest of the work for you!

.. code-block::

   pysyd setup -v

We used the :term:`verbose<-v, --verbose>` command so you can see what is being downloaded
and where it is being downloaded to.

.. code-block::
    
   Downloading relevant data from source directory:
     % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
   100    25  100    25    0     0    378      0 --:--:-- --:--:-- --:--:--   378
     % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
   100   810  100   810    0     0  11739      0 --:--:-- --:--:-- --:--:-- 11739
     % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
   100 1518k  100 1518k    0     0  8930k      0 --:--:-- --:--:-- --:--:-- 8930k
     % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
   100 3304k  100 3304k    0     0  11.4M      0 --:--:-- --:--:-- --:--:-- 11.4M
     % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
   100 1679k  100 1679k    0     0  9489k      0 --:--:-- --:--:-- --:--:-- 9489k
     % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
   100 3523k  100 3523k    0     0  13.0M      0 --:--:-- --:--:-- --:--:-- 13.0M
     % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
   100 1086k  100 1086k    0     0  7103k      0 --:--:-- --:--:-- --:--:-- 7103k
     % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
   100 2578k  100 2578k    0     0  10.2M      0 --:--:-- --:--:-- --:--:-- 10.2M
   
   Note(s):
    - created input file directory at /Users/ashleychontos/pysyd/info 
    - saved an example of a star list
    - saved an example for the star information file
    - created data directory at /Users/ashleychontos/pysyd/data 
    - example data saved to data directory
    - results will be saved to /Users/ashleychontos/pysyd/results
   
    
As shown above, example data and other relevant files were downloaded from the 
`public GitHub repo <https://github.com/ashleychontos/pySYD>`_. 

If you forget or accidentally happen to run this again (in the same directory), 
you will get the following *lovely* reminder:

.. code-block::

   pysyd setup -v
   
   Looks like you've probably done this
   before since you already have everything!
   

-----

.. _install-quickstart:

Quickstart
##########

Use the following to get up and running right away: 

.. code-block::

   python -m pip install pysyd
   mkdir pysyd
   cd pysyd
   pysyd setup [optional]

The last command which will provide you with example data and files to immediately get 
going. This is essentially a summary of all the steps discussed on this page but a more
consolidated version.

*You are now ready to do some asteroseismology!*

p.s. enter `pysyd fun` for a little surprise:

.. code-block::

   $ pysyd fun   
                                               |                                              
                                               |                                              
                                               |   |                                          
                                               |   |                                          
                                               |   |                                          
                                          |    ||  |                                          
                                          |    ||  |            |                             
                                |         |    ||  |    |       |                             
                                |  |      |    ||  |    |       |                             
                                |  |      |   |||  |    |       |                             
                                |  |      |   |||  ||   |       |                             
                    |           |  ||     |   |||  ||   |    |  |                             
                    |     |    ||  ||   | |   |||  ||   |   ||  |      |                      
                    |     |    ||  ||   | ||  |||  ||   |   ||  ||     |                      
                    |     |    ||  ||   | ||  |||  ||   ||  ||  ||     |                      
                    |     |    ||  ||  || ||  |||  ||   ||  ||  || |   ||   |                 
                    |    ||    ||  ||| || ||  |||  ||   ||  ||  || |   ||   |                 
             |  |  ||    ||    ||  ||| || ||  ||| |||   ||  ||  || || |||   |     | |         
       |     |  |  ||    ||   |||  ||| || ||  ||| ||||  || ||| ||| || |||   |     | |    |    
       ||   || ||  ||   ||||  |||| ||| || ||  ||| |||| ||| ||| ||| || |||  |||   || ||   ||   
       ||   || ||  |||  ||||  |||| ||| || || |||| |||| ||| ||||||| || |||  ||| | || ||| |||   
      |||  ||| || ||||  ||||  |||| ||| ||||| |||| |||| ||| ||||||| ||||||| ||| | || ||| ||||  
     ||||  ||| || ||||| ||||| |||| ||||||||| |||| |||| ||||||||||| ||||||| ||| | |||||| ||||  
     ||||||||| |||||||| ||||| |||| ||||||||| ||||||||| ||||||||||||||||||| ||||| |||||||||||| 
    |||||||||| ||||||||||||||||||||||||||||| ||||||||||||||||||||||||||||||||||| |||||||||||| 
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 


-----
