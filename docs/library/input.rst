******
Inputs
******

ICYMI: there is a very convenient way for you to get started right away -- and that is by
running the ``pySYD`` setup command. Running the command will provide all of the relevant 
files that are discussed on this page. 

If you happened to miss this on our quickstart page, :ref:`jump down <input/setup>` to see 
more information on exactly what the command does for you. 

Required
########

First and foremost, we need some data!

At the moment, we are implementing time-domain utilities (in that
we would love to calculate the power spectra for you) but it is not
ready yet!

We only realized later that some of these frequency analysis (and even
time-domain analysis) tools are not immediately obvious, especially not
for non-expert users -- which is who we want to use it! **so stay tuned!**

Data 
*****

For a given star, ID, the data : 

*  `'./data/ID_LC.txt'` : time series data in units of days
*  `'./data/ID_PS.txt'` : power spectrum in units of muHz versus power or power density

.. warning::

    It is **critical** that these files are in the proper units in order for ``pySYD`` 
    to work properly. 

Optional
########

Info
****

There are two main files that can be provided, but both are optional.

* `'./info/todo.txt'` : text file with one star ID per line, which must match the data ID. If no stars are specified via command line, the star(s) listed in the text file will be processed by default. This is recommended when running a large number of stars.

#. `'./info/star_info.csv'` : contains individual star information. Star IDs are crossmatched with this list and therefore, do not need to be in any particular order. In order to read information in properly, it **must** contain the following columns:

   * "stars" [int,str] : star IDs that should exactly match the star provided via command line or in todo.txt
   * "radius" [float] : stellar radius (units: solar radii)
   * "teff" [float] : effective temperature (units: K)
   * "logg" [float] : surface gravity (units: dex)
   * "numax" [float] : the frequency corresponding to maximum power (units: :math:`\rm \mu Hz`)
   * "lower_ex" [float] : lower frequency limit to use in the find_excess module (units: :math:`\rm \mu Hz`)
   * "upper_ex" [float] : upper frequency limit to use in the find_excess module (units: :math:`\rm \mu Hz`)
   * "lower_bg" [float] : lower frequency limit to use in the fit_background module (units: :math:`\rm \mu Hz`)
   * "upper_bg" [float] : upper frequency limit to use in the fit_background module (units: :math:`\rm \mu Hz`)
   * "seed" [int] : random seed generated when using the Kepler correction option, which is saved for future reproducibility purposes

.. TODO::

    Add all the available options (columns) to the csv 
    

Setup
#####

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
