.. _setup/index:

**********
Setting up
**********

Ok now that you have properly installed and tested the software, let's get started!

Make a local directory
######################

While `pip` installed ``pySYD`` to your `PYTHONPATH`, we recommend that you first 
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
################

The ``pySYD`` package comes with a convenient setup feature (accessed via
:ref:`pysyd.pipeline.setup<library/pipeline>`) which can be ran from the command 
line in a single step. 

We ***strongly encourage*** you to run this step regardless of how you intend to 
use the software because it:

- downloads data for three example stars
- provides the example [optional] input files to use with the software *and* 
- sets up the recommended local directory structure

The only thing you need to do from your end is initiate the command -- which now 
that you've created a local pysyd directory -- all you have to donown is jump into 
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
