**********
Setting up
**********

The ``pySYD`` package comes with a convenient setup feature that can be run 
from command line in a single step. We strongly encourage you to run this step
regardless of how you intend to use ``pySYD`` (i.e., command line, notebook) 
because it:

#. downloads data for three example stars,
#. provides the example input files to use along with the software, *and* 
#. sets up the recommended directory structure

Therefore, the only thing you need to do from your end is initiate the command
and let ``pySYD`` do the rest for you!

Make a local directory
######################

While `pip` installed ``pySYD`` in your `PYTHONPATH`, we recommend that you first 
create a local pysyd directory before running setup. This is the
only reason we didn't include our examples as package data, as it would've put them
in your root directory and we realize this can be difficult to find.

.. code-block::
    
    mkdir ~/path/to/local/pysyd/directory
    
This way you can keep all your pysyd-related data, results and information in an 
easy-to-find location. You also do not have to worry about restricted access and
all that other jazz.

``pySYD`` setup
################

Now that you've created a local pysyd directory, all you have to do now is
jump into that directory and run the one liner: 

.. code-block::

    cd ~/path/to/local/pysyd/directory
    pysyd setup

As alluded to before, this step will create some relative directory structure that
might be useful to know. So instead, run it with the :term:`--verbose<-v, --verbose>`
command so you can see what is being downloaded and where it is being downloaded
from.

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


Structure
#########
