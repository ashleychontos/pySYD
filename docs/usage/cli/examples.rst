**************
Basic examples
**************

If the pysyd setup feature was successfully executed, there should now be light curves and power spectra 
for three KIC stars in the input data directory. 

If so, then you are ready to get started! 

Below we have basic examples for single star applications and also for running many stars in parallel.

-----

A single star
#############

For applications to single stars, we will start with an easy, very high signal-to-noise (SNR)
example, followed by medium and low SNR examples. 

High SNR
*********

KIC 11618103 is our most evolved example star, with numax ~1400 muHz. The following command:


runs KIC 1435467 using the default method, which first estimates numax before deriving any parameters. 
The reason for this is that the frequency region with power excess is masked out to better estimate 
background parameters.

 
As you can read in the text output, the example started with n=2 Harvey-like components but reduced to 1 
based on the BIC statistic. 

The first, optional routine that estimates numax creates the following output figure:


The derived parameters from the global fit are summarized below:



**For a breakdown of what each panel in each figure means, please see :ref:<..library/output> for more details.**
  
  
The derived parameters are saved to an output csv file but also printed at the end of the verbose output.
To quantify uncertainties in these parameters, we need to turn on the Monte Carlo sampling option (``--mc``) with::
  
  
.. note::

    The sampling results can be saved by using the boolean flag ``-m`` or ``--samples``,
    which will save the posteriors of the fitted parameters for later use. 
    
And just like that, you are now an asteroseismologist!

Medium SNR
**********



Low SNR
*******

As if asteroseismology wasn't hard enough, let's make it even more difficult for you!

-----

An ensemble of stars
####################

There is a parallel processing option included in the software, which is helpful for
running many stars. This can be accessed through the following command:

.. code-block::

    $ pysyd parallel 

For parallel processing, ``pySYD`` will divide and group the list of stars based on the 
available number of threads. By default, this value is `0` but can be specified via 
the command line. If it is *not* specified and you are running in parallel mode, 
``pySYD`` will use ``multiprocessing`` package to determine the number of CPUs 
available on the current operating system and then set the number of threads to this 
value (minus `1`).

If you'd like to take up less memory, you can easily specify the number of threads with
the :term:`--nthreads<--nt, --nthread, --nthreads>` command:

.. code-block::

    $ pysyd parallel --nthreads 10 --list path_to_star_list.txt

.. note::

    Remember that by default, the stars to be processed (i.e. todo) will read in from **info/todo.txt**
    if no ``-list`` or ``-todo`` paths are provided.
   
-----