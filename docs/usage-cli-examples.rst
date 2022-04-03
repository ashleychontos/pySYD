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

For applications to single stars, we will start with a very easy, high signal-to-noise (SNR)
example, followed by medium, low, and no SNR examples. These won't be detailed walkthroughs 
but we are hoping to give pointers about what to look for in each class of examples.

High SNR: KIC 11618103
**********************

KIC 11618103 is our most evolved example, an RGB star with numax of ~100 muHz.

.. image:: ../../_static/examples/11618103_excess.png
  :width: 680
  :alt: Find excess output plot for KIC 11618103.

.. image:: ../../_static/examples/11618103_background.png
  :width: 680
  :alt: Fit background output plot for KIC 11618103.

.. image:: ../../_static/examples/11618103_samples.png
  :width: 680
  :alt: Distributions of Monte-Carlo samples for KIC 11618103.


**For a breakdown of what each panel is showing, please see :ref:<..library/output> for more details.**
  
  
.. note::

    The sampling results can be saved by using the boolean flag ``-m`` or ``--samples``,
    which will save the posteriors of the fitted parameters for later use. 
    
And just like that, you are now an asteroseismologist!

-----

Medium SNR: 
**********


-----

Low SNR: KIC 8801316
********************

As if asteroseismology wasn't hard enough, let's make it even more difficult for you!

KIC 8801316 is a subgiant with a numax ~1100 muHz, shown in the figures below. 

.. image:: ../../_static/examples/8801316_excess.png
  :width: 680
  :alt: Numax estimate KIC 8801316.

.. image:: ../../_static/examples/8801316_background.png
  :width: 680
  :alt: Global fit for KIC 8801316.

This would be classified as a detection despite the low SNR due to the following reasons:

- there is a clear power excess as seen in panel 3
- the power excess has a Gaussian shape as seen in panel 5 corresponding to the solar-like oscillations
- the autocorrelation function (ACF) in panel 6 show periodic peaks
- the echelle diagram in panel 8 shows the ridges, albeit faintly

-----

No SNR: KIC 6278992
*******************

KIC 6278992 is a main-sequence star with no solar-like oscillations.

.. image:: ../../_static/examples/6278992_excess.png
  :width: 680
  :alt: Find excess output plot for KIC 6278992.

.. image:: ../../_static/examples/6278992_background.png
  :width: 680
  :alt: Fit background output plot for KIC 6278992.

.. image:: ../../_static/examples/6278992_samples.png
  :width: 680
  :alt: Distributions of Monte-Carlo samples for KIC 6278992.


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