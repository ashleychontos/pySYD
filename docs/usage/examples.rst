.. role:: bash(code)
   :language: bash

.. role:: underlined
   :class: underlined

.. _user-guide-examples:

**CLI examples**

This page has command-line examples for different software applications, including single
star examples and running an ensemble of stars. 

.. _user-guide-examples-single:

************
Single stars
************

For applications to single stars, we will start with a very easy, high signal-to-noise (SNR)
example, followed by medium and low SNR examples as well as a null detection. These examples 
will not be as detailed as the :ref:`quickstart example <quickstart-script>` -- our goal 
here is to provide pointers on what to look for in each case. 



.. _user-guide-examples-single-high:

:underlined:`High SNR: KIC 11618103`
####################################

KIC 11618103 is our most evolved example, an RGB star with numax of ~100 muHz.

.. image:: ../_static/examples/11618103_excess.png
  :width: 680
  :alt: KIC 11618103 estimates

.. image:: ../_static/examples/11618103_background.png
  :width: 680
  :alt: KIC 11618103 global fit

.. image:: ../_static/examples/11618103_samples.png
  :width: 680
  :alt: KIC 11618103 parameter posteriors


**For a full breakdown of what each panel is showing, please see :ref:`this page <library/output>` for more details.**
  
  
.. note::

    The sampling results can be saved by using the boolean flag ``-m`` or ``--samples``,
    which will save the posteriors of the fitted parameters for later use. 



-----

.. _user-guide-examples-single-medium:

:underlined:`Medium SNR: KIC 1435467`
#####################################

We used this example for new users just getting started and therefore we will only show
the output and figures. Feel free to visit that page :ref:`getting started <>`, which 
breaks down every step and output for this example.

KIC 1435467 is our least evolved example, with :math:`\rm \nu_{max} \sim 1300 \mu Hz`.

.. image:: ../_static/examples/1435467_estimates.png
  :width: 680
  :alt: KIC 1435467 estimates

.. image:: ../_static/examples/1435467_global.png
  :width: 680
  :alt: KIC 1435467 global fit

.. image:: ../_static/examples/1435467_samples.png
  :width: 680
  :alt: KIC 1435467 parameter posteriors


-----

.. _user-guide-examples-single-low:

:underlined:`Low SNR: KIC 8801316`
##################################

As if asteroseismology wasn't hard enough, let's make it even more difficult for you!

KIC 8801316 is a subgiant with a numax ~1100 muHz, shown in the figures below. 

.. image:: ../_static/examples/8801316_estimates.png
  :width: 680
  :alt: KIC 8801316 estimates

.. image:: ../_static/examples/8801316_global.png
  :width: 680
  :alt: KIC 8801316 global fit

.. image:: ../_static/examples/8801316_samples.png
  :width: 680
  :alt: KIC 8801316 parameter posteriors


This would be classified as a detection despite the low SNR due to the following reasons:

- there is a clear power excess as seen in panel 3
- the power excess has a Gaussian shape as seen in panel 5 corresponding to the solar-like oscillations
- the autocorrelation function (ACF) in panel 6 show periodic peaks
- the echelle diagram in panel 8 shows the ridges, albeit faintly


-----

.. _user-guide-examples-single-no:

:underlined:`No SNR: KIC 6278992`
#################################

KIC 6278992 is a main-sequence star with no solar-like oscillations.

.. image:: ../_static/examples/6278992_estimates.png
  :width: 680
  :alt: KIC 6278992 estimates

.. image:: ../_static/examples/6278992_global.png
  :width: 680
  :alt: KIC 6278992 global fit

.. image:: ../_static/examples/6278992_samples.png
  :width: 680
  :alt: KIC 6278992 parameter posteriors

-----

.. _user-guide-examples-multiple:

***********
Star sample
***********

Depending on how large your sample is, you may choose to do it one of two ways.

Regular mode
############

Since this is optimized for running many stars via command line, the star names will be read in 
and processed from `'info/todo.txt'` if nothing else is provided:

.. code-block::

    $ pysyd run


Parallel mode
#############

There is a parallel processing option included in the software, which is helpful for
running many stars. This can be accessed through the following command:

.. code-block::

    $ pysyd parallel 

For parallel processing, `pySYD` will divide and group the list of stars based on the 
available number of threads. By default, this value is `0` but can be specified via 
the command line. If it is *not* specified and you are running in parallel mode, 
``pySYD`` will use ``multiprocessing`` package to determine the number of CPUs 
available on the current operating system and then set the number of threads to this 
value (minus `1`).

If you'd like to take up less memory, you can easily specify the number of threads with
the :term:`--nthreads<--nt, --nthread, --nthreads>` command:

.. code-block::

    $ pysyd parallel --nthreads 10 --list path_to_star_list.txt
   
