.. role:: bash(code)
   :language: bash

.. _usage/cli/examples:

*************
More examples
*************

If the pysyd setup feature was successfully executed, there should now be light curves and power spectra 
for three KIC stars in the input data directory. 

If so, then you are ready to get started! 

This page has command-line examples for the following cases:
 -  :ref:`single star applications <example/single>` of varying signal-to-noise (SNR) detections:
     -  :ref:`High SNR <usage/cli/examples/single/high>`
     -  :ref:`Medium SNR <usage/cli/examples/single/medium>`
     -  :ref:`Low SNR <usage/cli/examples/single/low>`
     -  :ref:`No SNR <usage/cli/examples/single/no>`

 -  running :ref:`many stars <usage/cli/examples/multiple>`
 -  :ref:`advanced examples <usage/cli/examples/advanced>` for special commands 

including what to look for in each case.

-----

.. _usage/cli/examples/single:

A single star
#############

For applications to single stars, we will start with a very easy, high signal-to-noise (SNR)
example, followed by medium, low, and no SNR examples. These won't be detailed walkthroughs 
but we are hoping to give pointers about what to look for in each class of examples.

.. _usage/cli/examples/single/high:

High SNR: KIC 11618103
**********************

KIC 11618103 is our most evolved example, an RGB star with numax of ~100 muHz.

.. image:: _static/examples/11618103_excess.png
  :width: 680
  :alt: Find excess output plot for KIC 11618103.

.. image:: _static/examples/11618103_background.png
  :width: 680
  :alt: Fit background output plot for KIC 11618103.

.. image:: _static/examples/11618103_samples.png
  :width: 680
  :alt: Distributions of Monte-Carlo samples for KIC 11618103.


**For a full breakdown of what each panel is showing, please see :ref:`this page <library/output>` for more details.**
  
  
.. note::

    The sampling results can be saved by using the boolean flag ``-m`` or ``--samples``,
    which will save the posteriors of the fitted parameters for later use. 
    
And just like that, you are now an asteroseismologist!

-----

.. _usage/cli/examples/single/medium:

Medium SNR: KIC 1435467
***********************

We used this example for new users just getting started and therefore we will only show
the output and figures. Feel free to visit that page :ref:`getting started <>`, which 
breaks down every step and output for this example.

KIC 1435467 is our least evolved example, with :math:`\rm \nu_{max} \sim 1300 \mu Hz`.

.. image:: _static/examples/1435467_estimates.png
  :width: 680
  :alt: Find excess output plot for KIC 11618103.

.. image:: _static/examples/1435467_global.png
  :width: 680
  :alt: Fit background output plot for KIC 11618103.

.. image:: _static/examples/1435467_samples.png
  :width: 680
  :alt: Distributions of Monte-Carlo samples for KIC 11618103.


-----

.. _usage/cli/examples/single/low:

Low SNR: KIC 8801316
********************

As if asteroseismology wasn't hard enough, let's make it even more difficult for you!

KIC 8801316 is a subgiant with a numax ~1100 muHz, shown in the figures below. 

.. image:: _static/examples/8801316_excess.png
  :width: 680
  :alt: Numax estimate KIC 8801316.

.. image:: _static/examples/8801316_background.png
  :width: 680
  :alt: Global fit for KIC 8801316.

This would be classified as a detection despite the low SNR due to the following reasons:

- there is a clear power excess as seen in panel 3
- the power excess has a Gaussian shape as seen in panel 5 corresponding to the solar-like oscillations
- the autocorrelation function (ACF) in panel 6 show periodic peaks
- the echelle diagram in panel 8 shows the ridges, albeit faintly

-----

.. _usage/cli/examples/single/no:

No SNR: KIC 6278992
*******************

KIC 6278992 is a main-sequence star with no solar-like oscillations.

.. image:: _static/examples/6278992_excess.png
  :width: 680
  :alt: Find excess output plot for KIC 6278992.

.. image:: _static/examples/6278992_background.png
  :width: 680
  :alt: Fit background output plot for KIC 6278992.

.. image:: _static/examples/6278992_samples.png
  :width: 680
  :alt: Distributions of Monte-Carlo samples for KIC 6278992.


-----

.. _usage/cli/examples/multiple:

*******************
Parallel processing
*******************

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

.. _usage/cli/examples/advanced:

**************
Advanced usage
**************


Below are examples of how to use specific ``pySYD`` features, as well as plots showing before and after results.

-----

``--dnu``: force dnu
********************


+-------------------------------------------------------+---------------------------------------------------------+
| Before                                                | After                                                   |
+=======================================================+=========================================================+
| Fix the dnu value if the desired dnu is not automatically selected by `pySYD`.                                  |
+-------------------------------------------------------+---------------------------------------------------------+
|:bash:`pysyd run --star 9512063 --numax 843`           |:bash:`pysyd run --star 9512063 --numax 843 --dnu 49.54` |
+-------------------------------------------------------+---------------------------------------------------------+
| .. figure:: _static/advanced/9512063_before.png       | .. figure:: _static/advanced/9512063_after.png          |
|    :width: 680                                        |    :width: 680                                          |
+-------------------------------------------------------+---------------------------------------------------------+

-----

``--ew``: excess width
**********************

+------------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                           | After                                                            |
+==================================================================+==================================================================+
| Changed the excess width in the background corrected power spectrum used to calculate the ACF (and hence dnu).                      |
+------------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run --star 9542776 --numax 900`                     | :bash:`pysyd run --star 9542776 --numax 900 --ew 1.5`            |
+------------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/advanced/9542776_before.png                  | .. figure:: _static/advanced/9542776_after.png                   |
|    :width: 680                                                   |    :width: 680                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+

-----

``--ie``: smooth echelle
************************

+------------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                           | After                                                            |
+==================================================================+==================================================================+
| Smooth echelle diagram by turning on the interpolation, in order to distinguish the modes for low SNR cases.                        |
+------------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run 3112889 --numax 871.52 --dnu 53.2`              | :bash:`pysyd run --star 3112889 --numax 871.52 --dnu 53.2 --ie`  |
+------------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/advanced/3112889_before.png                  | .. figure:: _static/advanced/3112889_after.png                   |
|    :width: 680                                                   |    :width: 680                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+

-----

``--kc``: *Kepler* correction
*****************************

+---------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                        | After                                                            |
+===============================================================+==================================================================+
| Remove *Kepler* artefacts from the power spectrum for an accurate numax estimate.                                                |
+---------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run --star 8045442 --numax 550`                  | :bash:`pysyd run --star 8045442 --numax 550 --kc`                |
+---------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/advanced/8045442_before.png               | .. figure:: _static/advanced/8045442_after.png                   |
|    :width: 680                                                |    :width: 680                                                   |
+---------------------------------------------------------------+------------------------------------------------------------------+

-----

``--lp``: lower frequency of power excess
*****************************************

+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Before                                                                   | After                                                                    |
+==========================================================================+==========================================================================+
| Set the lower frequency limit in zoomed in power spectrum; useful when an artefact is present close to the excess and cannot be removed otherwise.  |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| :bash:`pysyd run --star 10731424 --numax 750`                            | :bash:`pysyd run --star 10731424 --numax 750 --lp 490`                   |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| .. figure:: _static/advanced/10731424_before.png                         | .. figure:: _static/advanced/10731424_after.png                          |
|    :width: 680                                                           |    :width: 680                                                           |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+

-----

:term:`--npeaks<--peaks, --npeaks>`
###################################

+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Before                                                                   | After                                                                    |
+==========================================================================+==========================================================================+
| Change the number of peaks chosen in ACF; useful in low SNR cases where the spectrum is noisy and ACF has many peaks close to the expected dnu.     |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| :bash:`pysyd run --star 9455860`                                         | :bash:`pysyd run --star 9455860 --npeaks 10`                             |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| .. figure:: _static/advanced/9455860_before.png                          | .. figure:: _static/advanced/9455860_after.png                           |
|    :width: 680                                                           |    :width: 680                                                           |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+

-----

:term:`--numax`
###############

+--------------------------------------------------------+-------------------------------------------------------+
| Before                                                 | After                                                 |
+========================================================+=======================================================+
| Set the numax value if pySYD chooses the wrong excess in the power spectrum.                                   |
+--------------------------------------------------------+-------------------------------------------------------+
| :bash:`pysyd run --star 5791521`                       | :bash:`pysyd run --star 5791521 --numax 670`          |
+--------------------------------------------------------+-------------------------------------------------------+
| .. figure:: _static/advanced/5791521_before.png        | .. figure:: _static/advanced/5791521_after.png        |
|    :width: 680                                         |    :width: 680                                        |
+--------------------------------------------------------+-------------------------------------------------------+


-----

:term:`--upperx<--ux, --upperx>`
################################

+-----------------------------------------------------+-------------------------------------------------------+
| Before                                              | After                                                 |
+=====================================================+=======================================================+
| Set the upper frequency limit in power spectrum; useful when `pySYD` latches on to an artefact.             |
+-----------------------------------------------------+-------------------------------------------------------+
| :bash:`pysyd run --star 11769801`                   | :bash:`pysyd run --star 11769801 -ux 3500`            |
+-----------------------------------------------------+-------------------------------------------------------+
| .. figure:: _static/advanced/11769801_before.png    | .. figure:: _static/advanced/11769801_after.png       |
|    :width: 680                                      |    :width: 680                                        |
+-----------------------------------------------------+-------------------------------------------------------+

-----
