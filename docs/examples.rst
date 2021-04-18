.. _examples:

Examples
########

If ``pysyd setup`` was successfully executed, there should now be light curves and power spectra 
for three KIC stars in the **data/** directory. If so, then you are ready to test out the software!

====================

High SNR Examples
*****************

KIC 1435467
+++++++++++

KIC 1435467 is our least evolved example star, with numax ~1400 muHz.

pySYD ``find_excess`` results:

.. image:: figures/1435467_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 1435467.

pySYD ``fit_background`` results:

.. image:: figures/1435467_background.png
  :width: 600
  :alt: Fit background output plot for KIC 1435467.

pySYD ``sampling`` results:

.. image:: figures/1435467_samples.png
  :width: 600
  :alt: Posterior distributions for derived parameters of KIC 1435467.

====================

KIC 2309595
+++++++++++

KIC 2309595 is a subgiant, with numax ~650 muHz.

pySYD ``find_excess`` results:

.. image:: figures/2309595_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 2309595.

pySYD ``fit_background`` results:

.. image:: figures/2309595_background.png
  :width: 600
  :alt: Fit background output plot for KIC 2309595.

pySYD ``sampling`` results:

.. image:: figures/2309595_samples.png
  :width: 600
  :alt: Posterior distributions for derived parameters of KIC 2309595.

====================

KIC 11618103
++++++++++++

KIC 11618103 is an evolved RGB star, with numax of ~100 muHz.

pySYD ``find_excess`` results:

.. image:: figures/11618103_excess.png
  :width: 600
  :alt: Find excess output plot for KIC 11618103.

pySYD ``fit_background`` results:

.. image:: figures/11618103_background.png
  :width: 600
  :alt: Fit background output plot for KIC 11618103.

pySYD ``sampling`` results:

.. image:: figures/11618103_samples.png
  :width: 600
  :alt: Posterior distributions for derived parameters of KIC 11618103.


====================

Low SNR Examples
****************

TODO

Non-detection Examples
**********************

TODO (what to look for with non-detections)


Ensemble of Stars
*****************

If you are running OSX, and want to run an ensemble of stars in parallel, you 
may need to perform some additional installation steps. See ###.
