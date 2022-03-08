.. _performance/index:

***********
Performance
***********

.. _comparison:


``pySYD`` vs IDL ``SYD``
***************************

We ran pySYD on ~100 Kepler legacy stars observed in short-cadence and compared the output to IDL SYD results from `Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_. The same time series and power spectra were used for both analyses.
The resulting values are compared for the two methods below for the frequency of maximum power 
(left) and the large frequency separation (Dnu) on the right. For reference,
``SYD == pySYD`` and ``IDL == SYD``.

.. image:: figures/performance/comparison.png
  :width: 680
  :alt: Comparison of ``pySYD`` and ``SYD``

There residuals show no strong systematics to within <0.5% in Dnu and <~1% in numax, which is smaller than the typical 
random uncertainties. This confirms that the open-source python package pySYD provides consistent results with the legacy 
IDL version that has been used extensively in the literature.

*** NOTE **** Add tutorial or jupyter notebook to reproduce this figure.


.. _speed:

Speed
*******
