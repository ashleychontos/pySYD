.. _performance:

Performance
###########

.. _comparison:

We tested pySYD on 30 short-cadence Kepler stars using 3 different time series lengths (1 month, 
3-11 months, 12+ months, see below), and compared the output to IDL SYD results from `2015 <https://ui.adsabs.harvard.edu/abs/2014ApJS..211....2H/abstract>`_.
The resulting values are compared for the two methods below, with frequency corresponding to maximum power 
(or numax) on the left and the characteristic frequency separation (or dnu) on the right. For reference,
``SYD == pySYD`` and ``IDL == SYD``.

.. image:: figures/comparison.png

There appears to be little to no systematics to at least <0.5% in Dnu and <~1% in numax, which is smaller than the typical 
random uncertainties. 