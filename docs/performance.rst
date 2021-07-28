.. _performance:

Performance
###########

.. _comparison:

We tested pySYD on 30 Kepler stars observed in short-cadence for 3 different time series lengths (1 month, 
3-11 months, 12+ months) and compared the output to IDL SYD results from `Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_. The same time series and power spectra were used for both analyses.
The resulting values are compared for the two methods below for the frequency of maximum power 
(left) and the large frequency separation (Dnu) on the right. For reference,
``SYD == pySYD`` and ``IDL == SYD``.

.. image:: figures/new.png

There residuals show no strong systematics to within <0.5% in Dnu and <~1% in numax, which is smaller than the typical 
random uncertainties. This confirms that the open-source python implementation of the SYD pipeline presented here provides consistent results with the legacy IDL version that has been used in the literature.

*** NOTE **** Add tutorial or jupyter notebook to reproduce this figure.
