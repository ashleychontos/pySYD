*****
About
*****

When ``pySYD`` was first established, it was meant to be a pure translation of the 
`IDL`-based ``SYD`` pipeline `(Huber et al. 2009) <https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract>`_.
In the *Kepler* days, ``SYD`` was extensively used to measure :term:`global asteroseismic parameters` 
for many many stars. Papers based on parameters measured by ``SYD`` include many ensemble
studies, including 
`Huber et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract>`_, 
`Chaplin et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014ApJS..210....1C/abstract>`_, 
`Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_ 
and `Yu et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJS..236...42Y/abstract>`_.

In order to process and analyze the enormous amounts of data from *Kepler* in real time, there were a
handful of other closed-source pipelines developed around the same time that perform roughly
similar types of analyses. In fact, there were several papers that compared results from each
of these pipelines in order to ensure the reproducibility of science results from the :term:`Kepler legacy sample`

``pySYD`` vs ``SYD``
####################

We ran ``pySYD`` on ~100 *Kepler* legacy stars observed in short-cadence and compared 
the output to ``SYD``` results from `Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_. 
The same time series and power spectra were used for both analyses. The resulting values 
are compared for the two methods below for :term:`numax` (:math:`\rm \nu_{max}`, left) and 
:term:`dnu` (:math:`\Delta\nu`, right). 

.. image:: figures/performance/comparison.png
  :width: 680
  :alt: Comparison of ``pySYD`` and ``SYD``

The residuals show no strong systematics to within <0.5% in Dnu and <~1% in numax, which 
is smaller than the typical random uncertainties. This confirms that the open-source `Python` 
package ``pySYD`` provides consistent results with the legacy IDL version that has been 
used extensively in the literature.

*** NOTE **** Add tutorial or jupyter notebook to reproduce this figure.


Overview
########

When running the software, initialization of ``pySYD`` via command line will look in the following paths:

- ``INFDIR`` : '~/path_to_put_pysyd_stuff/info'
- ``INPDIR`` : '~/path_to_put_pysyd_stuff/data'
- ``OUTDIR`` : '~/path_to_put_pysyd_stuff/results'

which by default, is the absolute path of the current working directory (or however you choose to set it up). All of these paths should be ready to go
if you followed the suggestions in :ref:`structure` or used our ``setup`` feature.

A ``pySYD`` pipeline ``Target`` class object has two main function calls:

#. The first module :
    * **Summary:** a crude, quick way to identify the power excess due to solar-like oscillations
    * This uses a heavy smoothing filter to divide out the background and then implements a frequency-resolved, collapsed 
      autocorrelation function (ACF) using 3 different ``box`` sizes
    * The main purpose for this first module is to provide a good starting point for the
      second module. The output from this routine provides a rough estimate for numax, which is translated 
      into a frequency range in the power spectrum that is believed to exhibit characteristics of p-mode
      oscillations
#. The second module : 
    * **Summary:** performs a more rigorous analysis to determine both the stellar background contribution
      as well as the global asteroseismic parameters.
    * Given the frequency range determined by the first module, this region is masked out to model 
      the white- and red-noise contributions present in the power spectrum. The fitting procedure will
      test a series of models and select the best-fit stellar background model based on the BIC.
    * The power spectrum is corrected by dividing out this contribution, which also saves as an output text file.
    * Now that the background has been removed, the global parameters can be more accurately estimated. Numax is
      estimated by using a smoothing filter, where the peak of the heavily smoothed, background-corrected power
      spectrum is the first and the second fits a Gaussian to this same power spectrum. The smoothed numax has 
      typically been adopted as the default numax value reported in the literature since it makes no assumptions 
      about the shape of the power excess.
    * Using the masked power spectrum in the region centered around numax, an autocorrelation is computed to determine
      the large frequency spacing.

.. note::

    By default, both modules will run and this is the recommended procedure if no other information 
    is provided. 

    If stellar parameters like the radius, effective temperature and/or surface gravity are provided in the **info/star_info.csv** file, ``pySYD`` 
    can estimate a value for numax using a scaling relation. Therefore the first module can be bypassed,
    and the second module will use the estimated numax as an initial starting point.

    There is also an option to directly provide numax in the **info/star_info.csv** (or via command line, 
    see :ref:`advanced usage<advanced>` for more details), which will override the value found in the first module. This option 
    is recommended if you think that the value found in the first module is inaccurate, or if you have a visual 
    estimate of numax from the power spectrum.

