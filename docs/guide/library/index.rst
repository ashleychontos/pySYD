.. module:: pysyd

*****************
``pySYD`` library
*****************

We are excited to share the software with you all!


.. toctree::
   :titlesonly:
   :maxdepth: 2
   :caption: How it works
   :hidden:

   intro
   input
   pipeline
   target
   models	      
   plots
   output
   Utility functions <utils>


About
#####

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

``pySYD`` adapts the well-tested methodology from ``SYD`` while also improving these 
existing analyses and expanding upon numerous new features. Improvements include:

- Automated best-fit background model selection
- Parallel processing
- Easily accessible + command-line friendly interface
- Ability to save samples for further analyses

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
