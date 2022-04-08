.. _library-about:

******************
About the software
******************

`pySYD` was initially established as a pure translation of the `IDL`-based ``SYD`` pipeline 
[huber2009]_. In the *Kepler* days, `SYD` was extensively used to measure :term:`global asteroseismic parameters` 
for many stars. Papers based on parameters measured by ``SYD`` include many ensemble
studies, including [huber2011]_, [chaplin2014]_, [serenelli2017]_, and [yu2018]_.

In order to process and analyze the enormous amounts of data from *Kepler* in real time, there were a
:ref:`a handful of other closed-source pipelines <library-about-related>` developed around the same time that perform roughly
similar types of analyses. In fact, there were several papers that compared results from each
of these pipelines in order to ensure the reproducibility of science results from the :term:`Kepler legacy sample`.

`pySYD` adapts the well-tested methodology from `SYD` while also improving these 
existing analyses and expanding upon numerous new features. Some of the improvements include:

- Automated best-fit background model selection
- Parallel processing
- Easily accessible + command-line friendly interface
- Ability to save samples for further analyses

.. TODO:: implement seeds for any random parts of the pipeline for reproducibility purposes

-----

.. _library-about-benchmark:

Benchmarking to the *Kepler* legacy sample
##########################################

We ran `pySYD` on ~100 *Kepler* legacy stars (defined :term:`here <Kepler legacy sample>`) observed in short-cadence and compared 
the output to ``SYD`` results from `Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_. 
The same time series and power spectra were used for both analyses, which are publicly available
and hosted online c/o KASOC [#]_. The resulting values are compared for the two methods below for 
:term:`numax` (:math:`\rm \nu_{max}`, left) and :term:`dnu` (:math:`\Delta\nu`, right). 

.. image:: ../_static/comparison.png
  :width: 680
  :alt: Comparison of the `pySYD` and `SYD` pipelines

The residuals show no strong systematics to within <0.5% in Dnu and <~1% in numax, which 
is smaller than the typical random uncertainties. This confirms that the open-source `Python` 
package ``pySYD`` provides consistent results with the legacy IDL version that has been 
used extensively in the literature.

.. TODO:: Add script or jupyter notebook to reproduce this figure.

-----

.. _library-about-related:

Related Tools
#############

``pySYD`` provides general purpose tools for performing asteroseismic analysis in the frequency domain.
Several tools have been developed to solve related scientific and data analysis problems. We have compiled 
a list of software packages that performs similar or complementary analyses.

 * ``AMP``:
   - language: 
   - reference:
   - documentation: no
   - publicly available: no
   - requires license: n/a

 * ``A2Z``: determining global parameters of the oscillations of solar-like stars
    - language: `?`
    - reference: yes <https://ui.adsabs.harvard.edu/abs/2010A%26A...511A..46M>
    - documentation: no
    - publicly available: no
    - requires license: n/a

* ``Background``: an extension of ``DIAMONDS`` that fits the background signal of solar-like oscillators 
   - language: `c++11`
   - reference: no
   - documentation: no
   - publicly available: yes <https://github.com/EnricoCorsaro/Background>
   - requires license: no

* ``CAN``: on the detection of Lorentzian profiles in a power spectrum
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2009A%26A...506.1043G>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``COR``: on detecting the large separation in the autocorrelation of stellar oscillation times series
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2009A%26A...508..877M>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``DIAMONDS``: high-DImensional And multi-MOdal NesteD Sampling
   - language: `c++11`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..71C>
   - documentation: yes <https://diamonds.readthedocs.io/en/latest/>
   - publicly available: yes <https://github.com/EnricoCorsaro/DIAMONDS>
   - requires license: n/a

* ``DLB``:
   - language: ``?``
   - reference: no
   - documentation: n/a
   - publicly available: no
   - requires license: n/a 

* ``FAMED``: Fast & AutoMated pEakbagging with Diamonds
   - language: `IDL` (currently being developed in `Python`)
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2020A%26A...640A.130C>
   - documentation: yes <https://famed.readthedocs.io/en/latest/>
   - publicly available: yes <https://github.com/EnricoCorsaro/FAMED>
   - requires license: yes

* Flicker Flipper?: 
   - language:
   - reference:
   - documentation: 
   - publicly available: 
   - requires license: n/a

* ``KAB``: automated asteroseismic analysis of solar-type stars
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2010arXiv1003.4167K>
   - documentation: no
   - publicly available: no
   - requires license: n/a
  
* ``lightkurve``: a friendly Python package for making discoveries with *Kepler* & TESS
   - language: `Python`
   - reference: no
   - documentation: yes <https://docs.lightkurve.org>
   - publicly available: yes <https://github.com/lightkurve/lightkurve>
   - requires license: no 

* ``OCT``: automated pipeline for extracting oscillation parameters of solar-like main-sequence stars
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2010MNRAS.402.2049H>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``ORK``: using the comb response function method to identify spacings
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2008ApJ...676.1248B>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``QML``: a power-spectrum autocorrelation technique to detect global asteroseismic parameters
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2011arXiv1104.0631V>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``PBjam``: a python package for automating asteroseismology of solar-like oscillators
   - language: `Python`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2021AJ....161...62N>
   - documentation: yes <https://pbjam.readthedocs.io/en/latest/>
   - publicly available: yes <https://github.com/grd349/PBjam>
   - requires license: no 

* ``SYD``: automated extraction of oscillation parameters for *Kepler* observations of solar-type stars
   - language: `IDL`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H>
   - documentation: no
   - publicly available: no
   - requires license: yes


.. important:: 

    If your software is not listed or if something listed is incorrect/missing, please 
    open a pull request to add it, we aim to be inclusive of all *Kepler*-, K2- and TESS-
    related tools!

-----

References
##########

.. [#] `Kepler Asteroseismic Science Operations Center <https://kasoc.phys.au.dk>`
.. [huber2009] `Huber et al. (2009) <https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H>`_
.. [huber2011] `Huber et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H>`_
.. [chaplin2014] `Chaplin et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJS..210....1C>`_
.. [serenelli2017] `Serenelli et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S>`_
.. [yu2018] `Yu et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018ApJS..236...42Y>`_
.. [lund2017] `Lund et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJ...835..172L>`_
.. [silva2017] `Silva Aguirre et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJ...835..173S>`_


.. bibliography:: ../references.bib

   verner2011
   boole1854