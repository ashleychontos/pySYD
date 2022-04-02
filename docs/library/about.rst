*****
About
*****

``pySYD`` was initially established as a pure translation of the `IDL`-based ``SYD`` pipeline 
`(Huber et al. 2009) <https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract>`_.
In the *Kepler* days, ``SYD`` was extensively used to measure :term:`global asteroseismic parameters` 
for many stars. Papers based on parameters measured by ``SYD`` include many ensemble
studies, including 
`Huber et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract>`_, 
`Chaplin et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014ApJS..210....1C/abstract>`_, 
`Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_ 
and `Yu et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJS..236...42Y/abstract>`_.

In order to process and analyze the enormous amounts of data from *Kepler* in real time, there were a
:ref:`a handful of other closed-source pipelines <vision/related>` developed around the same time that perform roughly
similar types of analyses. In fact, there were several papers that compared results from each
of these pipelines in order to ensure the reproducibility of science results from the :term:`Kepler legacy sample`.

``pySYD`` adapts the well-tested methodology from ``SYD`` while also improving these 
existing analyses and expanding upon numerous new features. Some of the improvements include:

- Automated best-fit background model selection
- Parallel processing
- Easily accessible + command-line friendly interface
- Ability to save samples for further analyses

.. TODO:: implement seeds for any random parts of the pipeline for reproducibility purposes

``pySYD`` vs ``SYD``
####################

We ran `pySYD` on ~100 *Kepler* legacy stars (defined :term:`here <Kepler legacy sample>`) observed in short-cadence and compared 
the output to ``SYD`` results from `Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_. 
The same time series and power spectra were used for both analyses. The resulting values 
are compared for the two methods below for :term:`numax` (:math:`\rm \nu_{max}`, left) and 
:term:`dnu` (:math:`\Delta\nu`, right). 

.. image:: ../_static/comparison.png
  :width: 680
  :alt: Comparison of the `pySYD` and `SYD` pipelines

The residuals show no strong systematics to within <0.5% in Dnu and <~1% in numax, which 
is smaller than the typical random uncertainties. This confirms that the open-source `Python` 
package ``pySYD`` provides consistent results with the legacy IDL version that has been 
used extensively in the literature.

.. TODO:: Add script or jupyter notebook to reproduce this figure.


Other related packages
######################

`pySYD` provides general purpose tools for performing asteroseismic analysis in the frequency domain.
Several tools have been developed to solve related scientific and data analysis problems. We have compiled 
a list of software packages that performs similar or complementary analyses.

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

Overview
########

When running the software,  ``pySYD`` will look in the following paths:

- ``INFDIR`` : '~/path/to/local/pysyd/directory/info'
- ``INPDIR`` : '~/path/to/local/pysyd/directory/data'
- ``OUTDIR`` : '~/path/to/local/pysyd/directory/results'

which by default, is the absolute path of the current working directory (think wherever you
ran setup from).

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

