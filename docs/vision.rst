*******************************
Vision of the ``pySYD`` project
*******************************

The NASA space telescopes *Kepler*, K2 and TESS have recently
provided very large databases of high-precision light curves of stars.
By detecting brightness variations due to stellar oscillations, these
light curves allow the application of asteroseismology to large numbers
of stars, which requires automated software tools to efficiently extract
observables. 

Several tools have been developed for asteroseismic analyses, but many of 
them are closed-source and therefore inaccessible to the general astronomy 
community. Some open-source tools exist, but they are either optimized for 
smaller samples of stars or have not yet been extensively tested against 
closed-source tools. 


.. note::

    We've attempted to collect these tools in a :ref:`single place <related>` 
    for easy comparisons. Please let us know if we've somehow missed yours --
    we would be happy to add it!

 
Goals
#####

The initial vision of this project was intended to be a direct translation of 
the IDL-based ``SYD`` pipeline



Related Tools
#############

``pySYD`` provides general purpose tools for performing asteroseismic analysis in the frequency domain.
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