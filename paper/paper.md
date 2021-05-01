---
title: 'pySYD: Automated extraction of global asteroseismic parameters'
tags:
  - Python
  - astronomy
  - stellar astrophysics
authors:
 - name: Ashley Chontos^[corresponding author: achontos@hawaii.edu]
   orcid: 0000-0003-1125-2564
   affiliation: 1
 - name: Daniel Huber
   orcid: 0000-0001-8832-4488
   affiliation: 1
 - name: Maryum Sayeed 
   affiliation: 1
affiliations:
 - name: Institute for Astronomy, University of Hawaii, 2680 Woodlawn Drive, Honolulu, HI 96822, USA
   index: 1
date: 1 May 2021
bibliography: paper/paper.bib

---

# Summary

Asteroseismology, the study of stellar oscillations, is a powerful tool for determining fundamental stellar 
properties `[@Aerts:2011]`. Specifically for stars that are similar to the Sun, turbulent 
near-surface convection excites sound waves that propagate within the stellar cavity. Different waves probe 
different regions of the star and therefore, provide powerful constraints on stellar interiors that would 
otherwise be inaccessible. Asteroseismology is well-established in astronomy as the gold standard for 
characterizing fundamental properties like stellar masses, radii, densities, and ages. For galactic archeology, 
ages of stars are important to reconstruct the formation history of the Milky Way. For exoplanets that are 
measured indirectly through changes in stellar observables, precise and accurate stellar masses and radii 
are critical for learning about these planets.

Several tools have been developed for asteroseismic analyses (i.e. `OCT`, `COR`, `A2Z`), but nearly all of 
them are closed-source and therefore inaccessible to the general astronomy community. Some open-source tools 
exist (e.g., `DIAMONDS` and `lightkurve`), but they are either not optimized for large samples of stars or 
have not been extensively tested against closed-source tools. Additionally, thanks to *Kepler*, K2 and TESS, 
we now have very large data volumes that require automated software tools to extract observables. 

`pySYD` is a Python package for detecting solar-like oscillations and measuring global asteroseismic 
parameters (e.g., numax, dnu, granulation amplitudes and timescales). `pySYD` is adapted from the framework 
of the IDL-based ``SYD`` pipeline `[@Huber:2009]`, which was extensively used to measure asteroseismic parameters 
for Kepler stars. Papers based on asteroseismic parameters measured using the `SYD` pipeline include 
`@Huber:2011`, `@Chaplin:2014`, `@Serenelli:2017`, and `@Yu:2018`. `pySYD` was developed using the same 
well-tested methodology, but has improved functionality including automated background model selection 
and parallel processing as well as improved flexibility through a user-friendly interface, while still 
maintaining its speed and efficiency. Well-documented, open-source asteroseismology software that has been 
benchmarked against closed-source tools are critical to ensure the reproducibility of legacy results from 
the *Kepler* mission.

`pySYD` was developed to extend the userbase to expert and non-expert researchers alike. `pySYD` is already 
being used by students that are new to astronomy and has already aided in new asteroseismic detections in
over 100 stars. Additionally, there are already several manuscripts in preparation that are featuring results 
from the software. The combination of well-tested methodology, improved flexibility, and parallel processing, 
`pySYD` is a promising tool for experts and non-experts alike, which will be especially exciting for forthcoming 
data releases from the TESS extended mission.


# Acknowledgements

We acknowledge contributions from Pavadol Yamsiri, and Dennis Stello, as well as support from NASA and the 
National Science Foundation.

# References
