---
title: "`pySYD`: Automated measurements of global asteroseismic parameters"
tags:
  - Python
  - astronomy
  - stellar astrophysics
  - asteroseismology
  - stellar oscillations
  - fundamental stellar properties
  - solar-like oscillations
  - global asteroseismology
authors:
 - name: Ashley Chontos
   orcid: 0000-0003-1125-2564
   affiliation: "1, 2"
 - name: Daniel Huber
   orcid: 0000-0001-8832-4488
   affiliation: 1
 - name: Maryum Sayeed 
   orcid: 0000-0001-6180-8482
   affiliation: 1
affiliations:
 - name: Institute for Astronomy, University of Hawai'i, 2680 Woodlawn Drive, Honolulu, HI 96822, USA
   index: 1
 - name: NSF Graduate Research Fellow
   index: 2
date: 6 May 2021
bibliography: paper.bib
---

# Summary

Asteroseismology, the study of stellar oscillations, is a powerful tool for determining fundamental stellar 
properties [@aerts2021]. Specifically for stars that are similar to the Sun, turbulent 
near-surface convection excites sound waves that propagate within the stellar cavity. These waves penetrate 
into different depths within the star and therefore provide powerful constraints on stellar interiors that would 
otherwise be inaccessible. Asteroseismology is well-established in astronomy as the gold standard for 
characterizing fundamental properties like masses, radii, densities, and ages for single stars, which has
broad impact on several fields in astronomy. For example, ages of stars are important to reconstruct the 
formation history of the Milky Way (so-called galactic archeology). For exoplanets that are measured indirectly 
through changes in stellar observables, precise and accurate stellar masses and radii are critical for 
learning about these planets.

# Statement of Need

Thanks to *Kepler*, K2 and TESS, we now have very large data volumes that require automated software tools
to extract observables. Several tools have been developed for asteroseismic analyses (i.e. `OCT`, `COR`, `A2Z`), 
but nearly all of them are closed-source and therefore inaccessible to the general astronomy community. Some 
open-source tools exist (e.g., `DIAMONDS` and `lightkurve`), but they are either not optimized for large samples 
of stars or have not been extensively tested against closed-source tools.

`pySYD` is a Python package for detecting solar-like oscillations and measuring global asteroseismic 
parameters (e.g., numax, dnu, granulation amplitudes and timescales). 

`pySYD` is adapted from the framework of the IDL-based ``SYD`` pipeline [@huber2009], which was extensively used 
to measure asteroseismic parameters for Kepler stars. Papers based on asteroseismic parameters measured using the 
`SYD` pipeline include @huber2011, @chaplin2014, @serenelli2017, and @yu2018. `pySYD` was developed using the same 
well-tested methodology, but has improved functionality including automated background model selection 
and parallel processing as well as improved flexibility through a user-friendly interface, while still 
maintaining its speed and efficiency. Well-documented, open-source asteroseismology software that has been 
benchmarked against closed-source tools are critical to ensure the reproducibility of legacy results from 
the *Kepler* mission. The combination of well-tested methodology, improved flexibility, and parallel processing, 
will also make `pySYD` a promising tool for the broader community to analyze current and forthcoming data from
the NASA TESS mission.

# Acknowledgements

We acknowledge contributions from Dennis Stello, Jie Yu, Pavadol Yamsiri, and other users of the `SYD` pipeline.

We also acknowledge support from the Alfred P. Sloan Foundation, the National Aeronatuics and Space Administration
(80NSSC19K0597), and the National Science Foundation (AST-1717000, DGE-1842402).

# References
