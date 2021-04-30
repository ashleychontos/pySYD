---
title: 'pySYD: Automated extraction of global asteroseismic parameters'
tags:
  - Python
  - astronomy
  - Kepler
  - solar-like
  - stellar astrophysics
  - asteroseismology
  - global asteroseismology
  - stellar oscillations
  - fundamental stellar properties
authors:
  - name: Ashley Chontos^[corresponding author: achontos@hawaii.edu]
    orcid: 0000-0003-1125-2564
    affiliation: "1, 2"
  - name: Daniel Huber
    orcid: 0000-0001-8832-4488
    affiliation: 1
affiliations:
 - name: Institute for Astronomy, University of Hawai'i, 2680 Woodlawn Drive, Honolulu, HI 96822, USA
   index: 1
 - name: NSF Graduate Research Fellow
   index: 2
date: 30 April 2021
bibliography: paper.bib
---

# Summary

Asteroseismology, or the study of stellar oscillations, is a powerful tool for 
determining fundamental properties of stars. Specifically for stars that are similar 
to the Sun, turbulent near-surface processes excite acoustic sound waves that propagate 
within the stellar cavity. Different waves probe different regions of the star and 
therefore, provide powerful constraints on stellar interiors that would otherwise be 
inaccessible. Asteroseismology is well-established in astronomy as a gold standard for 
stellar characterization, which has broader implications to fields like galactic archeology 
and planetary science. The field of asteroseismology is niche however, where most available 
tools to perform these analyses typically require licenses and an expert user.

# Statement of need

`pySYD` is a Python package for detecting solar-like oscillations and measuring global
asteroseismic parameters. `pySYD` is adapted from the framework of the IDL-based `SYD` 
pipeline [`@huber2009`], which was extensively used to measure asteroseismic parameters 
for Kepler stars. Papers based on asteroseismic parameters measured using the `SYD` 
pipeline include `@huber2011`, `@chaplin2014`, `@serenelli2017`, and `@yu2018`. `pySYD` 
was developed using the same well-tested methodology, but has improved functionality 
such as background model-fitting and selection, parallel processing and user-friendly 
interfaces while still maintaining speed and efficiency.

`pySYD` was designed with intent of extending its userbase to the non-expert 
researcher. In fact, `pySYD` has already been beneficial for students that are new to 
astronomy, including new detections in over 100 stars. Additionally, there are already 
several manuscripts in preparation that are featuring results from the software. The 
combination of well-tested methodology, improved flexibility, and parallel processing, 
`pySYD` is a promising tool for experts and non-experts alike, which will be especially 
exciting for forthcoming data releases from the *TESS* extended mission.

# Acknowledgements

We acknowledge contributions from Maryum Sayeed, Pavadol Yamsiri, and Dennis Stello, 
as well as support from NASA and the National Science Foundation.

# References
