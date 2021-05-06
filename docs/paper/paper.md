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
properties [@aerts2021]. Specifically for stars that are similar to the Sun, turbulent near-surface convection 
excites sound waves that propagate within the stellar cavity [@bedding2014]. These waves penetrate 
into different depths within the star and therefore provide powerful constraints on stellar interiors that would 
otherwise be inaccessible. Asteroseismology is well-established in astronomy as the gold standard for 
characterizing fundamental properties like masses, radii, densities, and ages for single stars, which has
broad impacts on several fields in astronomy. For example, ages of stars are important to reconstruct the 
formation history of the Milky Way (so-called galactic archeology). For exoplanets that are measured indirectly 
through changes in stellar observables, precise and accurate stellar masses and radii are critical for 
learning about the planets that orbit them.

# Statement of Need

Thanks to *Kepler*, K2 and TESS, we now have very large data volumes that require automated software tools
to extract observables. Several tools have been developed for asteroseismic analyses [e.g., `A2Z`, see @mathur2010; 
`COR`, see @mosser2009; `OCT`, see @hekker2010], but nearly all of them are closed-source and therefore inaccessible to 
the general astronomy community. Some open-source tools exist [e.g., `DIAMONDS`, see @corsaro2014; `lightkurve`, see
@lightkurve], but they are either not optimized for large samples of stars or have not been extensively tested 
against closed-source tools.

`pySYD` is adapted from the framework of the IDL-based `SYD` pipeline [@huber2009], which was extensively used 
to measure asteroseismic parameters for Kepler stars. Papers based on asteroseismic parameters measured using the 
`SYD` pipeline include @huber2011, @chaplin2014, @serenelli2017, and @yu2018. `pySYD` was developed using the same 
well-tested methodology, but has improved functionality including automated background model selection 
and parallel processing as well as improved flexibility through a user-friendly interface, while still 
maintaining its speed and efficiency. Well-documented, open-source asteroseismology software that has been 
benchmarked against closed-source tools are critical to ensure the reproducibility of legacy results from 
the *Kepler* mission. The combination of well-tested methodology, improved flexibility and parallel processing
capabilities will also make `pySYD` a promising tool for the broader community to analyze current and 
forthcoming data from the NASA TESS mission.

# The `pySYD` software

`pySYD` is a Python package for detecting solar-like oscillations and measuring global asteroseismic 
parameters. Derived parameters include the frequency of maximum power (numax) and the large frequency spacing
(dnu), as well as characteristic amplitues and timescales of correlated red-noise signals present in a 
power spectrum which are due to stellar granulation processes.

A `pySYD` pipeline `Target` class object has two main methods:
- The first module searches for the power excess due to solar-like oscillations by implementing a collapsed 
  autocorrelation function using different bin sizes. The main purpose for the first module is to provide a 
  good starting point for the second module, which is when all the parameters are estimated. The output from 
  this routine provides an estimate for numax, which is translated into a frequency range in the power spectrum 
  exhibiting the solar-like oscillations.
- Before global parameters like numax and dnu are estimated, the second module will first optimize the global 
  fit by selecting the best-fit stellar background model based on a reduced chi-squared analysis. 
  
The `pySYD` software was built using a number of powerful libraries, including Astropy [@astropy1,@astropy2], 
Matplotlib, Numpy, and SciPy [@scipy].

# Documentation & Examples

The main documentation for the `pySYD` software is hosted by [ReadTheDocs](https://readthedocs.org) at
[pysyd.readthedocs.io](https://pysyd.readthedocs.io). `pySYD` provides a convenient setup feature that 
will download data for three example stars and automatically create the recommended files for an easy 
quickstart. The features of the `pySYD` output results are described in detail in the documentation.

# Acknowledgements

We acknowledge contributions from Dennis Stello, Jie Yu, Pavadol Yamsiri, and other users of the `SYD` pipeline.

We also acknowledge support from the Alfred P. Sloan Foundation, the National Aeronatuics and Space Administration
(80NSSC19K0597), and the National Science Foundation (AST-1717000, DGE-1842402).

# References
