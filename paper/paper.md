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
   affiliation: 3
 - name: Pavadol Yamsiri
   affiliation: 4
affiliations:
 - name: Institute for Astronomy, University of Hawai'i, 2680 Woodlawn Drive, Honolulu, HI 96822, USA
   index: 1
 - name: NSF Graduate Research Fellow
   index: 2
 - name: Department of Astronomy, Columbia University, Pupin Physics Laboratories, New York, NY 10027, USA
   index: 3
 - name: Sydney Institute for Astronomy, School of Physics, University of Sydney, NSW 2006, Australia
   index: 4
date: 1 Aug 2021
bibliography: paper.bib
---

# Summary

Asteroseismology is well-established in astronomy as the gold standard for determining 
precise and accurate fundamental stellar properties like masses, radii, and ages. Several 
tools have been developed for asteroseismic analyses but many of them are closed-source 
and therefore not accessible to the general astronomy community. Here we present `pySYD`, 
a Python package for detecting solar-like oscillations and measuring global asteroseismic 
parameters. `pySYD` was adapted from the IDL-based `SYD` pipeline, which was extensively 
used to measure asteroseismic parameters for *Kepler* stars. `pySYD` was developed using 
the same well-tested methodology and comes with several new improvements to provide 
accessible and reproducible results. Well-documented, open-source asteroseismology software 
that has been benchmarked against closed-source tools are critical to ensure the 
reproducibility of legacy results from the *Kepler* mission. Moreover, `pySYD` will also 
be a promising tool for the broader astronomy community to analyze current and forthcoming 
data from the NASA TESS mission.

# Introduction

The study of stellar oscillations is a powerful tool for studying the interiors 
of stars and determining their fundamental properties [@aerts2021]. For stars 
with temperatures that are similar to the Sun, turbulent near-surface convection 
excites sound waves that propagate within the stellar cavity [@bedding2014]. 
These waves probe different depths within the star and therefore, provide critical 
constraints for stellar interiors that would otherwise be inaccessible by other means. 

Asteroseismology of such "solar-like" oscillators provide precise fundamental 
properties like masses, radii, densities, and ages for single stars, which has 
broad impacts on several fields in astronomy. For example, ages of stars are 
important to reconstruct the formation history of the Milky Way (so-called 
galactic archaeology). For exoplanets that are discovered indirectly through 
changes in stellar observables, precise and accurate stellar masses and radii 
are critical for learning about the planets that orbit them.

The NASA space telescopes *Kepler* [@borucki2010] and TESS [@ricker2015]
have recently provided very large databases of high-precision light curves of 
stars. By detecting brightness variations due to oscillations, these 
light curves allow the application of asteroseismology to large numbers of stars, 
which requires automated software tools to efficiently extract observables. 

Several tools have been developed for asteroseismic analyses [e.g., `A2Z`, 
@mathur2010; `COR`, @mosser2009; `OCT`, @hekker2010; `SYD`, @huber2009], but 
many of them are closed-source and therefore inaccessible to the general 
astronomy community. Some open-source tools exist [e.g., `DIAMONDS` and `FAMED`, 
@corsaro2014; `PBjam`, @nielsen2021; `lightkurve`, @lightkurve], but they are 
either optimized for smaller samples of stars or have not yet been extensively 
tested against closed-source tools.

# Statement of need

There is a strong need within the astronomy community for an open-source 
asteroseismology tool that is 1) accessible, 2) reproducible, and 3) scalable, 
which  will only grow with the continued success of the NASA TESS mission. In 
this paper we present a Python tool that automatically detects solar-like 
oscillations and characterizes their properties, called `pySYD`, 
which prioritizes these three key aspects: 

- **Accessible.** The `pySYD` library and source directory are both 
  publicly available, hosted on the Python Package Index 
  ([PyPI](https://pypi.org/project/pysyd/)) and GitHub. The 
  [`pySYD` GitHub Page](https://github.com/ashleychontos/pySYD) 
  also serves as a multifaceted platform to promote community engagement 
  through discussion forums to communicate and share science, laying out 
  instructions to contribute and encourage inclusivity, and providing 
  a clear path for issue tracking. To facilitate future use and adaptations, 
  the [documentation](https://pysyd.readthedocs.io) includes a broad 
  spectrum of examples that showcase the versatility of the software. 
  Additionally, Python usage has become standard practice within the 
  community, which will promote integrations with complementary tools 
  like [`lightkurve`](https://docs.lightkurve.org) and [`echelle`](https://github.com/danhey/echelle). 
- **Reproducible.** `pySYD` implements a similar framework to the closed-source IDL-based 
  `SYD` pipeline [@huber2009], which has been used frequently to measure global asteroseismic 
  parameters for many *Kepler* stars [@huber2011;@chaplin2014;@serenelli2017;@yu2018] and has 
  been extensively tested against other closed-source tools [@verner2011;@hekker2011]. 
  \autoref{fig:benchmark} compares global parameter results from the `pySYD` and `SYD` pipelines 
  for $\sim100$ *Kepler* legacy stars, showing excellent agreement with no significant offsets. 
  In fact, the small amount of scatter is likely because `pySYD` is *not* a direct 1:1 translation, 
  incorporating many new custom features and software enhancements. In addition to the important 
  benchmark sample, `pySYD` ensures reproducible results for *every* locally-processed star by 
  saving and setting seeds for any randomly occurring analyses.
- **Scalable.** `pySYD` was developed for speed and efficiency. `pySYD` has more 
  than 50 optional commands that enable a customized analysis at the individual star 
  level and on average, takes less than a minute to complete a single star 
  (with sampling). The software also features parallel processing capabilities 
  and is therefore suitable for large samples of stars.

Well-documented, open-source asteroseismology software that has been benchmarked 
against closed-source tools are critical to ensure the reproducibility of legacy 
results from the *Kepler* mission. `pySYD` will also be a promising tool for the 
broader community to analyze current and forthcoming data from the NASA TESS mission.

# Software package overview

`pySYD` depends on a number of powerful libraries, including [`astropy`](https://www.astropy.org) 
[@astropy1;@astropy2], [`matplotlib`](https://matplotlib.org) [@matplotlib], [`numpy`](https://numpy.org) [@numpy], 
[`pandas`](https://pandas.pydata.org) [@pandas] and [`scipy`](https://scipy.org) [@scipy]. The software package is structured 
around the following main modules, details of which are described in the 
online package documentation:

- [`target`](https://pysyd.readthedocs.io/en/latest/library/target.html) includes 
  the `Target` class object, which is instantiated for every processed star and 
  roughly operates in the following steps:
  
    * checks for and loads in data for a given star and applies any relevant time- and/or 
      frequency-domain tools e.g., computing spectra, mitigating *Kepler* artefacts, etc.
    * searches for localized power excess due to solar-like oscillations and then estimates 
      its initial properties 
    * uses estimates to mask out that region in the power spectrum and implements an 
      automated background fitting routine that characterizes amplitudes ($\sigma$) and 
      characteristic time scales ($\tau$) of various granulation processes
    * derives global asteroseismic quantities $\rm \nu_{max}$ and $\Delta\nu$ from the 
      background-corrected power spectrum
    * performs Monte-Carlo simulations by drawing from a chi-squared distribution (with 2 dof) 
      to estimate uncertainties
       
- [`plots`](https://pysyd.readthedocs.io/en/latest/library/plots.html) includes all plotting routines
- [`models`](https://pysyd.readthedocs.io/en/latest/library/utils.html) comprises different 
  frequency distributions used to fit and model properties in a given power spectrum
- [`cli`](https://pysyd.readthedocs.io/en/latest/usage/intro.html) & 
  [`pipeline`](https://pysyd.readthedocs.io/en/latest/library/pipeline.html) are the main 
  entry points for command-line usage
- [`utils`](https://pysyd.readthedocs.io/en/latest/library/utils.html) includes a suite 
  of utilities such as the container class `Parameters`, which contains all default 
  parameters, or utility functions like binning data or finding peaks in a series of data 

![Comparison of global parameters $\rm \nu_{max}$ (left) and $\Delta\nu$ (right) 
measured by `pySYD` and `SYD` for $\sim100$ *Kepler* stars [@serenelli2017], with 
fractional residuals shown in the bottom panels. The comparison shows excellent 
agreement, with median offsets of $0.07 \pm 0.07\%$ for $\rm \nu_{max}$ and 
$0.004 \pm 0.008\%$ for $\Delta\nu$. Typical random errors for such measurements 
are 1-2 orders of magnitude larger.\label{fig:benchmark}](benchmark.png)

# Documentation

For installation instructions and package information, the main documentation 
for the `pySYD` software is hosted at [ReadTheDocs](https://pysyd.readthedocs.io/en/latest/). 
`pySYD` comes with a setup feature which will download information and data for 
three example stars and then establish the recommended, local directory structure. 
The documentation comprises a diverse range of applications and examples to 
make the software more accessible and adaptable. Tutorials include:

 - basic command-line examples for stars of varying signal-to-noise detections
 - customized command-line examples to showcase some of the new, optional features 
 - different ways to run a large number of stars
 - a notebook tutorial walkthrough of a single star from data to results
 - other notebook tutorials demonstrating the use of some optional commands 
   and/or software hacks

The documentation also contains a [complete list](https://pysyd.readthedocs.io/en/latest/usage/glossary.html) 
of all parameters, which includes everything from their object type, 
default value(s), and how it is stored within the package, as well as
relevant links or similar keyword arguments.

# Acknowledgements

We thank Dennis Stello, Tim Bedding, Marc Hon, Yifan Chen, Yaguang Li, and other
`pySYD` users for discussion and suggestions which helped with the development
of this software.

We also acknowledge support from: 
- The National Science Foundation (DGE-1842402)
- The National Aeronautics and Space Administration (80NSSC19K0597, 80NSSC21K0652)
- The Alfred P. Sloan Foundation

# References
