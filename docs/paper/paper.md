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

Asteroseismology, the study of stellar oscillations, is a powerful tool for studying the interiors of stars
and determining their fundamental properties [@aerts2021]. For stars that are similar to the Sun, turbulent 
near-surface convection excites sound waves that propagate within the stellar cavity [@bedding2014]. These 
waves penetrate into different depths within the star and therefore provide powerful constraints on stellar 
interiors that would otherwise be inaccessible. Asteroseismology is well-established in astronomy as the 
gold standard for characterizing fundamental properties like masses, radii, densities, and ages for single 
stars, which has broad impacts on several fields in astronomy. For example, ages of stars are important to 
reconstruct the formation history of the Milky Way (so-called galactic archeology). For exoplanets that are 
discovered indirectly through changes in stellar observables, precise and accurate stellar masses and radii 
are critical for learning about the planets that orbit them.

# Statement of Need

The NASA space telescopes *Kepler*, K2 and TESS have recently provided very large databases of high-precision 
light curves of stars. By detecting brightness variations due to stellar oscillations, these light curves allow the 
application of asteroseismology to large numbers of stars, which requires automated software tools to efficiently 
extract observables. Several tools have been developed for asteroseismic analyses [e.g., `A2Z`, see @mathur2010; 
`COR`, see @mosser2009; `OCT`, see @hekker2010], but nearly all of them are closed-source and therefore inaccessible 
to the general astronomy community. Some open-source tools exist [e.g., `DIAMONDS`, see @corsaro2014; `PBjam`, 
see @nielsen2021; `lightkurve`, see @lightkurve], but they are either not optimized for large samples of stars or 
have not been extensively tested against closed-source tools.

`pySYD` is adapted from the framework of the IDL-based `SYD` pipeline [@huber2009], which was extensively used 
to measure asteroseismic parameters for Kepler stars. Papers based on asteroseismic parameters measured using the 
`SYD` pipeline include @huber2011, @bastien2013 @chaplin2014, @serenelli2017, and @yu2018. `pySYD` was developed 
using the same well-tested methodology, but has improved functionality including automated background model selection 
and parallel processing as well as improved flexibility through a user-friendly interface, while still 
maintaining its speed and efficiency. Well-documented, open-source asteroseismology software that has been 
benchmarked against closed-source tools are critical to ensure the reproducibility of legacy results from 
the *Kepler* mission. The combination of well-tested methodology, improved flexibility and parallel processing
capabilities will also make `pySYD` a promising tool for the broader community to analyze current and 
forthcoming data from the NASA TESS mission.

# The `pySYD` library

The excitation mechanism for solar-like oscillations is stochastic and modes are observed over a range of frequencies. 
Oscillation modes are separated by the so-called large frequency spacing (or $\Delta\nu$), with an approximately
Gaussian-shaped power excess centered on $\rm \nu_{max}$, the frequency of maximum power. The observables 
$\rm \nu_{max}$ and $\Delta\nu$ are directly related to fundamental properties such as surface gravity, density,
mass and radius.  

`pySYD` is a Python package for detecting solar-like oscillations and measuring global asteroseismic parameters. 
Derived parameters include $\rm \nu_{max}$ and $\Delta\nu$, as well as characteristic amplitudes and timescales 
of correlated red-noise signals due to stellar granulation.

A `pySYD` pipeline `Target` class object has two main methods:

- The first module searches for the power excess due to solar-like oscillations by implementing a frequency-resolved 
  collapsed autocorrelation method.  The output from this routine provides an estimate for $\rm \nu_{max}$. 
- The second module starts by optimizing and determining the best-fit stellar background model. The results from the 
  first module are translated into a frequency range in the power spectrum centered on the estimated $\rm \nu_{max}$,
  which is masked out to determine the stellar background contribution. After subtracting the best-fit model from 
  the power spectrum, the peak of the smoothed power spectrum is used to estimate $\rm \nu_{max}$. An autocorrelation 
  function (ACF) is computed using the region centered on $\rm \nu_{max}$, and used to calculate an estimate of
  $\Delta\nu$. 
  
The `pySYD` software was built using a number of powerful libraries, including Astropy [@astropy1;@astropy2], 
Matplotlib [@matplotlib], Numpy [@numpy], and SciPy [@scipy]. `pySYD` has been tested against `SYD` using 
results from the *Kepler* sample for differing time series lengths (\autoref{fig:comparison}). 
The comparison demonstrates that there are no systematic differences to at least $<0.5\%$ in $\Delta\nu$ and 
$\sim1\%$ in $\rm \nu_{max}$, which is smaller or comparable to the typical random uncertainties [@huber2011].

![Comparison of `pySYD` and `SYD` results for global parameters $\rm \nu_{max}$ (left) and $\Delta\nu$ 
(right) for 30 *Kepler* stars, which are colored by the time series baseline. The bottom panels show the
fractional residuals.\label{fig:comparison}](comparison.png)

# Documentation & Examples

The main documentation for the `pySYD` software is hosted at [pysyd.readthedocs.io](https://pysyd.readthedocs.io). 
`pySYD` provides a convenient setup feature that will download data for three example stars and automatically create 
the recommended files for an easy quickstart. The features of the `pySYD` output results are described in detail in 
the documentation.

# Acknowledgements

We acknowledge contributions from Dennis Stello, Jie Yu, Pavadol Yamsiri, and other users of the `SYD` pipeline.

We also acknowledge support from the Alfred P. Sloan Foundation, the National Aeronatuics and Space Administration
(80NSSC19K0597), and the National Science Foundation (AST-1717000, DGE-1842402).

# References
