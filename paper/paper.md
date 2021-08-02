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

Asteroseismology, the study of stellar oscillations, is a powerful tool
for studying the interiors of stars and determining their fundamental
properties [@aerts2021]. For stars with temperatures that are similar
to the Sun, turbulent near-surface convection excites sound waves that
propagate within the stellar cavity [@bedding2014]. These waves
penetrate into different depths within the star and therefore provide
powerful constraints on stellar interiors that would otherwise be
inaccessible. Asteroseismology is well-established in astronomy as the
gold standard for characterizing fundamental properties like masses,
radii, densities, and ages for single stars, which has broad impacts on
several fields in astronomy. For example, ages of stars are important to
reconstruct the formation history of the Milky Way (so-called galactic
archaeology). For exoplanets that are discovered indirectly through
changes in stellar observables, precise and accurate stellar masses and
radii are critical for learning about the planets that orbit them.

# Statement of Need

The NASA space telescopes *Kepler*, K2 and TESS have recently
provided very large databases of high-precision light curves of stars.
By detecting brightness variations due to stellar oscillations, these
light curves allow the application of asteroseismology to large numbers
of stars, which requires automated software tools to efficiently extract
observables. Several tools have been developed for asteroseismic
analyses [e.g., `A2Z`, @mathur2010; `COR`, @mosser2009; `OCT`, @hekker2010; 
`SYD`, @huber2009], but many of them are closed-source and therefore 
inaccessible to the general astronomy community. Some open-source tools 
exist [e.g., `DIAMONDS` and `FAMED`, @corsaro2014; `PBjam`, @nielsen2021; 
`lightkurve`, @lightkurve], but they are either optimized for smaller
samples of stars or have not yet been extensively tested against
closed-source tools.

`pySYD` is adapted from the framework of the IDL-based
`SYD` pipeline [@huber2009; hereafter referred to as `SYD`], 
which has been used frequently to measure asteroseismic parameters 
for *Kepler* stars and has been extensively tested against other
closed-source tools on *Kepler* data [@verner2011;@hekker2011]. 
Papers based on asteroseismic parameters measured using the `SYD`
pipeline include @huber2011, @bastien2013, @chaplin2014, @serenelli2017, 
and @yu2018. `pySYD` was developed using the same well-tested 
methodology, but has improved functionality including automated background 
model selection and parallel processing as well as improved flexibility
through a user-friendly interface, while still maintaining its speed and
efficiency. Well-documented, open-source asteroseismology software that
has been benchmarked against closed-source tools are critical to ensure
the reproducibility of legacy results from the *Kepler* mission 
[@borucki2010]. The combination of well-tested methodology, improved 
flexibility and parallel processing capabilities will make `pySYD` a
promising tool for the broader community to analyze current and
forthcoming data from the NASA TESS mission [@ricker2015].

# The `pySYD` library

The excitation mechanism for solar-like oscillations is stochastic and
modes are observed over a range of frequencies. Oscillation modes are
separated by the so-called large frequency spacing ($\Delta\nu$), with
an approximately Gaussian-shaped power excess centered on the frequency 
of maximum power ($\rm \nu_{max}$). The observables $\rm \nu_{max}$ and 
$\Delta\nu$ are directly related to fundamental properties such as 
surface gravity, density, mass and radius [@kjeldsen1995].

`pySYD` is a Python package for detecting solar-like oscillations
and measuring global asteroseismic parameters. Derived parameters
include $\rm \nu_{max}$ and $\Delta\nu$, as well as characteristic
amplitudes and timescales of correlated red-noise signals due to stellar
granulation.

A `pySYD` pipeline `Target` class object has two main methods:

- The first module searches for signatures of solar-like oscillations by 
  implementing a frequency-resolved, collapsed autocorrelation (ACF) method. 
  The output from this routine provides an estimate for $\rm \nu_{max}$,
  which is used as an initial guess for the main pipeline function (i.e. second 
  module). However if $\rm \nu_{max}$ is already known, the user can provide 
  an estimate and hence bypass this module.
- The second routine begins by masking out the region in the power spectrum (PS)
  with the power excess in order to characterize the stellar background. Next, 
  `pySYD` optimizes and selects the stellar background model that minimizes the 
  Bayesian Information Criterion [BIC; @schwarz1978]. The best-fit background 
  model is then subtracted from the PS, where the peak of the smoothed, 
  background-corrected PS is $\rm \nu_{max}$. An ACF is computed 
  using the region in the power spectrum centered on $\rm \nu_{max}$ and the 
  peak in the ACF closest to the expected value for the large frequency 
  separation is $\Delta\nu$.

The `pySYD` software depends on a number of powerful libraries, including 
Astropy [@astropy1;@astropy2], Matplotlib [@matplotlib], Numpy [@numpy], and 
SciPy [@scipy]. `pySYD` has been tested against `SYD` using results from the 
*Kepler* sample for $\sim100$ stars (\autoref{fig:comparison}). The comparisons 
show no significant systematic differences, with a median offset and scatter 
of $0.2\%$ and $0.4\%$ for $\rm \nu_{max}$ as well as $0.002\%$ and $0.09\%$
for $\Delta\nu$, which is smaller or comparable to the typical random
uncertainties [@huber2011].

![Comparison of global parameters $\rm \nu_{max}$ (left) and $\Delta\nu$ (right) 
measured by `pySYD` and IDL `SYD` for $\sim100$ *Kepler* stars. The bottom panels 
show the fractional residuals.\label{fig:comparison}](comparison.png)

# Documentation & Examples

The main documentation for the `pySYD` software is hosted at [pysyd.readthedocs.io](https://pysyd.readthedocs.io).
`pySYD` provides a convenient setup feature that will download data for three example stars and automatically 
create the recommended data structure for an easy quickstart. The features of the `pySYD` output results are 
described in detail in the documentation.

# Acknowledgements

We thank Dennis Stello, Jie Yu, Marc Hon, and other `pySYD` users 
for discussion and suggestions which helped with the development of this code.

We also acknowledge support from: 
- The National Science Foundation (DGE-1842402, AST-1717000)
- The National Aeronautics and Space Administration (80NSSC19K0597)
- The Alfred P. Sloan Foundation

# References
