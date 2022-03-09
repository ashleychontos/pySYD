.. |br| raw:: html

   <br />

.. module:: pysyd

**************************************************************************
pySYD: |br| Automated Measurements |br| of Global Asteroseismic Parameters
**************************************************************************

Asteroseismology, or the study of stellar oscillations, is a powerful tool
for studying the internal structure of stars and determining their fundamental 
properties :cite:p:`aertz2021`. Specifically for stars similar to the Sun, turbulent near-surface 
convection excites sound waves that propagate within the stellar cavity, and hence
providing powerful constraints on stellar interiors that are inaccessible by 
any other means. While it is well-established and widely-accepted as the gold 
standard for characterizing fundamental stellar properties (e.g., masses, radii,
ages, etc.), asteroseismology is still mostly a niche field and closed-door science. 

In an effort to make asteroseismology more accessible to the broader astronomy
community, ``pySYD`` was established as a Python package to automatically detect
solar-like oscillations and measure their global properties. Therefore, the outputs 
from ``pySYD`` can be used to determine precise and accurate stellar properties 
without the need for substantial background knowledge in asteroseismology. In other words, 
``pySYD`` has been extensively tested and developed to be as hands-off as possible.

This package is being actively developed in 
`a public repository on GitHub <https://github.com/ashleychontos/pySYD>`_ -- we especially 
welcome and *encourage* any new contributions to help make ``pySYD`` better! Please see 
our :ref:`community guidelines <guidelines/index>` to find out how you can help. No contribution 
is too small!


Contributors
############

Our list of contributors continues to grow!

.. include:: CONTRIBUTORS.rst


Bibliography
############

.. bibliography::
    :cited:

-----

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:
   :caption: Introduction

   installation
   getting_started
   glossary
   attribution
   
   
.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:
   :caption: Advanced
   
   questions
   advanced
   guide/index
   guidelines
   
