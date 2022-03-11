.. |br| raw:: html

   <br />

.. module:: pysyd

.. image:: _static/latex.png

Asteroseismology, or the study of stellar oscillations, is a powerful tool
for studying the internal structure of stars and determining their fundamental 
properties [aertz2021]_. Specifically for stars similar to the Sun, turbulent near-surface 
convection excites sound waves that propagate within the stellar cavity, and hence
provides powerful constraints on stellar interiors that are inaccessible by 
any other means. While it is well-established and widely-accepted as the gold 
standard for characterizing fundamental stellar properties (e.g., masses, radii,
ages, etc.), asteroseismology is a niche field and still mostly a closed-door science. 

In an effort to make asteroseismology more accessible to the broader astronomy
community, ``pySYD`` was developed as a Python package to automatically detect
solar-like oscillations and measure their global properties. Therefore for non-expert
users, ``pySYD`` is designed to be an end-to-end product -- serving as a hands-off tool
that does not require substiantial background knowledge in asteroseismology but still
reap the benefits! Even for the expert users, ``pySYD`` has many new features and 
improvements to customize your asteroseismology experience.

This package is being actively developed in 
`a public repository on GitHub <https://github.com/ashleychontos/pySYD>`_ -- we especially 
welcome and *encourage* any new contributions to help make ``pySYD`` better! Please see 
our :ref:`community guidelines <guidelines/index>` to find out how you can help. No contribution 
is too small!

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: About

   Installation <installation>
   Quickstart <getting_started>
   Citing ``pySYD`` <attribution>
   Contributing <guidelines>
   Other Software <other>
   glossary
   
.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Advanced
   
   Tutorials <tutorials/index>
   User Guide <guide/index>
   



Bibliography
############

.. bibliography::
    :cited:
