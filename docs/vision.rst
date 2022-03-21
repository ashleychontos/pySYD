*******************************
Vision of the ``pySYD`` project
*******************************

The NASA space telescopes *Kepler*, K2 and TESS have recently
provided very large databases of high-precision light curves of stars.
By detecting brightness variations due to stellar oscillations, these
light curves allow the application of asteroseismology to large numbers
of stars, which requires automated software tools to efficiently extract
observables. The growing number of detections from these missions has 
been associated with a growing interest from the community that could
easily see the utility in the application of asteroseismology.
 led to a growing interest and demand from the community
there was a growing 

Several tools have been developed for asteroseismic analyses, but many of 
them are closed-source and therefore inaccessible to the general astronomy 
community. Some open-source tools exist, but they are either optimized for 
smaller samples of stars or have not yet been extensively tested against 
closed-source tools. 

We recognized the very straightforward solution to this problem -- take one
of the closed-source pipelines that is benchmarked to *Kepler* legacy 
results and translate it to an open-source language, thus killing two 
birds with one stone. We also saw this as an *opportunity* to establish a 
much-needed relation or connection to non-experts that recognized the 
utility of asteroseismology. 


Therefore the initial vision of this project was intended to be a direct 
translation of the IDL-based ``SYD`` pipeline, which has been extensively 
used to measure asteroseismic parameters for many *Kepler* stars and tested 
against other closed-source pipelines. While many of the resident experts
are still clinging to their IDL, there was a gap growing between experts 
and new incoming students, the latter who typically possess some basic
`Python` knowledge. for mentoring new or younger
students -- most of them coming in with some basic `Python` knowledge.
This was actually the best thing that could've happened for us because it
was basically like having our own beta testers, which has ultimately 
helped make pySYD even better than it already was!


.. note::

    We've attempted to collect these tools in a :ref:`single place<related>` 
    for easy comparisons. Please let us know if we've somehow missed yours --
    we would be happy to add it!

 
Goals
#####

Therefore the initial vision of this project was intended to be a direct translation of 
the IDL-based ``SYD`` pipeline, which has been extensively used to measure 
asteroseismic parameters for many *Kepler* stars and tested against other
closed-source pipelines.

``pySYD`` is a `Python`-based implementation of the `IDL`-based ``SYD`` pipeline 
`(Huber et al. 2009) <https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract>`_, 
which was extensively used to measure asteroseismic parameters for Kepler stars. 
Papers based on asteroseismic parameters measured using the ``SYD`` pipeline include 
`Huber et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract>`_, 
`Chaplin et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014ApJS..210....1C/abstract>`_, 
`Serenelli et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract>`_ 
and `Yu et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJS..236...42Y/abstract>`_.

``pySYD`` adapts the well-tested methodology from ``SYD`` while also improving these 
existing analyses and expanding upon numerous new (optional) features. Improvements include:

- Automated best-fit background model selection
- Parallel processing
- Easily accessible + command-line friendly interface
- Ability to save samples for further analyses