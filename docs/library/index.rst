.. module:: pysyd

*********************
The ``pySYD`` library
*********************

We are so excited that you are ready to take a deeper dive into asteroseismology.

Overview
########

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

.. toctree::
   :titlesonly:
   :maxdepth: 2
   :caption: How it works

   Inputs <input>
   Main pipeline driver <pipeline>
   Utility functions <utils>
   Target stars <target>
   Frequency distributions <models>	      
   Plotting routines <plots>
   Outputs <output>
