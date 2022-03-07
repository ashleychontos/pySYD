.. |br| raw:: html

   <br />

pySYD: |br| Automated Measurements |br| of Global Asteroseismic Parameters
==========================================================================

``pySYD`` is an open-source python package to detect solar-like oscillations and measure global asteroseismic parameters. ``pySYD`` provides best-fit values and uncertainties for the following parameters:

- Granulation background, including characteristic time scales and amplitudes
- Frequency of maximum power
- Large frequency separation
- Mean oscillation amplitudes

For a basic introduction to these parameters and asteroseismic data analysis of 
solar-like oscillators see  `Bedding et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014aste.book...60B/abstract>`_.

Please cite our recent JOSS paper `Chontos+2021 <https://arxiv.org/abs/2108.00582>`_ if you 
make use of ``pySYD`` in your work. The recommended BibTeX entry for this citation is::

    {@ARTICLE{2021arXiv210800582C,
           author = {{Chontos}, Ashley and {Huber}, Daniel and {Sayeed}, Maryum and {Yamsiri}, Pavadol},
            title = "{$\texttt{pySYD}$: Automated measurements of global asteroseismic parameters}",
          journal = {arXiv e-prints},
         keywords = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
             year = 2021,
            month = aug,
              eid = {arXiv:2108.00582},
            pages = {arXiv:2108.00582},
    archivePrefix = {arXiv},
           eprint = {2108.00582},
     primaryClass = {astro-ph.SR}, 
           adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210800582C},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


Code contributions are welcome and should be submitted as a pull request.

Bug reports and feature requests should be posted to the `GitHub issue tracker <https://github.com/ashleychontos/pySYD/issues>`_.

.. toctree::
   :maxdepth: 1
   :hidden: 
   :caption: Getting Started

   installation
   overview
   examples
   performance
   
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced Usage
   
   
   cli
   faq
   advanced
   api
   

Indices and tables for the source code
****************************************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
