********
Overview
********

When running the software,  ``pySYD`` will look in the following paths:

- ``INFDIR`` : '~/path/to/local/pysyd/directory/info'
- ``INPDIR`` : '~/path/to/local/pysyd/directory/data'
- ``OUTDIR`` : '~/path/to/local/pysyd/directory/results'

which by default, is the absolute path of the current working directory (think wherever you
ran setup from).

A ``pySYD`` pipeline ``Target`` class object has two main function calls:

#. The first module :
    * **Summary:** a crude, quick way to identify the power excess due to solar-like oscillations
    * This uses a heavy smoothing filter to divide out the background and then implements a frequency-resolved, collapsed 
      autocorrelation function (ACF) using 3 different ``box`` sizes
    * The main purpose for this first module is to provide a good starting point for the
      second module. The output from this routine provides a rough estimate for numax, which is translated 
      into a frequency range in the power spectrum that is believed to exhibit characteristics of p-mode
      oscillations
#. The second module : 
    * **Summary:** performs a more rigorous analysis to determine both the stellar background contribution
      as well as the global asteroseismic parameters.
    * Given the frequency range determined by the first module, this region is masked out to model 
      the white- and red-noise contributions present in the power spectrum. The fitting procedure will
      test a series of models and select the best-fit stellar background model based on the BIC.
    * The power spectrum is corrected by dividing out this contribution, which also saves as an output text file.
    * Now that the background has been removed, the global parameters can be more accurately estimated. Numax is
      estimated by using a smoothing filter, where the peak of the heavily smoothed, background-corrected power
      spectrum is the first and the second fits a Gaussian to this same power spectrum. The smoothed numax has 
      typically been adopted as the default numax value reported in the literature since it makes no assumptions 
      about the shape of the power excess.
    * Using the masked power spectrum in the region centered around numax, an autocorrelation is computed to determine
      the large frequency spacing.

.. note::

    By default, both modules will run and this is the recommended procedure if no other information 
    is provided. 

    If stellar parameters like the radius, effective temperature and/or surface gravity are provided in the **info/star_info.csv** file, ``pySYD`` 
    can estimate a value for numax using a scaling relation. Therefore the first module can be bypassed,
    and the second module will use the estimated numax as an initial starting point.

    There is also an option to directly provide numax in the **info/star_info.csv** (or via command line, 
    see :ref:`advanced usage<advanced>` for more details), which will override the value found in the first module. This option 
    is recommended if you think that the value found in the first module is inaccurate, or if you have a visual 
    estimate of numax from the power spectrum.



****
FAQs
****

#. I'm getting an error. What do I do?
	
	Please submit a pull request `here <https://github.com/ashleychontos/pySYD/pulls>`_. Be sure to include:
	  1. the version of ``pySYD`` used, 
	  2. the full command that was used, 
	  3. a screenshot of the error,
	  4. the times series and/or power spectrum that was used,
	  5. and any other relevant details you can provide. 
	The GitHub page for ``pySYD`` is: _`https://github.com/ashleychontos/pySYD`_.


#. There are a lot of upper and lower bound options, what do they all mean? 
	
	- ``--lb``/``--ub``: limits the frequency range of the power spectrum that is used to derive the granulation background and global parameters 
	- ``--le``/``--ue``: the folded frequency range containing mixed modes to "whiten" and thus better estimate dnu (requires an estimate for dnu as well)
	- ``--lp``/``--up``: limits the region around the power excess that is used to calculate the autocorrelation function (ACF)
	- ``--lx``/``--ux``: limits the region used to search for the power excess and estimate numax


#. What's the difference between ``--bf``, ``--sw``, ``--sm``, ``--sp`` and ``--se``?
	
	- ``--bf``: smooths the power spectrum to calculate red noise components *after* the power excess has been masked out.
	- ``--sw``: smooths the power spectrum in the first module when ``pySYD`` searches for the power excess. This value does not affect the power spectrum in the second module. The default value is 50 muHz.
	- ``--sm``: smooths the power spectrum in the second module when ``pySYD`` measures numax as the center of the Gaussian around the power excess.
	- ``--sp``: smooths the power spectrum in panel 5 that is used to calculate the ACF in panel 6. The default value is 2.5 muHz.
	- ``--se``: smooths the echelle diagram in panel 8. To use this argument, specify the smoothing box filter in muHz.

#. I'm not happy with the dnu that ``pySYD`` has measured. How can I change dnu?
	
	The best way to change the dnu measurement is to specify a value of dnu using the ``--dnu`` flag (ie. ``--dnu 42.1``). If the ACF contains many peaks around its higher amplitude peaks due to low signal-to-noise data, you can smooth the power spectrum that is used to calculate the ACF by changing the ``--sp`` flag (ie. ``--sp 3.5``). Alternatively, if the peak corresponding to the best dnu measurement is not one of the five highest peaks, you can change the number of peaks found in the ACF by specifying ``--npeaks`` (ie. ``--npeaks 10``). 
	
	If the power spectrum contains mixed modes that mask the dnu measurement, you can remove these mixed modes by specifying the lower and upper bound of where the mixed modes are found in the echelle diagram using the following flags: ``--le`` and ``--ue``. Refer to the advanced usage section below for an example on how to remove mixed modes. **This feature is coming soon!**

	Lastly, you can also use ``--xx`` flag to find the dnu manually. This uses the `Echelle <https://github.com/danhey/echelle>`_ package to faciliate dnu measurement. **This feature is coming soon!** 

#. There's an artefact in my data preventing an accurate numax and/or dnu measurement. What can I do?
	
	If you're using *Kepler* data, you can use the ``--kc`` flag to remove the artefact. If the artefact is preventing an accurate numax estimate and it is *not* near the power excess, you can use the ``--lx/--ux`` flags to limit the region that is used to search for the power excess. If the artefact is near the power excess envelope and is preventing an accurate dnu measurement, you can use the ``--lp/--up`` flags to limit how much of the region around the power excess should be used to calculate the ACF.

#. What is the difference between ``--exwidth``, ``--indwidth`` and ``--threshold``?
	
	- ``--exwidth``: specifies the fractional value of power excess used to calculate ACF. The default is 1. To include more of the spectrum around the power excess, increase this value (ie. ``--exwidth 1.2``). To limit the region around the power excess, decrease this value (ie. ``--exwidth 0.7``).  
	- ``--indwith``: bins the power spectrum used for modeling the background.
	- ``--threshold``: specifies the fractional value of FWHM of the peak corresponding the measured dnu. This region is used to in MC iterations when calculating the dnu uncertainties. To include more of the peak, increase this number (ie. ``--threshold 1.5``). To limit how much of the peak is used, decrease this number (ie. ``--threshold 0.8``).

#. What can I do to change how the echelle plot looks?
	
	To smooth the echelle diagram via interpolation, use the ``--ie`` flag. You can also smooth the echelle diagram by specifying a box filter in muHz using the ``--se`` flag (ie. ``--se 5``). Another option is to change the width (dnu modulus) and height (frequency) of the axes by specifying ``--xe`` and ``--ye``. Lastly, there's also an option to change the clip value with ``--ce``. 

******
Vision
******

There has been a growing interest from the broader astronomy community, 
recognizing the utility in the application of asteroseismology. 

We recognized the very straightforward solution to this problem -- take one of
the closed-source pipelines that is benchmarked to *Kepler* legacy results and
translate it to an open-source language, thus killing two birds with one stone.
We also saw it as an *opportunity* to establish the much-needed connection
with non-expert astronomers that recognize the utility of asteroseismology.

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


*************
Related Tools
*************

``pySYD`` provides general purpose tools for performing asteroseismic analysis in the frequency domain.
Several tools have been developed to solve related scientific and data analysis problems. On this page we
list software that may complement ``pySYD``.

Packages
########

We have compiled a list of software packages that performs similar or complementary analyses.

* ``A2Z``: determining global parameters of the oscillations of solar-like stars
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2010A%26A...511A..46M>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``Background``: an extension of ``DIAMONDS`` that fits the background signal of solar-like oscillators 
   - language: `c++11`
   - reference: no
   - documentation: no
   - publicly available: yes <https://github.com/EnricoCorsaro/Background>
   - requires license: no

* ``CAN``: on the detection of Lorentzian profiles in a power spectrum
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2009A%26A...506.1043G>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``COR``: on detecting the large separation in the autocorrelation of stellar oscillation times series
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2009A%26A...508..877M>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``DIAMONDS``: high-DImensional And multi-MOdal NesteD Sampling
   - language: `c++11`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..71C>
   - documentation: yes <https://diamonds.readthedocs.io/en/latest/>
   - publicly available: yes <https://github.com/EnricoCorsaro/DIAMONDS>
   - requires license: n/a

* ``DLB``:
   - language: ``?``
   - reference: no
   - documentation: n/a
   - publicly available: no
   - requires license: n/a 

* ``FAMED``: Fast & AutoMated pEakbagging with Diamonds
   - language: `IDL` (currently being developed in `Python`)
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2020A%26A...640A.130C>
   - documentation: yes <https://famed.readthedocs.io/en/latest/>
   - publicly available: yes <https://github.com/EnricoCorsaro/FAMED>
   - requires license: yes

* Flicker Flipper?: 
   - language:
   - reference:
   - documentation: 
   - publicly available: 
   - requires license: n/a

* ``KAB``: automated asteroseismic analysis of solar-type stars
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2010arXiv1003.4167K>
   - documentation: no
   - publicly available: no
   - requires license: n/a
  
* ``lightkurve``: a friendly Python package for making discoveries with *Kepler* & TESS
   - language: `Python`
   - reference: no
   - documentation: yes <https://docs.lightkurve.org>
   - publicly available: yes <https://github.com/lightkurve/lightkurve>
   - requires license: no 

* ``OCT``: automated pipeline for extracting oscillation parameters of solar-like main-sequence stars
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2010MNRAS.402.2049H>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``ORK``: using the comb response function method to identify spacings
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2008ApJ...676.1248B>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``QML``: a power-spectrum autocorrelation technique to detect global asteroseismic parameters
   - language: `?`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2011arXiv1104.0631V>
   - documentation: no
   - publicly available: no
   - requires license: n/a

* ``PBjam``: a python package for automating asteroseismology of solar-like oscillators
   - language: `Python`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2021AJ....161...62N>
   - documentation: yes <https://pbjam.readthedocs.io/en/latest/>
   - publicly available: yes <https://github.com/grd349/PBjam>
   - requires license: no 

* ``SYD``: automated extraction of oscillation parameters for *Kepler* observations of solar-type stars
   - language: `IDL`
   - reference: yes <https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H>
   - documentation: no
   - publicly available: no
   - requires license: yes


.. important:: 

    If your software is not listed or if something listed is incorrect/missing, please 
    open a pull request to add it, we aim to be inclusive of all *Kepler*-, K2- and TESS-
    related tools!
