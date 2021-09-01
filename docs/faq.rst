.. _faq:

############
FAQs
############



#. I'm getting an error. What do I do?
	
	Please submit a pull request `here <https://github.com/ashleychontos/pySYD/pulls>`_. Be sure to include:
	  1. the version of ``pySYD`` used, 
	  2. the full command that was used, 
	  3. a screenshot of the error,
	  4. the times series and/or power spectrum that was used,
	  5. and any other relevant details you can provide. 
	The GitHub page for ``pySYD`` is: _`https://github.com/ashleychontos/pySYD`_.


#. There are a lot of upper and lower bound options, what do they all mean? 
	
	- ``--lb``/``--ub``: frequency range limits the region used to derive the global parameters and granulation background
	- ``--le``/``--ue``: limits the region used to whiten mixed modes (requires an estimate for dnu as well)
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
