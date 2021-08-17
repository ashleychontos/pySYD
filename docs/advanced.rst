.. _advanced:
.. role:: bash(code)
   :language: bash

FAQs & Advanced Usage
########################

FAQs
*******

#. I'm getting an error. What do I do?
	
	Please submit a pull request `here <https://github.com/ashleychontos/pySYD/pulls>`_. Be sure to include the version of ``pySYD`` used, the command you ran, screenshot of the error and any other details you can provide. The GitHub page for ``pySYD`` can be found here: `https://github.com/ashleychontos/pySYD`.

#. What is the difference between ``--lx/--ux``, ``--lb/--ub`` and ``--lp/--up``?
	
	- ``--lx/--ux``: limits the region used to search for the power excess.
	- ``--lb/--ub``: limits the region used to model the background. 
	- ``--lp/--up``: limits the region around the power excess used to calculate the ACF.

#. What's the difference between ``--bf``, ``--sw``, ``--sm``, ``--sp`` and ``--se``?
	
	- ``--bf``: smooths the power spectrum to calculate red noise components *after* the power excess has been masked out.
	- ``--sw``: smooths the power spectrum in the first module when ``pySYD`` searches for the power excess. This value does not affect the power spectrum in the second module. The default value is 50 muHz.
	- ``--sm``: smooths the power spectrum in the second module when ``pySYD`` measures numax as the center of the Gaussian around the power excess.
	- ``--sp``: smooths the power spectrum in panel 5 that is used to calculate the ACF in panel 6. The default value is 2.5 muHz.
	- ``--se``: smooths the echelle diagram in panel 8. To use this argument, specify the smoothing box filter in muHz.

#. I'm not happy with ``pySYD``'s measured dnu. How can I change the dnu value?
	
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

Advanced Usage
*****************

Below are examples of how to use specific ``pySYD`` features, as well as plots showing results before and after their usage.


``--dnu: force dnu``
++++++++

+-------------------------------------------------+---------------------------------------------------------+
| Before                                          | After                                                   |
+=================================================+=========================================================+
| Fix the dnu value if the desired dnu is not automatically selected by `pySYD`.                            |
+-------------------------------------------------+---------------------------------------------------------+
|:bash:`pysyd run --star 9512063 --numax 843`     |:bash:`pysyd run --star 9512063 --numax 843 --dnu 49.54` |
+-------------------------------------------------+---------------------------------------------------------+
| .. figure:: figures/advanced/9512063_before.png | .. figure:: figures/advanced/9512063_after.png          |
|    :width: 600                                  |    :width: 600                                          |
+-------------------------------------------------+---------------------------------------------------------+


``--ew: excess width``
++++++++

+------------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                           | After                                                            |
+==================================================================+==================================================================+
| Changed the excess width in the background corrected power spectrum used to calculate the ACF (and hence dnu).                      |
+------------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run --star 9542776 --numax 900`                     | :bash:`pysyd run --star 9542776 --numax 900 --ew 1.5`            |
+------------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: figures/advanced/9542776_before.png                  | .. figure:: figures/advanced/9542776_after.png                   |
|    :width: 600                                                   |    :width: 600                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+


``--ie: smooth echelle``
++++++++

+------------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                           | After                                                            |
+==================================================================+==================================================================+
| Smooth echelle diagram by turning on the interpolation, in order to distinguish the modes for low SNR cases.                        |
+------------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run 3112889 --numax 871.52 --dnu 53.2`              | :bash:`pysyd run --star 3112889 --numax 871.52 --dnu 53.2 --ie`  |
+------------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: figures/advanced/3112889_before.png                  | .. figure:: figures/advanced/3112889_after.png                   |
|    :width: 600                                                   |    :width: 600                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+


``--kc: Kepler correction``
++++++++

+------------------------------------------------------------------+------------------------------------------------------------------+
| Before                                                           | After                                                            |
+==================================================================+==================================================================+
| Remove *Kepler* artefacts from the power spectrum for an accurate numax estimate.                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+
| :bash:`pysyd run --star 8045442 --numax 550`                     | :bash:`pysyd run --star 8045442 --numax 550 --kc`                |
+------------------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: figures/advanced/8045442_before.png                  | .. figure:: figures/advanced/8045442_after.png                   |
|    :width: 600                                                   |    :width: 600                                                   |
+------------------------------------------------------------------+------------------------------------------------------------------+


``--lp: lower frequency of power excess``
++++++++

+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Before                                                                   | After                                                                    |
+==========================================================================+==========================================================================+
| Set the lower frequency limit in zoomed in power spectrum; useful when an artefact is present close to the excess and cannot be removed otherwise.  |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| :bash:`pysyd run --star 10731424 --numax 750`                            | :bash:`pysyd run --star 10731424 --numax 750 --lp 490`                   |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| .. figure:: figures/advanced/10731424_before.png                         | .. figure:: figures/advanced/10731424_after.png                          |
|    :width: 600                                                           |    :width: 600                                                           |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+


``--npeaks: number of peaks``
++++++++

+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Before                                                                   | After                                                                    |
+==========================================================================+==========================================================================+
| Change the number of peaks chosen in ACF; useful in low SNR cases where the spectrum is noisy and ACF has many peaks close to the expected dnu.     |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| :bash:`pysyd run --star 9455860`                                         | :bash:`pysyd run --star 9455860 --npeaks 10`                             |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+
| .. figure:: figures/advanced/9455860_before.png                          | .. figure:: figures/advanced/9455860_after.png                           |
|    :width: 600                                                           |    :width: 600                                                           |
+--------------------------------------------------------------------------+--------------------------------------------------------------------------+


``--numax``
++++++++

+-------------------------------------------------------+-------------------------------------------------------+
| Before                                                | After                                                 |
+=======================================================+=======================================================+
| Set the numax value if pySYD chooses the wrong excess in the power spectrum.                                  |
+-------------------------------------------------------+-------------------------------------------------------+
| :bash:`pysyd run --star 5791521`                      | :bash:`pysyd run --star 5791521 --numax 670`          |
+-------------------------------------------------------+-------------------------------------------------------+
| .. figure:: figures/advanced/5791521_before.png       | .. figure:: figures/advanced/5791521_after.png        |
|    :width: 600                                        |    :width: 600                                        |
+-------------------------------------------------------+-------------------------------------------------------+


``--ux: upper frequency of PS used in the first module``
++++++++

+--------------------------------------------------+-------------------------------------------------------+
| Before                                           | After                                                 |
+==================================================+=======================================================+
| Set the upper frequency limit in power spectrum; useful when `pySYD` latches on to an artefact.          |
+--------------------------------------------------+-------------------------------------------------------+
| :bash:`pysyd run --star 11769801`                | :bash:`pysyd run --star 11769801 -ux 3500`            |
+--------------------------------------------------+-------------------------------------------------------+
| .. figure:: figures/advanced/11769801_before.png | .. figure:: figures/advanced/11769801_after.png       |
|    :width: 600                                   |    :width: 600                                        |
+--------------------------------------------------+-------------------------------------------------------+


