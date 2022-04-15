.. role:: bash(code)
   :language: bash

.. _user-guide-cli-examples-advanced:

**************
Advanced usage
**************


Below are examples of different commands, including their before and after plots to demonstrate
the desired effects.

-----

:term:`--ew<--ew, --exwidth>` & :term:`--exwidth<--ew, --exwidth>`
##################################################################

Fractional amount to scale the width of the oscillations envelope by -- which is normally calculated
w.r.t. solar values.

+-------------------------------------------------------+-------------------------------------------------------+
| Before                                                | After                                                 |
+=======================================================+=======================================================+
| :bash:`pysyd run --star 9542776 --numax 900`          | :bash:`pysyd run --star 9542776 --numax 900 --ew 1.5` |
+-------------------------------------------------------+-------------------------------------------------------+
| .. figure:: ../_static/examples/9542776_before.png    | .. figure:: ../_static/examples/9542776_after.png     |
|    :width: 680                                        |    :width: 680                                        |
+-------------------------------------------------------+-------------------------------------------------------+

-----

:term:`-k<-k, --kc, --kepcorr>`, :term:`--kc<-k, --kc, --kepcorr>` & :term:`--kepcorr<-k, --kc, --kepcorr>`
###########################################################################################################

Remove the well-known *Kepler* short-cadence artefact that occurs at/near the long-cadence :term:`nyquist frequency` 
(:math:`\sim 270 \mu \mathrm{Hz}`) by simulating white noise

+-------------------------------------------------------+------------------------------------------------------+
| Before                                                | After                                                |
+=======================================================+======================================================+
| :bash:`pysyd run --star 8045442 --numax 550`          | :bash:`pysyd run --star 8045442 --numax 550 --kc`    |
+-------------------------------------------------------+------------------------------------------------------+
| .. figure:: ../_static/examples/8045442_before.png    | .. figure:: ../_static/examples/8045442_after.png    |
|    :width: 680                                        |    :width: 680                                       |
+-------------------------------------------------------+------------------------------------------------------+

-----

:term:`--lp<--lp, --lowerp>` & :term:`--lowerp<--lp, --lowerp>`
###############################################################

Manually set the lower frequency bound (or limit) of the power excess, which is helpful
in the following scenarios:

 #. the width of the power excess is wildly different from that estimated by the solar scaling relation
 #. artefact or strange (typically not astrophysical) feature is close to the power excess and cannot be removed otherwise
 #. power excess is near the :term:`nyquist frequency`


+---------------------------------------------------------+--------------------------------------------------------+
| Before                                                  | After                                                  |
+=========================================================+========================================================+
| :bash:`pysyd run --star 10731424 --numax 750`           | :bash:`pysyd run --star 10731424 --numax 750 --lp 490` |
+---------------------------------------------------------+--------------------------------------------------------+
| .. figure:: ../_static/examples/10731424_before.png     | .. figure:: ../_static/examples/10731424_after.png     |
|    :width: 680                                          |    :width: 680                                         |
+---------------------------------------------------------+--------------------------------------------------------+

-----

:term:`--npeaks<--peaks, --npeaks>` & :term:`--peaks<--peaks, --npeaks>`
########################################################################

Change the number of peaks chosen in the autocorrelation function (:term:`ACF`) - this is especially
helpful for low S/N cases, where the spectrum is noisy and the ACF has many peaks close the expected
spacing (**FIX THIS**)

+-------------------------------------------------------+------------------------------------------------------+
| Before                                                | After                                                |
+=======================================================+======================================================+
| :bash:`pysyd run --star 9455860`                      | :bash:`pysyd run --star 9455860 --npeaks 10`         |
+-------------------------------------------------------+------------------------------------------------------+
| .. figure:: ../_static/examples/9455860_before.png    | .. figure:: ../_static/examples/9455860_after.png    |
|    :width: 680                                        |    :width: 680                                       |
+-------------------------------------------------------+------------------------------------------------------+

-----

:term:`--numax<--numax>`
########################

If the value of :math:`\rm \nu_{max}` is known, this can be provided to bypass the first module and save some time. 
There are also other ways to go about doing this, please see our notebook tutorial that goes through these different
ways.

+--------------------------------------------------------+-------------------------------------------------------+
| Before                                                 | After                                                 |
+========================================================+=======================================================+
| :bash:`pysyd run --star 5791521`                       | :bash:`pysyd run --star 5791521 --numax 670`          |
+--------------------------------------------------------+-------------------------------------------------------+
| .. figure:: ../_static/examples/5791521_before.png     | .. figure:: ../_static/examples/5791521_after.png     |
|    :width: 680                                         |    :width: 680                                        |
+--------------------------------------------------------+-------------------------------------------------------+

-----

:term:`--ux<--ux, --upperx>` & :term:`--upperx<--ux, --upperx>`
###############################################################

Set the upper frequency limit in the power spectrum when estimating :math:`\rm \nu_{max}` before the main fitting
routine. This is helpful if there are high frequency artefacts that the software latches on to.

+--------------------------------------------------------+-------------------------------------------------------+
| Before                                                 | After                                                 |
+========================================================+=======================================================+
| :bash:`pysyd run --star 11769801`                      | :bash:`pysyd run --star 11769801 --ux 3500`           |
+--------------------------------------------------------+-------------------------------------------------------+
| .. figure:: ../_static/examples/11769801_before.png    | .. figure:: ../_static/examples/11769801_after.png    |
|    :width: 680                                         |    :width: 680                                        |
+--------------------------------------------------------+-------------------------------------------------------+

-----

:term:`-i<-i, --ie, --interpech>`, :term:`--ie<-i, --ie, --interpech>` & :term:`--interpech<-i, --ie, --interpech>`
###################################################################################################################

Smooth the echelle diagram output by turning on the (bilinear) interpolation, which is helpful for identifying
ridges in low S/N cases

+--------------------------------------------------------+--------------------------------------------------------+
| Before                                                 | After                                                  |
+========================================================+========================================================+
| :bash:`pysyd run 3112889 --numax 871.52`               | :bash:`pysyd run --star 3112889 --numax 871.52 --ie`   |
+--------------------------------------------------------+--------------------------------------------------------+
| .. figure:: ../_static/examples/3112889_before.png     | .. figure:: ../_static/examples/3112889_after.png      |
|    :width: 680                                         |    :width: 680                                         |
+--------------------------------------------------------+--------------------------------------------------------+

-----