.. role::  raw-html(raw)
    :format: html

***************
Getting started
***************

The examples on this page assume that the user already has some basic-level knowledge or
experience with `Python`. If not, we recommend visiting the Python website and going through
some of `their tutorials <https://docs.python.org/3/tutorial/>`_ first before attempting 
ours!

If you have *any* questions, check out our :ref:`user guide <usage/index>` for more 
information. If this still does not address your question or problem, please do not hesitate
to contact `Ashley <achontos@hawaii.edu>`_ directly.

-----

TL;DR
#####

If you (understandably) do not have time to go through the entire user guide, we have summarized 
a couple of important tidbits. 

 - The first is that the userbase for the initial `pySYD` release was intended for non-expert 
   astronomers. **With this in mind, the software was originally developed to be as hands-off as
   possible -- as a *strictly* command-line end-to-end tool.** However since then, the software has 
   become more modular in recent updates, thus enabling broader capabilities that can be used across 
   other applications (e.g., Jupyter notebooks). 
 - In addition to being a command-line tool, the software is optimized for running many stars. 
   This means that many of the options that one would typically use or prefer, such as printing 
   output information and displaying figures, is `False` by default. For our purposes 
   here though, we will invoke them to better understand how the software operates. 

.. warning::

    It is **critical** that the input data are in the proper units in order for ``pySYD`` 
    to work properly and provide reliable results. If you are unsure about any of the units, 
    we recommend that you provide the light curve (in days) and then let us compute the power
    spectrum for you! 

-----

crashteroseismology
###################
:raw-html:`&rightarrow;` **crash course into asteroseismology**

We will go through two examples -- each demonstrating a different usage scenario. We will 
start with the command-line example to break everything down and then put it all together 
in a condensed version for the other application.

For purposes of this first example, we will assume that we do not know anything about the star or
its properties. I say this because typically we can provide optional inputs (e.g., the center
of the frequency range with the oscillations, or :term:`numax` :math:`\rm \nu_{max}`) 
that can save time and bypass some of the extra steps but we won't do that here so the software 
will run from start to finish on its own.

-----

When running ``pySYD`` from command line, you will like use something similar to the following 
statement: 

.. code-block::

    pysyd run --star 1435467 -dv --ux 5000 --mc 200

**Important: when running** `pysyd` **as a script, there is one positional argument for the pipeline "mode".** 

Breaking down the arguments
+++++++++++++++++++++++++++

Now let's deconstruct the above statement!

``pysyd``
   if you used `pip` install, the binary (or executable) should be available. In fact, the setup
   file defines this entry point for ``pysyd`` and is accessed through the :mod:`pysyd.cli.main` 
   script -- which is also where you can find the parser with all the available commands and options.

``run`` 
   regardless of how you choose to use the software, the most common way you will likely implement
   the ``pySYD`` pipeline is in run mode -- which, just as it sounds, will process stars in the order 
   they were provided. This is saved to the argument ``NameSpace`` as the ``mode`` which will run
   the pipeline by calling :mod:`pysyd.pipeline.run`. There are currently five available 
   modes, all which are described in more detail :ref:`here <library/pipeline>`

``--star 1435467``
   here we are running a single star, KIC 1435467. You can also provide multiple targets,
   where the stars will append to a list and then be processed consecutively. On the other 
   hand if no targets are provided, the program would default to reading in the star or 'todo' 
   list (via 'info/todo.txt'). Again, this is because the software is optimized for 
   running many stars.

``-dv``
   adapting Linux-like features, we reserved the single hash options for booleans which
   can all be grouped together, as shown above. The ``-d`` and ``-v`` are short for display and verbose, 
   respectively, and show the figures and verbose output. For a full list of options available, please 
   see our :ref:`command-line glossary <usage/cli/glossary>`. There are dozens of options to make your 
   experience as customized as you'd like!

``--ux 5000``
   this is an upper frequency limit for the first module that identifies the power eXcess 
   due to solar-like oscillations. In this case, there are high frequency artefacts that we would 
   like to ignore. *We actually made a special notebook tutorial specifically on how to address
   and fix this problem.* If you'd like to learn more about this or are having a similar issue, 
   please visit :ref:`this page <usage/nb/estimatenumax.ipynb>`.

``--mc 200``
   last but certainly not least - the ``mc`` (for Monte Carlo-like) option sets the number 
   of iterations the pipeline will run for. In this case, the pipeline will run for 200 steps, 
   which allows us to bootstrap uncertainties on our derived properties. 

**Note:** For a *complete* list of options which are currently available via command-line interface (CLI), 
see our special CLI :ref:`glossary<usage/cli/glossary>`.

-----

How it works
++++++++++++

So in case you haven't already, execute the command. You will immediately notice that the printed
output is actually quite long but not to worry though, as we will break it down by four main 
sections (corresponding to the approximate pipeline workflow):
 #. :ref:`Loads in parameters and data <stepone>`
 #. :ref:`Gets initial values <steptwo>`
 #. :ref:`Fits global parameters <stepthree>`
 #. :ref:`Estimates uncertainties <stepfour>`

For each step, we will first show the relevant block of printed (or verbose) output, then
describe what the software is actually doing and if applicable, conclude with the section-
specific results (i.e. files, figures, etc.).

-----

.. _stepone:

Load in parameters and data
+++++++++++++++++++++++++++

.. code-block::

    -----------------------------------------------------------
    Target: 1435467
    -----------------------------------------------------------
    # LIGHT CURVE: 37919 lines of data read
    # Time series cadence: 59 seconds
    # POWER SPECTRUM: 99518 lines of data read
    # PS oversampled by a factor of 5
    # PS resolution: 0.426868 muHz
    -----------------------------------------------------------

During this step, it will take the star name along with the command-line arguments and 
created an instance of the :mod:`pysyd.target.Target` object. Initialization of this class
will automatically search for and load in data for a given star, as shown in the output above.

For this target, KIC 1435467, both the light curve and power spectrum were available and it automatically
calculated the oversampling factor. **Note:** it will process the pipeline on oversampled spectra for 
single iterations but will *always* switch to critically-sampled spectra for estimating uncertainties. 
**Calculating uncertainties with oversampled spectra can produce unreliable results and uncertainties!**

*If there are issues during the first step,* ``pySYD`` *will flag this and immediately halt 
any further execution of the code.* If something seems questionable during this step but 
is not fatal for executing the pipeline, it will only return some warnings. In fact, all 
:mod:`pysyd.target` class instances will have an ``ok`` attribute - literally meaning 
that the star is 'ok' to be processed. By default, the pipeline checks this attribute before 
moving on. 

Since none of this happened, we can move on to the next step.

-----

.. _steptwo:

Get initial values
++++++++++++++++++

.. code-block::

    -----------------------------------------------------------
    PS binned to 173 datapoints
    
    Numax estimates
    ---------------
    Numax estimate 1: 1416.12 +/- 86.91
    S/N: 2.18
    Numax estimate 2: 1464.42 +/- 76.62
    S/N: 4.33
    Numax estimate 3: 1438.28 +/- 97.24
    S/N: 12.38
    Selecting model 3
    -----------------------------------------------------------

The main thing we need to know before performing a global fit is a rough starting point 
for the frequency corresponding to maximum power, or :term:`numax` (:math:`\rm \nu_{max}`).
Please read the next section for more information about this.

It does this by making a very rough approximation of the stellar background by binning the 
power spectrum in both log and linear spaces (think a very HEAVY smoothing filter) and divides
this out so that we are left with very little residual slope in the power spectrum. The 'Crude
Background Fit' is shown below in the second panel by the lime green line. Then we have our
background-divided power spectrum directly to the right of this panel.

.. image:: _static/quickstart/1435467_estimates.png
  :width: 680
  :alt: Parameter estimates for KIC 1435467

Next it uses a "collapsed" autocorrelation function (ACF) technique with different bin sizes
to identify localized power excess in the power spectrum due to solar-like oscillations. By
default, this is done three times (or trials) and hence, provides three different estimates.
The bottom row in the above figure shows these three trials, highlighting the one that was 
selected -- which is based on the signal-to-noise (S/N) of the detection.

Finally, it saves this best estimate in a basic csv:


.. csv-table:: 1435467 parameter estimates
   :header: "stars", "numax", "dnu", "snr"
   :widths: 20, 20, 20, 20

   1435467, 1438.27561061044, 72.3140769912867, 12.3801364686659

-----

.. _stepthree:

Fit global parameters
+++++++++++++++++++++

.. code-block::

    -----------------------------------------------------------
    GLOBAL FIT
    -----------------------------------------------------------
    PS binned to 335 data points
    
    Background model
    ----------------
    Comparing 6 different models:
    Model 0: 0 Harvey-like component(s) + white noise fixed
     BIC = 981.74 | AIC = 2.93
    Model 1: 0 Harvey-like component(s) + white noise term
     BIC = 1009.29 | AIC = 3.00
    Model 2: 1 Harvey-like component(s) + white noise fixed
     BIC = 80.37 | AIC = 0.22
    Model 3: 1 Harvey-like component(s) + white noise term
     BIC = 90.83 | AIC = 0.24
    Model 4: 2 Harvey-like component(s) + white noise fixed
     BIC = 81.50 | AIC = 0.20
    Model 5: 2 Harvey-like component(s) + white noise term
     BIC = 94.42 | AIC = 0.22
    Based on AIC statistic: model 4
    -----------------------------------------------------------

A bulk of the heavy lifting is done in this main fitting routine, which is actually done 
in two separate steps: 1) modeling and characterizing the stellar background and 2) determining 
the global asteroseismic parameters. We do this *separately* in two steps because they have 
fairly different properties and we wouldn't want either of the estimates to be influenced by 
the other in any way. 

Ultimately the stellar background has more of a presence in the power spectrum in that it is 
observed over a wider range of frequencies compared to the solar-like oscillations. Therefore 
by attempting to identify where the oscillations are in the power spectrum, we can mask 
them out to better characterize the background.

Unlike previous versions of this software and previous versions of this software (i.e. `SYD`), 
we have now implemented an automated background model selection. After much trial and error, 
the :term:`AIC` seems to perform better for our purposes - which is now the default metric used.

We should take a sidestep to explain something happening behind the scenes here. The reason why
`SYD` was so successful is because it assumed that the estimated numax and granulation timescales
could be scaled with the Sun -- a fact that was not known at the time but greatly improved the 
ability to quickly and efficiently process stars.

Of course measuring these time scales is limited by the total duration of the time series but
in general, we can resolve 3 Harvey-like components (or laws) at best (for now anyway). We use 
all this information to guess how many we should be able to resolve and over what time scales.
In fact for a given star, we end up with

.. math::

    n_{\mathrm{models}} = 2 \cdot (n_{\mathrm{laws}}+1)

The fact of 2 is because we give the options to fix the white noise or for it to also be a free
parameter. The +1 is because we also consider the model where we are not able to resolve any.
From our perspective, the main purpose of implementing this was to try to identify null detections,
since we do not expect to see oscillations in every star we observe. Ultimately this has not
worked for this purpose yet but you have an idea, please reach out and let us know!

Model 4 was selected for our example, consisting of two Harvey-like components, each with their characteristic
time scale and amplitude. In this case, the white noise was *not* a free parameter.

.. image:: _static/quickstart/1435467_global.png
  :width: 680
  :alt: Global parameters for KIC 1435467

.. note::

   To learn more about what each panel is showing, please visit :ref:`this page<library/output>`.

If this was run in its default setting, with ``--mc`` = `1`, for a single iteration, the output
parameters would look like that below. **We urge folks to run new stars for a single step first 
(ALWAYS) before running it several iterations to make sure everything looks ok.**


.. csv-table:: 1435467 global parameters
   :header: "parameter", "value", "uncertainty"
   :widths: 20, 20, 20

   numax_smooth, 1303.82549513, --
   A_smooth, 1.6981881189944,--
   numax_gauss, 1354.18609943197, --
   A_gauss, 1.45587282712706, --
   FWHM, 284.631831313442, --
   dnu, 70.653293964844, --
   tau_1, 1069.91765124738, --
   sigma_1, 31.1026782311927, --
   tau_2, 218.303624326155, --
   sigma_2, 85.4836783903674, --

-----

.. _stepfour:

Estimate uncertainties
++++++++++++++++++++++

.. code-block::

    -----------------------------------------------------------
    Sampling routine:
    100%|███████████████████████████████████████| 200/200 [00:21<00:00,  9.23it/s]
    -----------------------------------------------------------
    Output parameters
    -----------------------------------------------------------
    numax_smooth: 1303.83 +/- 65.19 muHz
    A_smooth: 1.70 +/- 0.21 ppm^2/muHz
    numax_gauss: 1354.19 +/- 43.04 muHz
    A_gauss: 1.46 +/- 0.29 ppm^2/muHz
    FWHM: 284.63 +/- 64.57 muHz
    dnu: 70.65 +/- 0.81 muHz
    tau_1: 1069.92 +/- 2121.15 s
    sigma_1: 31.10 +/- 42.95 ppm
    tau_2: 218.30 +/- 20.25 s
    sigma_2: 85.48 +/- 3.68 ppm
    -----------------------------------------------------------
     - displaying figures
     - press RETURN to exit
     - combining results into single csv file
    -----------------------------------------------------------

 We include the 
progress bar in the sampling step iff the verbose output is `True` *and* ``pySYD`` is not 
executed in parallel mode. This is hard-wired since the latter would produce a nightmare mess.

.. image:: _static/quickstart/1435467_samples.png
  :width: 680
  :alt: KIC 1435467 posteriors

^^ posteriors for KIC 1435467

Now, notice the difference in the output parameters this time... they have uncertainties!

.. csv-table:: 1435467 global parameters
   :header: "parameter", "value", "uncertainty"
   :widths: 20, 20, 20

   numax_smooth, 1303.82549513, 65.1861645150548
   A_smooth, 1.6981881189944, 0.208329237417828
   numax_gauss, 1354.18609943197, 43.0399300425255
   A_gauss, 1.45587282712706, 0.286045233580998
   FWHM, 284.631831313442, 64.5689284576161
   dnu, 70.653293964844, 0.81171745376397
   tau_1, 1069.91765124738, 2121.15050259705
   sigma_1, 31.1026782311927, 42.9475567908216
   tau_2, 218.303624326155, 20.2541392707925
   sigma_2, 85.4836783903674, 3.68355287162928

* matches expected output for model 4 selection - notice how there is no white noise term
in the output. this is because the model preferred for this to be fixed
   
.. note::

    While observations have shown that solar-like oscillations have an approximately 
    Gaussian-like envelope, we have no reason to believe that they should behave exactly 
    like that. This is why you will see two different estimates for :term:`numax` 
    (:math:`\rm \nu_{max}`) under the output parameters. ***In fact for this methodology 
    first demonstrated in Huber+2009, traditionally the smoothed numax has been used in 
    the literature and we recommend that you do the same.***

-----

Running your favorite star
##########################

A majority of the heavy lifting is done in the ``pySYD.target.Target`` class. Each star
that is processed is initialized as a new target object, which in this case, we'll call star.

    >>> from pysyd import utils
    >>> from pysyd.target import Target

hey

    >>> name = '1435467'
    >>> args = utils.Parameters(stars=[name])
    >>> star = Target(name, args)
    >>> if star.ok:
    ...    star.estimate_parameters()
    ...    plots.set_plot_params()
    ...    plots.plot_estimates()


.. plot::
    :align: center
    :context: close-figs
    :width: 60%

    from pysyd import utils
    from pysyd import plots
    from pysyd.target import Target
    import matplotlib.pyplot as plt

    name='1435467'
    args = utils.Parameters()
    star = Target(name, args)
    star.estimate_parameters()
    plots.set_plot_params()
    plots.plot_estimates()

    >>> from pysyd import plots



-----