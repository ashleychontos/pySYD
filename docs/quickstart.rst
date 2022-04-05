.. role::  raw-html(raw)
    :format: html

***************
Getting started
***************

The examples on this page assume that the user already has some basic-level knowledge or
experience with `Python`. If not, we recommend visiting the Python website and going through
some of `their tutorials <https://docs.python.org/3/tutorial/>`_ first before attempting 
ours.

On this page we will work through two examples -- each demonstrating a different way to
use the software. Since the software was initially intended to be a hands-off command-line tool,
the first example will run ``pySYD`` as a script to introduce both the software *and* science 
in a crash course to asteroseismology -- what we like to refer to as crashteroseismology.
We'll break everything down to be sure everyone is on the same page and then once we are all
expert asteroseismologists, we will reconstruct it in a condensed form by importing ``pySYD``
as a module.

If you have *any* questions, check out our :ref:`user guide <usage/index>` for more 
information. If this still does not address your question or problem, please do not hesitate
to contact `Ashley <achontos@hawaii.edu>`_ directly.

.. warning::

    It is **critical** that the input data are in the proper units in order for ``pySYD`` 
    to work properly and provide reliable results. If you are unsure about any of the units, 
    we recommend that you provide the light curve (in days) and then let us compute the power
    spectrum for you! For more information about formatting and input data, please visit
    :ref:`this page <library/input>`.

-----

.. _quickstart/crash:

Crashteroseismology
###################
**crash course in asteroseismology**

For purposes of this first example, we will assume that we do not know anything about the star or
its properties so that the software runs from start to finish. In any typical circumstance,
however,  we can provide optional inputs (e.g., the center of the frequency range with the 
oscillations, or :term:`numax` :math:`\rm \nu_{max}`) that can bypass some additionally steps
and save time. 

-----

.. _quickstart/script:

Initialize script
*****************

When running ``pySYD`` from command line, you will likely use something similar to the 
following statement: 

.. _quickstart/script/command:

.. code-block::

    pysyd run --star 1435467 -dv --ux 5000 --mc 200

Now let's deconstruct this statement.

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
   please visit :ref:`this page <usage/nb/numaxhacks.ipynb>`.

``--mc 200``
   last but certainly not least - the ``mc`` (for Monte Carlo-like) option sets the number 
   of iterations the pipeline will run for. In this case, the pipeline will run for 200 steps, 
   which allows us to bootstrap uncertainties on our derived properties. 

**Note:** For a *complete* list of options which are currently available via command-line interface (CLI), 
see our special CLI :ref:`glossary <usage/cli/glossary>`.

-----

.. _quickstart/script/steps:

Typical workflow
****************

The software operates in roughly the following steps:
 #. :ref:`Load in parameters and data <quickstart/script/steps/one>`
 #. :ref:`Get initial values <quickstart/script/steps/two>`
 #. :ref:`Fit global parameters <quickstart/script/steps/three>`
 #. :ref:`Estimate uncertainties <quickstart/script/steps/four>`

For each step, we will first show the relevant block of printed (or :term:`verbose<-v, --verbose>`) output, then
describe what the software is doing behind the scenes and if applicable, conclude with the section-specific 
results (i.e. files, figures, etc.).


.. _quickstart/script/steps/one:

1. Load in parameters and data
++++++++++++++++++++++++++++++

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

.. _quickstart/script/steps/two:

2. Get initial values
+++++++++++++++++++++

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

Finally, it saves this best estimate in a basic csv file:


.. csv-table:: 1435467 parameter estimates
   :header: "stars", "numax", "dnu", "snr"
   :widths: 20, 20, 20, 20

   1435467, 1438.27561061044, 72.3140769912867, 12.3801364686659


.. _quickstart/script/steps/three:

3. Fit global parameters
++++++++++++++++++++++++

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

   To learn more about what each panel is showing, please visit :ref:`this page <library/output/figures/png>`.

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


.. note::

    While observations have shown that solar-like oscillations have an approximately 
    Gaussian-like envelope, we have no reason to believe that they should behave exactly 
    like that. This is why you will see two different estimates for :term:`numax` 
    (:math:`\rm \nu_{max}`) under the output parameters. **In fact for this methodology 
    first demonstrated in Huber+2009, traditionally the smoothed numax has been used in 
    the literature and we recommend that you do the same.**


.. _quickstart/script/steps/four:

4. Estimate uncertainties
+++++++++++++++++++++++++

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

Notice the difference in the printed parameters this time - they now have uncertainties!

We include the progress bar in the sampling step iff the verbose output is `True` *and* ``pySYD`` is not 
executed in parallel mode. This is hard-wired since the latter would produce a nightmare mess.

.. image:: _static/quickstart/1435467_samples.png
  :width: 680
  :alt: KIC 1435467 posteriors

^^ posteriors for KIC 1435467

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
   


-----

.. _quickstart/module:

Running your favorite star
##########################

The two primary pieces to the `pySYD` puzzle are the 1) parameters and 2) target(s). Initially
all defaults were set and saved from the command line parser but we recently extended the 
software capabilities -- which means that it is more user-friendly now! 

Analogous to the command-line arguments, we have a container class :mod:`pysyd.utils.Parameters`
that can easily be loaded in and modified to the user's needs. Initialization of a `pysyd.utils.Parameters` 
class object also automatically inherits all attributes from the :mod:`pysyd.utils.Constants` class.

There are two keyword arguments that the Parameter class object accepts -- `args` and `stars` --
both which are `None` by default. This is convenient for this case, since we do not have any 
parameter (i.e. argument) information *yet*. In fact, the :mod:`pysyd.utils.Parameters` 
class was also initialized in the first example but immediately knew it was executed as a script 
because `args` was *not* `None`.

If we are going through these steps, there's probably a decent chance that we know what star we want
to process. Therefore, we can at least provide the star name in this first step.

    >>> from pysyd import utils 
    >>> name = '1435467'
    >>> args = utils.Parameters(stars=[name])
    >>> args
    <pysyd Parameters>

As shown in the third line, we put the star list in list form **even though we are only processing 
a single star**. This is because both ``pySYD`` `run` and `parallel` modes iterate through stars, so 
we need something that is iterable. Now that we have our parameters, we need a star. Well *technically*
we already have our star but we need to load in the data by creating an instance of the 
:mod:`pysyd.target.Target`.

    >>> from pysyd.target import Target
    >>> star = Target(name, args)
    >>> star
    <Star Object 1435467>

Typically this step will flag anything that doesn't seem right in the event that data is missing or
the path is not correct *but just in case*, there is also an `ok` attribute -- which literally means 
the star is o-k to go! `Target.ok` is simply a boolean flag but let's check it for good practice:

    >>> star.ok
    True

Finally, we will use the same settings we used in the first example -- so we need to update those first
before running.

    >>> star.params['verbose'] = True
    >>> star.params['show'] = True
    >>> star.params['upper_ex'] = 5000.
    >>> star.params['mc_iter'] = 200

Ok, now that we have our desired settings and target, we can go ahead and process the star!

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