**********
Quickstart
**********

The examples on this page assume that the user already has some basic-level knowledge or
experience with `Python`. If not, we recommend visiting the Python website and going through
some of `their tutorials <https://docs.python.org/3/tutorial/>`_ first before attempting 
our examples.

**Jump to** ``pySYD`` **used:**
 - :ref:`via command line<script>`
 - :ref:`as a module<module>`

-----

Installation & setup
####################

We recommend creating a local directory to keep all your pysyd-related data, results 
and information in a single, easy-to-find location. The software package comes with a 
convenient setup feature that downloads example data so let's start there.

.. TODO:: add an option to download example data/files as a package in the root directory.

Open up a terminal window and enter the following commands:

.. code-block::

    python -m pip install pysyd
    mkdir ~/path/to/local/pysyd/directory
    cd ~/path/to/local/pysyd/directory
    pysyd setup --verbose

You may have noticed that for this last point, we used the optional 
:term:`--verbose<-v, --verbose>` command, which will print the following:

.. code-block::
    
    Downloading relevant data from source directory:
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100    25  100    25    0     0     49      0 --:--:-- --:--:-- --:--:--    49
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100   239  100   239    0     0    508      0 --:--:-- --:--:-- --:--:--   508
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 1518k  100 1518k    0     0  1601k      0 --:--:-- --:--:-- --:--:-- 1601k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 3304k  100 3304k    0     0  2958k      0  0:00:01  0:00:01 --:--:-- 2958k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 1679k  100 1679k    0     0  1630k      0  0:00:01  0:00:01 --:--:-- 1630k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 3523k  100 3523k    0     0  3101k      0  0:00:01  0:00:01 --:--:-- 3099k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 1086k  100 1086k    0     0   943k      0  0:00:01  0:00:01 --:--:--  943k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
     100 2578k  100 2578k    0     0  2391k      0  0:00:01  0:00:01 --:--:-- 2391k
    
      - created input file directory: /Users/ashleychontos/Desktop/pysyd/info
      - created data directory at /Users/ashleychontos/Desktop/pysyd/data
      - example data saved
      - results will be saved to /Users/ashleychontos/Desktop/pysyd/results
    
As shown above, :mod:`pysyd.pipeline.setup` downloaded example data and other relevant files
from the `public GitHub repo <https://github.com/ashleychontos/pySYD>`_. It also established
a local, relative directory structure that is both straightforward for the pipeline and 
intuitive to the user.

You are now ready to become an asteroseismologist!

-----

TL;DR
#####

We wanted to go over a couple general tidbits about the software before going through the
examples. 

The first is that the intended userbase for the original `pySYD` release was meant for 
non-expert astronomers. Therefore, the software was developed to be as hands-off as possible 
as a *strictly* command-line end-to-end tool. However since then, the software has become 
more modular in recent updates, thus enabling broader capabilities that can be used in other 
applications (e.g., Jupyter notebooks). 

In addition to being a command-line tool, the software is optimized for running an ensemble
of stars. This means that many of the options that one would typically use, such as printing 
output information and displaying figures, is actually `False` by default. For our purposes 
here though, we will enable them to better understand how the software works. 

Here we will go through two different scenarios -- each demonstrating the two mains ways 
you can use the software. We will start with the command-line example to break everything 
down, and then put it back together in a condensed version for the second. 

If you have *any* questions, please check our :ref:`user guide <usage/index>` for more 
information. If this still does not address your question or problem, feel free to contact 
`Ashley <achontos@hawaii.edu>`_ directly.

-----

*****************************************
Running your first asteroseismic analyses
*****************************************

In general, the software operates in the following steps:
 #. :ref:`Load in parameters and data <stepone>`
 #. :ref:`Estimate initial values <steptwo>`
 #. :ref:`Fit global parameters <stepthree>`
 #. :ref:`Extrapolate uncertainties <stepfour>`

Each of the four main steps are discussed in detail below.

.. warning::

    It is **critical** that the input data are in the proper units in order for ``pySYD`` 
    to work properly and provide reliable results. If you are unsure about any of the units, 
    we recommend that you provide the light curve (in days) and then let us compute the power
    spectrum for you! 

.. _script:

I. As a script
##############

In a terminal window, type the following statement:

.. code-block::

    pysyd run --star 1435467 -dv --ux 5000 --mc 200

which we will now deconstruct. 

 * ``pysyd`` : if you used `pip` install, the binary (or executable) ``pysyd`` is available. In fact, the setup
   file defines this entry point for ``pysyd`` and is accessed through the :mod:`pysyd.cli.main` script -- which is
   also where you can find the parser with all the available commands and options.
 * ``run`` : regardless of how you choose to use the software, the most common way you will likely implement
   the ``pySYD`` pipeline is in its run (i.e. :mod:`pysyd.pipeline.run`) mode -- which, just as it sounds, will process
   stars in the order they were provided. This is saved to the argument ``NameSpace`` as the 'mode' as in the 
   `pysyd` pipeline mode. There are currently five available modes, all which are described in more detail
   :ref:`here <library/pipeline>`
 * ``--star 1435467`` : here we are running a single star, KIC 1435467. You can also provide multiple targets,
   the stars which will be appended to a list and then processed consecutively. On the other 
   hand if no targets are provided, the program would default to reading in the star or 'todo' 
   list (via 'info/todo.txt'). Again, this is because the software is optimized for 
   running an ensemble of stars.
 * ``-dv`` : adapting Linux-like features, we reserved the single hash options for booleans which
   can all be grouped together, as shown above. The ``-d`` and ``-v`` are short for display and verbose, 
   respectively, and show the figures and verbose output. For a full list of options available, please 
   see our :ref:`command-line glossary <usage/cli/glossary>`. There are dozens of options to make your 
   experience as customized as you'd like!
 * ``--ux 5000`` : this is an upper frequency limit for the first module that identifies the power eXcess 
   due to solar-like oscillations. In this case, there are high frequency artefacts that we would 
   like to ignore. *We actually made a special notebook tutorial specifically on how to address
   and fix this problem.* If you'd like to learn more about this or are having a similar issue, 
   please visit :ref:`this page <usage/nb/estimatenumax.ipynb>`.
 * ``--mc 200`` : last but certainly not least - the ``mc`` (for Monte Carlo-like) option sets the number 
   of iterations the pipeline will run for. In this case, the pipeline will run for 200 steps, which allows 
   us to bootstrap uncertainties on our derived properties. 

After hitting return, you'll immediately notice that the output for the above command is actually 
quite long. Not to worry though - we will break it down into the four main ``pySYD`` steps
mentioned in the summary above.  different sections and explain 
each in great detail. In fact, each of the four sections correspond to the four main ``pySYD`` 
steps discussed in the summary above.

***Important: when running `pysyd` as a script, there is one positional argument.*** 


.. _stepone:

1. Load in parameters and data
******************************

If there are issues during the first step, ``pySYD`` will flag this and immediately halt 
any further execution of the code. 

Verbose output
^^^^^^^^^^^^^^

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

It appears as though this star, KIC 1435467, was observed in *Kepler* short-cadence (e.g., 
1-minute cadence) data - which was used to compute the (oversampled) power spectrum. 
There are many exceptions in place during this step that will flag anything that does not 
seem right. If something seems questionable during this step but is not fatal, it will only 
return some warnings. Since none of this happened, we can assume that there were no issues
accessing and storing the data.

All :mod:`pysyd.target` class instances will have an ``ok`` attribute - literally meaning 
that the star is 'ok' to be processed. By default, the pipeline checks this attribute before 
moving on. Since everything checks out, we can move on!


.. _steptwo:

2. Estimate initial values
**************************

For purposes of the example, we will assume that we do not know anything about its properties. 
Typically we can provide optional inputs in many different ways but we won't here so it can 
estimate the properties on its own.

Verbose output
^^^^^^^^^^^^^^

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

As discussed above, the main thing we need to know before doing the global fit is a rough 
starting point for the frequency corresponding to maximum power, or :term:`numax` (:math:`\rm \nu_{max}`).

It does this by making a very rough approximation of the stellar background by binning the 
power spectrum in both log and linear spaces (think a very HEAVY smoothing filter) and divides
this out so that we are left with very little residual slope in the power spectrum.

Next it uses a "collapsed" autocorrelation function (ACF) technique with different bin sizes
to identify localized power excess in the power spectrum due to solar-like oscillations. By
default, this is done three times (or trials) and hence, get three different estimates.


Result figure
^^^^^^^^^^^^^

.. image:: _static/quickstart/1435467_estimates.png
  :width: 680
  :alt: Parameter estimates for KIC 1435467


***To learn more about what each panel is showing, please visit :ref:`this page<library/output>`.***


Result file
^^^^^^^^^^^

.. csv-table:: 1435467 parameter estimates
   :header: "stars", "numax", "dnu", "snr"
   :widths: 20, 20, 20, 20

   1435467, 1438.27561061044, 72.3140769912867, 12.3801364686659


.. _stepthree:

3. Fit global parameters
************************

A bulk of the heavy lifting is done in this main fitting routine, which is actually done 
in two separate steps: 1) modeling and characterizing the stellar background and 2) determining 
the global asteroseismic parameters. We do this *separately* in two steps because they have 
fairly different properties and we wouldn't want either of the estimates to be influenced by 
the other in any way. 

Ultimately the stellar background has more of a presence in the power spectrum in that it is 
observed over a wider range of frequencies compared to the solar-like oscillations. Therefore 
by attempting to identify where the oscillations are in the power spectrum, we can mask 
them out to better characterize the background.


Verbose output
^^^^^^^^^^^^^^

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

Unlike previous versions of this software and previous versions of this software (i.e. `SYD`), 
we have now implemented an automated background model selection. For reference, 

After much trial and error, the :term:`AIC` seems to perform better for our purposes - which
is why this is now the default metric used.


Result figure
^^^^^^^^^^^^^

.. image:: _static/quickstart/1435467_global.png
  :width: 680
  :alt: Global parameters for KIC 1435467


Result file
^^^^^^^^^^^

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



.. _stepfour:

4. Extrapolate uncertainties
****************************

If this was run in its default settings (with --mc 1) for a single iteration, the output
would look comparable but with no progress bar and no parameter uncertainties.

Verbose output
^^^^^^^^^^^^^^

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


We include a progress bar in the sampling step iff the verbose output is `True` *and*
``pySYD`` is not executed in parallel mode. This is hard-wired since the latter would
produce a nightmare mess.


Result figure
^^^^^^^^^^^^^

.. image:: _static/quickstart/1435467_samples.png
  :width: 680
  :alt: KIC 1435467 posteriors

^^ posteriors for KIC 1435467

Result file
^^^^^^^^^^^

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

.. _module:

II. As a module
###############

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