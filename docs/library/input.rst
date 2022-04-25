.. role:: bolditalic
   :class: bolditalic

.. role:: underlined
   :class: underlined

.. _library-input:

**************
`pySYD` inputs
**************

**For what it's worth** and if you haven't done so already, running the `pySYD` 
:ref:`setup feature <install-setup>` will conveniently provide *all* of files which are 
discussed in detail on this page. 

.. _library-input-required:

:underlined:`Required` 
######################

The only thing that's really *required* is the data. 

For a given star `ID`, possible input data are its:
 #. light curve (`'ID_LC.txt'`) and/or
 #. power spectrum (`'ID_PS.txt'`).

**Light curve:** The *Kepler*, K2 & TESS missions have provided *billions* of stellar light curves, or a 
measure of the object's brightness (or flux) in time. Like most standard photometric 
data, we require that the time array is in units of days. **This is really important if
the software is calculating the power spectrum for you!** The y-axis is less critical here -- 
it can be anything from units of fraction flux or brightness as a function of time, along 
with any other normalization(s). **Units:** time (days) vs. normalized flux (ppm)

**Power spectrum:** the frequency series or :term:`power spectrum` is what's most important for 
the asteroseismic analyses applied and performed in this software. Thanks to open-source languages 
like `Python`, we have many powerful community-driven libraries like `astropy` that can fortunately 
compute these things for us. **Units:** frequency (:math:`\rm \mu Hz`) vs. power density 
(:math:`\rm ppm^{2} \mu Hz^{-1}`)

Cases
*****

Therefore for a given star, there are four different scenarios that arise from a combination of 
these two inputs and we describe how the software handles each of these cases.

Additionally, we will list these in the recommended order, where the top is the most preferred
and the bottom is the least.

Case 1: light curve *and* power spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, everything can be inferred and/or calculated from the data when both are provided. This
includes the time series :term:`cadence`, which is relevant for the :term:`nyquist frequency`,
or how high our sampling rate is. The total duration of the time series sets an upper limit
on the time scales we can measure and also sets the resolution of the power spectrum. Therefore
from this, we can determine if the power spectrum is oversampled or critically-sampled and
make the appropriate arrays for all input data.

The following are attributes saved to the `pysyd.target.Target` object in this scenario:

 - Parameter(s): 

   - time series cadence (`star.cadence`)
   - nyquist frequency (`star.nyquist`)
   - total time series length or baseline (`star.baseline`)
   - upper limit for granulation time scales (`star.tau_upper`)
   - frequency resolution (`star.resolution`)
   - oversampling factor (`star.oversampling_factor`)

 - Array(s):

   - time series (`star.time` & `star.flux`)
   - power spectrum (`star.frequency` & `star.power`)
   - copy of input power spectrum (`star.freq_os` & `star.pow_os`)
   - critically-sampled power spectrum (`star.freq_cs` & `star.pow_cs`)

Issue(s)

 #. the only problem that can arise from this case is if the power spectrum is not 
    normalized correctly or in the proper units (i.e. frequency is in :math:`\rm \mu Hz` and power 
    is in :math:`\rm ppm^{2} \mu Hz^{-1}`). This is actually more common than you think so if this 
    *might* be the case, we recommend trying CASE 2 instead


Case 2: light curve *only*
^^^^^^^^^^^^^^^^^^^^^^^^^^

Again we can determine the baseline and cadence, which set important features in the 
frequency domain as well. Since the power spectrum is not yet calculated, we can control
if it's oversampled or critically-sampled. So basically for this case, we can calculate
all the same things as in Case 1 *but* we just have a few more steps that may take a little
more time to do. 

The following are attributes saved to the `pysyd.target.Target` object in this scenario:

 - Parameter(s): 

   - time series cadence (`star.cadence`)
   - nyquist frequency (`star.nyquist`)
   - total time series length or baseline (`star.baseline`)
   - upper limit for granulation time scales (`star.tau_upper`)
   - frequency resolution (`star.resolution`)
   - oversampling factor (`star.oversampling_factor`)

 - Array(s):

   - time series (`star.time` & `star.flux`)
   - newly-computed power spectrum (`star.frequency` & `star.power`)
   - copy of oversampled power spectrum (`star.freq_os` & `star.pow_os`)
   - critically-sampled power spectrum (`star.freq_cs` & `star.pow_cs`)

Issue(s)

 #. 


Case 3: power spectrum *only*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This case can be *o-k*, so long as additional information is provided.

Calculation(s)
 - Parameter(s):
 - Array(s):

Issue(s)
 #. 

Issue(s): 1) if oversampling factor not provided
          2) if not normalized properly

Case 4: no data
^^^^^^^^^^^^^^^

well, we all know what happens when zero input is provided... but just in case,
this will raise a `PySYDInputError`

**CASE 1: light curve** :bolditalic:`and` **power spectrum**
- :underlined:`Summary`: 
- :underlined:`Calculation(s)`:
   - time series cadence (:math:`\Delta t`)
   - nyquist frequency (:math:`\rm \nu_{nyq}`)
   - time series duration or baseline (:math:`\Delta T`)
   - frequency resolution (:math:`\Delta frequency`)
   - oversampling factor (i.e. critically-sampled has an `of=1`)
   - critically-sampled power spectrum
- :underlined:`Issue(s)`: 
   - the only problem that can arise from this case is if the power spectrum is not 
     normalized correctly or in the proper units (i.e. frequency is in :math:`\rm \mu Hz` and power 
     is in :math:`\rm ppm^{2} \mu Hz^{-1}`). This is actually more common than you think so if this 
     *might* be the case, we recommend trying CASE 2 instead.

**CASE 2:** light curve *only*
- summary: Again we can determine the baseline and cadence, which set important features in the 
  frequency domain as well. Since the power spectrum is not yet calculated, we can control
  if it's oversampled or critically-sampled

**CASE 3:** power spectrum *only*
This case *can* be alright, as long as additional information is provided.
Issue(s): 1) if oversampling factor not provided
          2) if not normalized properly


.. important::

    For the saved power spectrum, the frequency array has units of :math:`\rm \mu Hz` and the
    power array is power density, which has units of :math:`\rm ppm^{2} \, \mu Hz^{-1}`. We 
    normalize the power spectrum according to Parseval's Theorem, which loosely means that the 
    fourier transform is unitary. This last bit is incredibly important for two main reasons,
    but both that tie to the noise properties in the power spectrum: 1) different instruments
    (e.g., *Kepler*, TESS) have different systematics and hence, noise properties, and 2) the 
    amplitude of the noise becomes smaller as your time series gets longer. Therefore when we 
    normalize the power spectrum, we can make direct comparisons between power spectra of not
    only different stars, but from different instruments as well!


.. _library-input-optional:

:underlined:`Optional`
######################

There are two main information files that can be provided but both are optional -- whether
you choose to use them or not is ultimately up to you! 

.. _library-input-optional-todo:

Target list
***********

For example, providing a star list via a basic text file is convenient for running a large 
sample of stars. We provided an example with the rest of the setup, but essentially all it
is is a list with one star ID per line. The star ID *must* match the same ID associated
with the data.

.. code-block::

    $ cat todo.txt
    11618103
    2309595
    1435467

**Note:** If no stars are specified via command line or in a notebook, ``pySYD`` will read 
in this text file and process the list of stars by default. 

.. _library-input-optional-info:

Star info
*********

As suggested by the name of the file, this contains star information on an individual basis. Similar to
the data, target IDs must *exactly* match the given name in order to be successfully crossmatched -- but
this also means that the information in this file need not be in any particular order. 

Below is a snippet of what the csv would look like:

.. csv-table:: Star info
   :header: "stars", "rs", "logg", "teff", "numax", "lower_se", "upper_se", "lower_bg"
   :widths: 20, 10, 10, 20, 20, 20, 20, 20

   1435467, , , , , 100.0, 5000.0, 100.0
   2309595, , , , , 100.0, , 100.0

Just like the input data, the `stars` *must* match their ID but also, the commands
must adhere to a special format. In fact, the columns in this csv are exactly equal to
the value (or `destination`) that the command-line parser saves each option to. Since
there are a ton of available columns, we won't list them all here but there are a few ways
you can view the columns for yourself.

The first is by visiting our special :ref:`command-line glossary <usage/cli/glossary>`, 
which explicitly states how each of the variables is defined. You can also see
them fairly easily by importing the :mod:`pysyd.utils.get_dict` module and doing a
basic `print` statement.

    >>> from pysyd import utils
    >>> columns = utils.get_dict('columns')
    >>> print(columns['all])
    ['rs', 'rs_err', 'teff', 'teff_err', 'logg', 'logg_err', 'cli', 'inpdir', 
     'infdir', 'outdir', 'overwrite', 'show', 'ret', 'save', 'test', 'verbose', 
     'dnu', 'gap', 'info', 'ignore', 'kep_corr', 'lower_ff', 'lower_lc', 'lower_ps',
     'mode', 'notching', 'oversampling_factor', 'seed', 'stars', 'todo', 'upper_ff', 
     'upper_lc', 'upper_ps', 'stitch', 'n_threads', 'ask', 'binning', 'bin_mode', 
     'estimate', 'adjust', 'lower_se', 'n_trials', 'smooth_width', 'step', 
     'upper_se', 'background', 'basis', 'box_filter', 'ind_width', 'n_laws', 
     'lower_bg', 'metric', 'models', 'n_rms', 'upper_bg', 'fix_wn', 'functions',  
     'cmap', 'clip_value', 'fft', 'globe', 'interp_ech', 'lower_osc', 'mc_iter', 
     'nox', 'noy', 'npb', 'n_peaks', 'numax', 'osc_width', 'smooth_ech', 'sm_par', 
     'smooth_ps', 'threshold', 'upper_osc', 'hey', 'samples']
    >>> len(columns['all'])
    77

**Note:** This file is *especially* helpful for running many stars with different options - you
can make your experience as customized as you'd like!

.. TODO:: Add all the available options (columns) to the csv and documentation
