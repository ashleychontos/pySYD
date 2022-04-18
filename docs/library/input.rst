.. role:: bolditalic
   :class: bolditalic

.. note::

   The initial `pySYD` release required *both* the light curve and power spectrum *but*
   this has since changed! For example, time-domain utilities in more recent implementations 
   have become available -- which is exciting but we are also still furiously working away on 
   this. **STAY TUNED FOR NEW TOOLS!** 

.. role:: underlined
   :class: underlined

.. _library-input:

**************
`pySYD` inputs
**************

In case you missed it -- there is a convenient way for you to get started right
away -- which is by running the ``pySYD`` setup feature. Running the command will provide 
*all* of the relevant files that are discussed in detail on this page. 

To read more details about the setup feature, please visit :ref:`this page <install-setup>` *or*
peep its API (:mod:`pysyd.pipeline.setup`). 

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
with any other normalization(s).

**Power spectrum:** the frequency series or :term:`power spectrum` is what's most important for 
the asteroseismic analyses applied and performed in this software. Thanks to open-source languages 
like `Python`, we have many powerful community-driven libraries like `astropy` that can fortunately 
compute these things for us.

Note: If you have both data series available but are not sure if the power spectrum is in the proper units,
we recommend that you provide the time series data as the only input (but of course still in the proper units).
That way, the software will calculate and normalize the power spectrum for you, which ensures
reliable results. **Please read below for more details about this!**

Cases
*****

Therefore for a given star, there are four different scenarios that arise from a combination of 
these two inputs and we describe how the software handles each of these cases.

Additionally, we will list these in the recommended order, where the top is the most preferred
and the bottom is the least.

:underlined:`Case 1: light curve *and* power spectrum`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, everything can be inferred and/or calculated from the data when both are provided. This
includes the time series :term:`cadence`, which is relevant for the :term:`nyquist frequency`,
or how high our sampling rate is. The total duration of the time series sets an upper limit
on the time scales we can measure and also sets the resolution of the power spectrum. Therefore
from this, we can determine if the power spectrum is oversampled or critically-sampled and
make the appropriate arrays for all input data.

Calculation(s)
 - Parameter(s):
   - time series cadence (:math:`\Delta t`)
   - nyquist frequency (:math:`\rm \nu_{nyq}`)
   - time series duration or baseline (:math:`\Delta T`)
   - frequency resolution (:math:`\Delta frequency`)
   - oversampling factor (i.e. critically-sampled has an `of=1`)
 - Array(s):
   - downsampled power density spectrum (when applicable)
   - critically-sampled power density spectrum

Issue(s)
 #. the only problem that can arise from this case is if the power spectrum is not 
    normalized correctly or in the proper units (i.e. frequency is in :math:`\rm \mu Hz` and power 
    is in :math:`\rm ppm^{2} \mu Hz^{-1}`). This is actually more common than you think so if this 
    *might* be the case, we recommend trying CASE 2 instead

:underlined:`Case 2: light curve *only*`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Again we can determine the baseline and cadence, which set important features in the 
frequency domain as well. Since the power spectrum is not yet calculated, we can control
if it's oversampled or critically-sampled

Calculation(s)
 - Parameter(s):
   - time series cadence (:math:`\Delta t`)
   - nyquist frequency (:math:`\rm \nu_{nyq}`)
   - time series duration or baseline (:math:`\Delta T`)
   - frequency resolution (:math:`\Delta frequency`)
   - oversampling factor (i.e. critically-sampled has an `of=1`)
 - Array(s):
   - oversampled power density spectrum
   - critically-sampled power density spectrum

Issue(s)
 #. 

:underlined:`Case 3: power spectrum *only*`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This case can be *o-k*, as long as additional information is provided.

Calculation(s)
 - Parameter(s):
 - Array(s):

Issue(s)
 #. 

Issue(s): 1) if oversampling factor not provided
          2) if not normalized properly

:underlined:`Case 4: no data`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   :header: "stars", "rs", "logg", "teff", "numax", "lower_ex", "upper_ex", "lower_bg"
   :widths: 20, 10, 10, 20, 20, 20, 20, 20

   1435467, 1.0, 4.4, 5777.0, 1400.0, 100.0, 5000.0, 100.0
   2309595, 1.0, 4.4, 5777.0, 1400.0, 100.0, 5000.0, 100.0

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
    >>> print(columns['all'])
    ['stars', 'rs', 'rs_err', 'teff', 'teff_err', 'logg', 'logg_err', 'show', 'save',  
     'verbose', 'overwrite', 'stitch', 'gap', 'kep_corr', 'oversampling_factor', 
     'excess', 'numax', 'dnu', 'binning', 'bin_mode', 'lower_ex', 'upper_ex', 'step', 
     'smooth_width', 'n_trials', 'ask', 'background', 'basis', 'box_filter', 'fix_wn', 
     'n_laws', 'ind_width', 'lower_bg', 'upper_bg', 'metric', 'n_rms', 'globe', 'ex_width',  
     'lower_ps', 'upper_ps', 'numax', 'sm_par', 'dnu', 'method', 'n_peaks', 'smooth_ps',  
     'threshold', 'hey', 'cmap', 'clip_value', 'interp_ech', 'notching', 'lower_ech', 
     'upper_ech', 'seed', 'nox', 'noy', 'smooth_ech', 'mc_iter', 'samples', 'n_threads',
     'inpdir', 'infdir', 'outdir', 'todo', 'info', 'functions']
    >>> len(columns['all'])
    67

**Note:** This file is *especially* helpful for running many stars with different options - you
can make your experience as customized as you'd like!

.. TODO:: Add all the available options (columns) to the csv and documentation
