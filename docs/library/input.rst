.. note::

   The initial ``pySYD`` release required *both* time series and power spectrum data but
   has since changed. The time-domain utilities and recent implementations are brand new
   but we are furiously working away on this so stay tuned for new tools!

******
Inputs
******

In case you missed it -- there is a convenient way for you to get started right
away -- which is by running the ``pySYD`` setup feature. Running the command will provide 
*all* of the relevant files that are discussed in detail on this page. 

To read more details about the setup feature, please visit our :ref:`Installation <installation>` 
or :ref:`Quickstart <quickstart>` pages. You can also see how it works directly by visiting
:ref:`the API <pipeline>`. 


Required
########

The only thing *required* to successfully run the software and obtain results is the data! 

Data 
****

For a given star with ID, input data are:
 #. the light curve, and
 #. the power spectrum.

The light curve data should be in units of fractional flux or brightness as a function of
days (the time unit is very important here). The power spectrum should be in units of power
or power density versus :math:`\rm \mu Hz`.

.. warning::

    It is **critical** that these files are in the proper units in order for ``pySYD`` 
    to work properly. If you are unsure about any of these units, your best bet is to
    provide a light curve (in days) and let us calculate the power spectrum for you! 

.. warning::

    Time and frequency *must* be in the specified units in order for the pipeline to properly process 
    the data and provide reliable results. **If you are unsure about this, we recommend**
    **ONLY providing the time series data in order to let** ``pySYD`` **calculate and
    normalize the power spectrum for you.** Again, if you choose to do this, the time series data
    *must* be in units of days in order for the frequency array to be calculated correctly. For
    more information on formatting and inputs, please see :ref:`here <library/input>`.


Optional 
########

There are two main information files which can be provided but both are optional -- whether
or not you choose to use them ultimately depends on how you will run the software. 

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

Star info
*********

As suggested by the name of the file, this contains star information on an individual basis. Similar to
the data, target IDs must *exactly* match the given name in order to be successfully crossmatched -- but
this also means that the information in this file need not be in any particular order. 

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

The first is by visiting our special :ref:`command-line glossary<usage/cli/glossary>`, 
which explicitly states how each of the variables is defined. You can also see
them fairly easily by importing the :mod:`pysyd.utils.get_dict` module and doing a
basic `print` statement.

    >>> from pysyd import utils
    >>> columns = utils.get_dict('columns')
    >>> print(columns['all'])
    ['stars', 'rs', 'rs_err', 'teff', 'teff_err', 'logg', 'logg_err', 'show', 'save', 'verbose', 
     'overwrite', 'stitch', 'gap', 'kep_corr', 'oversampling_factor', 'excess', 'numax', 'dnu', 
     'binning', 'bin_mode', 'lower_ex', 'upper_ex', 'step', 'smooth_width', 'n_trials', 'ask', 
     'background', 'basis', 'box_filter', 'fix_wn', 'n_laws', 'ind_width', 'lower_bg', 'upper_bg', 
     'metric', 'n_rms', 'globe', 'ex_width', 'lower_ps', 'upper_ps', 'numax', 'sm_par', 'dnu', 
     'method', 'n_peaks', 'smooth_ps', 'threshold', 'hey', 'cmap', 'clip_value', 'interp_ech', 
     'notching', 'lower_ech', 'upper_ech', 'seed', 'nox', 'noy', 'smooth_ech', 'mc_iter', 'samples', 
     'n_threads', 'inpdir', 'infdir', 'outdir', 'todo', 'info', 'functions']
    >>> len(columns['all'])
    67

**Note:** This file is *especially* helpful for running many stars with different options - you
can make your experience as customized as you'd like!

.. TODO:: Add all the available options (columns) to the csv and documentation
    

