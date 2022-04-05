.. note::

   The initial ``pySYD`` release required *both* the light curve and power spectrum *but*
   this has since changed! For example, time-domain utilities in more recent implementations 
   have become available -- which is exciting but we are also still furiously working away on 
   this. ***STAY TUNED FOR NEW TOOLS!*** 

.. _library/input:

******
Inputs
******

In case you missed it -- there is a convenient way for you to get started right
away -- which is by running the ``pySYD`` setup feature. Running the command will provide 
*all* of the relevant files that are discussed in detail on this page. 

To read more details about the setup feature, please visit our :ref:`Installation <installation>` 
or :ref:`Quickstart <quickstart>` pages. You can also see how it works directly by visiting
:ref:`the API <library-pipeline>`. 

-----

.. _library/input/required:

Required 
########

The only thing that is really *required* to successfully run the software is the data! 

Data types
**********

For a given star with ID, input data are:
 #. the light curve, and
 #. the power spectrum.

.. _library/input/required/lc:

Light curve
^^^^^^^^^^^

The *Kepler*, K2 & TESS missions have provided *billions* of stellar light curves, or a 
measure of the object's brightness (or flux) in time. Like most other standard photometric 
data, we require that the time array is in units of days. **This will be really important
for the processing of the data, which we'll discuss in detail in a little bit.**

For the time series data, the y-axis is less critical here. It can be anything from units 
of fraction flux or brightness as a function of time, along with any other normalization(s).

.. _library/input/required/ps:

Power spectrum
^^^^^^^^^^^^^^

What *REALLY* matters for asteroseismology is how the time series data looks in frequency space, 
which is generally calculated by taking the fourier transform (and often referred to as the
:term:`power spectrum`). Thanks to open-source languages like Python, we have powerful
community-driven software packages like `astropy` that can fortunately compute these things for us.

.. warning::

    Again, it is **critical** that these files are in the proper units in order for ``pySYD`` 
    to work properly. If you are unsure about any of these units, your best bet is to
    provide a light curve (in days) and let us calculate the power spectrum for you! 


-----

.. _library/input/optional:

Optional
########

There are two main information files which can be provided but both are optional -- whether
or not you choose to use them ultimately depends on how you will run the software. 

.. _library/input/optional/todo:

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

.. _library/input/optional/info:

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
    
-----
