.. _cli:

Command Line Interface
======================

By default, initialization of the ``pySYD`` command will use the paths:

- ``TODODIR`` : '~/path_to_put_pysyd_stuff/info/todo.txt'
- ``INFODIR`` : '~/path_to_put_pysyd_stuff/info/star_info.csv'
- ``INPDIR`` : '~/path_to_put_pysyd_stuff/data'
- ``OUTDIR`` : '~/path_to_put_pysyd_stuff/results'

based on the absolute path of the current working directory. All of these paths are already ready to go
if you followed the suggestions in :ref:`structure`, used our ``setup`` feature or ran the :ref:`sixtyseconds`
challenge.

Parent Parser
+++++++++++++

The ``pySYD`` command line feature has two subcommands: ``setup`` and ``run``, the former which should only really
be used once after installation. Command line options inherent to both subcommands include:

* ``-file``, ``--file``, ``-list``, ``--list``, ``-todo``, ``--todo`` : path to text file that contains the list of stars to process
   * dest = ``args.file``
   * type = string
   * default = ``TODODIR``
* ``-in``, ``--in``, ``-input``, ``--input``, ``-inpdir``, ``--inpdir`` : path to input data
   * dest = ``args.inpdir``
   * type = string
   * default = ``INPDIR``
* ``-info``, ``--info``, ``-information``, ``--information`` : path to the csv containing star information
   * dest = ``args.info``
   * type = string
   * default = ``INFODIR``
* ``-out``, ``--out``, ``-output``, ``--output``, ``-outdir``, ``--outdir`` : path that results are saved to
   * dest = ``args.outdir``
   * type = string
   * default = ``OUTDIR``
* ``-verbose``, ``--verbose`` : turn on verbose output
   * dest = ``args.verbose``
   * type = boolean
   * default = ``False``
   * action = ``store_true``
   

Setup
+++++

Initializes ``pysyd.pipeline.setup`` for quick and painless setup of directories, files, and examples. 


Run
+++

The main pySYD pipeline function occurs in two main steps: ``find_excess`` and ``fit_background``. Command line
options relevant to higher level pipeline initialization include:

* ``-bg``, ``--bg``, ``-fitbg``, ``--fitbg``, ``-background``, ``--background`` : turn off the background fitting procedure
   * dest = ``args.background``
   * type = boolean
   * default = ``True``
   * action = ``store_false``
* ``-ex``, ``--ex``, ``-findex``, ``--findex``, ``-excess``, ``--excess`` : turn off the find excess module
   * dest = ``args.background``
   * type = boolean
   * default = ``True``
   * action = ``store_false``
* ``-kc``, ``--kc``, ``-keplercorr``, ``--keplercorr`` : turn on the *Kepler* short-cadence artefact correction module
   * dest = ``args.keplercorr``
   * type = boolean
   * default = ``False``
   * action = ``store_true``
* ``-nt``, ``--nt``, ``-nthread``, ``--nthread``, ``-nthreads``, ``--nthreads`` : number of processes to run in parallel
   * dest = ``args.n_threads``
   * type = int
   * default = ``0``
* ``-par``, ``--par``, ``-parallel``, ``--parallel`` : enable parallel processes for an ensemble of stars
   * dest = ``args.parallel``
   * type = boolean
   * default = ``False``
   * action = ``store_true``
* ``-save``, ``--save`` : save output files and figures
   * dest = ``args.save``
   * type = boolean
   * default = ``True``
   * action = ``store_false``
* ``-show``, ``--show`` : show output figures (note: this is not recommended if running many stars)
   * dest = ``args.show``
   * type = boolean
   * default = ``False``
   * action = ``store_true``
* ``-star``, ``--star``, ``-stars``, ``--stars`` : list of stars to process (note: if ``None``, pySYD will default to star list read from ``args.file``)
   * dest = ``args.star``
   * nargs = '*'
   * type = int
   * default = ``None``
   
**Excess:**

**Background:**

* `-filter`, `--filter`, `-smooth`, `--smooth` [float]

Box filter width in muHz for the power spectrum. The default is `2.5` muHz but will change to `0.5` muHz if the numax derived from `find_excess` or the numax provided in `info/stars_info.csv` is <= 500 muHz so that it doesn't oversmooth the power spectrum.

* `-mc`, `--mc`, `-mciter`, `--mciter` [int]

Number of MC iterations to run to quantify measurement uncertainties. It is recommended to check the results first before implementing this option and therefore, this is set to `1` by default.
