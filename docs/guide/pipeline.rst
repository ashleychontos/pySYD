
*****************************
Pipeline [``pysyd.pipeline``]
*****************************

Introduction
############

There are currently five operational ``pySYD`` modes: 

#. ``setup`` : Initializes ``pysyd.pipeline.setup`` for quick and easy setup of directories, files and examples. This mode only
   inherits higher level functionality and has limited CLI (see :ref:`parent parser<parentparse>` below). Using this feature will
   set up the paths and files consistent with what is recommended and discussed in more detail below.

#. ``load`` : Loads in data for a single target through ``pysyd.pipeline.load``. Because this does handle data, this has 
   full access to both the :ref:`parent<parentparse>` and :ref:`main parser<mainparse>`.

#. ``run`` : The main pySYD pipeline function is initialized through ``pysyd.pipeline.run`` and runs the two core modules 
   (i.e. ``find_excess`` and ``fit_background``) for each star consecutively. This mode operates using most CLI options, inheriting
   both the :ref:`parent<parentparse>` and :ref:`main parser<mainparse>` options.

#. ``parallel`` : Operates the same way as the previous mode, but processes stars simultaneously in parallel. Based on the number of threads
   available, stars are separated into groups (where the number of groups is exactly equal to the number of threads). This mode uses all CLI
   options, including the number of threads to use for parallelization (:ref:`see here<parallel>`).

#. ``test`` : Currently under development but intended for developers.


API
###

Used with the command line interface (CLI), which will accept the command line arguments
and initialize the appropriate module. This includes the main pipeline initialization, which
now has parallelization capabilities.

.. automodule:: pysyd.pipeline
   :members:
