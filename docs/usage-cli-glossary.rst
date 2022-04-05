******************
List of CL options
******************

.. glossary::

    ``-a, --ask``
        the option to select which trial (or estimate) of numax to use from the first module
         * dest = ``args.ask``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
         * **see also:** :term:`--trials<--trials, --ntrials>`, :term:`--ntrials<--trials, --ntrials>`
    
    ``-b, --bg, --background``
        controls the background-fitting procedure -- BUT this should never be touched
        since a majority of the work done in the software happens here and it should 
        not need to be turned off
         * dest = ``args.background``
         * type = `bool`
         * default = `True`
         * action = ``store_false``
    
    ``--basis``
        which basis to use for the background fitting (i.e. `'a_b'`, `'pgran_tau'`, 
        `'tau_sigma'`), **NOT OPERATIONAL YET**
         * dest = ``args.basis``
         * type = `str`
         * default = `'tau_sigma'`
    
    ``--bf, --box, --boxfilter``
        box filter width for plotting the power spectrum **TODO:** make sure this does
        not affect any actual measurements and this is just an aesthetic
         * dest = ``args.box_filter``
         * type = `float`
         * default = `1.0`
         * unit = :math:`\mu \mathrm{Hz}`
         
    ``--bin, --binning``
        interval for the binning of spectrum in :math:`\mathrm{log(}\mu\mathrm{Hz)}`
        *this bins equally in logspace*
         * dest = ``args.binning``
         * type = `float`
         * default = `0.005`
         * unit = log(:math:`\mu \mathrm{Hz}`)

    ``--bm, --mode, --bmode``
        which mode to choose when binning. 
        Choices are ~[`"mean"`, `"median"`, `"gaussian"`]
         * dest = ``args.mode``
         * type = `str`
         * default = `"mean"`

    ``--ce, --cm, --color``
        change the colormap used in the echelle diagram, 
        which is `'binary'` by default
         * dest = ``args.cmap``
         * type = `str`
         * default = `'binary'`
    
    ``--cv, --value``
        the clip value to use for the output echelle diagram if and only if ``args.clip_ech`` is
        ``True``. If none is provided, it will use a value that is 3x the median value of the folded
        power spectrum
         * dest = ``args.clip_value``
         * type = `float`
         * default = `3.0`
         * unit = fractional psd
    
    ``--cli``
        this should never be touched - for internal workings on how to retrieve and save parameters
         * dest = ``args.cli``
         * type = `bool`
         * default = `True`
         * action = ``store_true``

    ``-d, --show, --display``
        show output figures, which is 
        not recommended if running many stars
         * dest = ``args.show``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--dnu``
        brute force method to provide value for dnu. **Note:** if using the ``pysyd.utils.whiten_mixed`` 
        modes module, this will need to be provided along with :term:`--le` and :term:`--ue`.
         * dest = ``args.dnu``
         * type = `float`
         * nargs = `'*'`
         * default = `None`
         * **REQUIRES:** :term:`--le<--le, --lowere>`/:term:`--lowere<--le, --lowere>` *and* :term:`--ue<--ue, --uppere>`/:term:`--uppere<--ue, --uppere>`

    ``-e, --est, --excess``
        turn off the find excess module -- this will automatically happen if :term:`numax`
        is provided **I think this has been deprecated**
         * dest = ``args.excess``
         * type = `bool`
         * default = `True`
         * action = ``store_false``
    
    ``--ew, --exwidth``
        the fractional value of the width to use surrounding the power excess, which is computed using a solar
        scaling relation (and then centered on the estimated :math:`\nu_{\mathrm{max}}`)
         * dest = ``args.width``
         * type = `float`
         * default = `1.0`
         * unit = fractional :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`--lp<--lp, --lowerp>`, :term:`--lowerp<--lp, --lowerp>`, :term:`--up<--up, --upperp>`, :term:`--upperp<--up, --upperp>`
           
    ``-f, --fix, --fixwn, --wn``
        fix the white noise level in the background fitting **NOT operational yet**
        this still needs to be tested
         * dest = ``args.fix``
         * type = `bool`
         * default = `False`
         * action = ``store_true``

    ``--file, --list, --todo``
        the path to the text file that contains the list of stars to process, which is convenient
        for running many stars
         * dest = ``args.file``
         * type = `str`
         * default = ``TODODIR``
         * **see also:** :term:`--star<--star, --stars>`, :term:`--stars<--star, --stars>`

    ``-g, --globe, --global``
        do not estimate the global asteroseismic parameter numax and dnu. **This is helpful for the
        application to cool dwarfs, where detecting solar-like oscillations is quite difficult
        but you'd still like to characterize the granulation components.**
         * dest = ``args.globe``
         * type = `bool`
         * default = `True`
         * action = ``store_false``

    ``--gap, --gaps``
        what constitutes a time series gap (i.e. how many cadences)
         * dest = ``args.gap``
         * type = `int`
         * default = `20`
         * **see also:** :term:`-x<-x, --stitch, --stitching>`, :term:`--stitch<-x, --stitch, --stitching>`, :term:`--stitching<-x, --stitch, --stitching>`

    ``-i, --ie, --interpech``
        turn on the bilinear interpolation 
        of the plotted echelle diagram
         * dest = ``args.interp_ech``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
         * **see also:** :term:`--se<--se, --smoothech>`, :term:`--smoothech<--se, --smoothech>`

    ``--in, --input, --inpdir``
        path to the input data
         * dest = ``args.inpdir``
         * type = `str`
         * default = ``INPDIR``

    ``--infdir``
        path to relevant `pySYD` information (defined in init file)
         * dest = ``args.infdir``
         * type = `str`
         * default = ``INFDIR``
         * **see also:** :term:`--file<--file, --list, --todo>`, :term:`--info<--info, --information>`, :term:`--information<--info, --information>`, :term:`--list<--file, --list, --todo>`, :term:`--todo<--file, --list, --todo>`

    ``--info, --information``
        path to the csv containing all the stellar information 
        (although *not* required)
         * dest = ``args.info``
         * type = `str`
         * default = ``star_info.csv``
    
    ``--iw, --indwidth``
        width of binning for the power spectrum used in the first module 
        **TODO: CHECK THIS** 
         * dest = ``args.ind_width``
         * type = `float`
         * default = `20.0`
         * unit = :math:`\mu \mathrm{Hz}`
    
    ``-k, --kc, --kepcorr``
        turn on the *Kepler* short-cadence artefact correction module. if you don't
        know what a *Kepler* short-cadence artefact is, chances are you shouldn't mess
        around with this option yet
         * dest = ``args.kepcorr``
         * type = ``bool``
         * default = `False`
         * action = ``store_true``
    
    ``--laws, --nlaws``
        force the number of red-noise component(s). **fun fact:** the older IDL version
        of ``SYD`` fixed this number to ``2`` for the *Kepler* legacy sample -- now we
        have made it customizable all the way down to an individual star!
         * dest = ``args.n_laws``
         * type = `int`
         * default = `None`

    ``--lb, --lowerb``
        the lower frequency limit of the power spectrum to use in the background-fitting
        routine. **Please note:** unless :math:`\nu_{\mathrm{max}}` is known, it is highly 
        recommended that you do *not* fix this beforehand
         * dest = ``args.lower_bg``
         * type = `float`
         * nargs = `'*'`
         * default = `1.0`
         * unit = :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`--ub<--ub, --upperb>`, :term:`--upperb<--ub, --upperb>`
         
    ``--le, --lowere``
        the lower frequency limit of the folded power spectrum to "whiten" mixed modes before
        estimating the final value for dnu 
         * dest = ``args.lower_ech``
         * type = `float`
         * nargs = `'*'`  
         * default = `None`
         * unit = :math:`\mu \mathrm{Hz}`
         * **REQUIRES:** :term:`--ue<--ue, --uppere>`/:term:`--uppere<--ue, --uppere>` *and* :term:`--dnu`
         
    ``--lp, --lowerp``
        to change the lower frequency limit of the zoomed in power spectrum (i.e. the region with the supposed
        power excess due to oscillations). Similar to :term:`--ew` but instead of a fractional value w.r.t. the 
        scaled solar value, you can provide hard boundaries in this case **TODO** check if it requires and upper
        bound -- pretty sure it doesn't but should check 
         * dest = ``args.lower_ps``
         * type = `float`
         * nargs = `'*'`
         * default = `None`
         * unit = :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`--up<--up, --upperp>`, :term:`--upperp<--up, --upperp>`
         
    ``--lx, --lowerx``
        the lower limit of the power spectrum 
        to use in the first module (to estimate numax)
         * dest = ``args.lower_ex``
         * type = `float`
         * default = `1.0`
         * unit = :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`--ux<--ux, --upperx>`, :term:`--upperx<--ux, --upperx>`
         
    ``-m, --samples``
        option to save the samples from the Monte-Carlo sampling (i.e. parameter 
        posteriors) in case you'd like to reproduce your own plots, etc.
         * dest = ``args.samples``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--mc, --iter, --mciter``
        number of Monte-Carlo-like iterations. This is `1` by default, since you should
        always check the data and output figures before running the sampling algorithm.
        But for purposes of generating uncertainties, `n=200` is typically sufficient.
         * dest = ``args.mc_iter``
         * type = `int`
         * default = `1`

    ``--method``
        development option to change the method used to determine dnu, choices are
        ~[`'M'`,`'A'`,`'D'`] for Maryum, Ashley and Dennis (respectively)
         * dest = ``args.method``
         * type = `str`
         * default = `'D'`

    ``--metric``
        which model metric to use for the best-fit background model, current choices are
        ~[`'bic'`, `'aic'`] but **still being developed and tested**
         * dest = ``args.metric``
         * type = `str`
         * default = `'aic'`
    
    ``-n, --notch``
        use notching technique to reduce effects from mixes modes (pretty sure this is not
        full functional yet, creates weird effects for higher SNR cases)
         * dest = ``args.notching``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--nox, --nacross``
        specifies the number of bins (i.e. the resolution) to use for the x-axis of the
        echelle diagram -- fixing this number if complicated because it depends on both the
        resolution of the power spectrum as well as the characteristic frequency separation.
        This is another example where, if you don't know what this means, you probably should
        not change it.
         * dest = ``args.nox``
         * type = `int`
         * default = `50`
         * **see also:** :term:`--noy<--noy, --ndown, --norders>`, :term:`--ndown<--noy, --ndown, --norders>`, :term:`--norders<--noy, --ndown, --norders>`
    
    ``--noy, --ndown, --norders``
        similar to :term:`nox`, this specifies the number of bins (i.e. orders) to use on the
        y-axis of the echelle diagram
         * dest = ``args.noy``
         * type = `int`
         * default = `0`
         * **see also:** :term:`--nox<--nox, --nacross>`, :term:`--nacross<--nox, --nacross>`
    
    ``--nt, --nthread, --nthreads``
        the number of processes to run in parallel. If nothing is provided when you run in ``pysyd.parallel``
        mode, the software will use the ``multiprocessing`` package to determine the number of CPUs on the
        operating system and then adjust accordingly. **In short:** this probably does not need to be changed
         * dest = ``args.n_threads``
         * type = `int`
         * default = `0`
         
    ``--numax``
        brute force method to bypass the first module and provide 
        an initial starting value for :math:`\rm \nu_{max}`
        ``Asserts len(args.numax) == len(args.targets)``
        * dest = ``args.numax``
        * type = `float`
        * nargs = `'*'`
        * default = `None`
        * unit = :math:`\mu \mathrm{Hz}`
    
    ``-o, --overwrite``
        newer option to overwrite existing files with the same name/path since it will now add extensions
        with numbers to avoid overwriting these files
         * dest = ``args.overwrite``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--of, --over, --oversample``
        the oversampling factor of the provided power spectrum. Default is `0`, which means it is calculated from
        the time series data. **Note:** this needs to be provided if there is no time series data!
         * dest = ``args.of_actual``
         * type = `int`
         * default = `None`
         
    ``--out, --output, --outdir``
        path to save 
        results to
         * dest = ``args.outdir``
         * type = `str`
         * default = `'OUTDIR'`
    
    ``-p, --par, --parallel``
        run ``pySYD`` in 
        parallel mode
         * dest = ``args.parallel``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
    
    ``--peak, --peaks, --npeaks``
        the number of peaks to identify 
        in the autocorrelation function
         * dest = ``args.n_peaks``
         * type = `int`
         * default = `5`
    
    ``--rms, --nrms``
        the number of points used to estimate the amplitudes of individual background (red-noise) components
        *Note: this should only rarely need to be touched*
         * dest = ``args.n_rms``
         * type = `int`
         * default = `20`
    
    ``-s, --save``
        turn off the automatic saving 
        of output figures and files
         * dest = ``args.save``
         * type = `bool`
         * default = `True`
         * action = ``store_false``

    ``--se, --smoothech``
        option to smooth the echelle diagram output 
        using a box filter of this width
         * dest = ``args.smooth_ech``
         * type = `float`
         * default = `None`
         * unit = :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`-e<-e, --ie, --interpech>`, :term:`--ie<-e, --ie, --interpech>`, :term:`--interpech<-e, --ie, --interpech>`

    ``--sm, --smpar``
        the value of the smoothing parameter to estimate the smoothed numax (that is really confusing)
        **note:** typical values range from `1`-`4` but this is fixed based on years of trial & error
         * dest = ``args.sm_par``
         * type = `float`
         * default = `None`
         * unit = fractional :math:`\mu \mathrm{Hz}`

    ``--sp, --smoothps``
        the box filter width used for smoothing of the power spectrum. The default is `2.5` but will switch to
        `0.5` for more evolved stars (if :math:`\rm \nu_{max}` < 500 :math:`\mu \mathrm{Hz}`)
         * dest = ``args.smooth_ps``
         * type = `float`
         * default = `2.5`
         * unit = :math:`\mu \mathrm{Hz}`

    ``--star, --stars``
        list of stars to process. Default is `None`, which will read 
        in the star list from ``args.file`` instead
         * dest = ``args.star``
         * type = `str`
         * nargs = `'*'`
         * default = `None`
         * **see also:** :term:`--file<--file, --list, --todo>`, :term:`--list<--file, --list, --todo>`, :term:`--todo<--file, --list, --todo>`

    ``--step, --steps``
        the step width for the collapsed autocorrelation function w.r.t. the fraction of the
        boxsize. **Please note:** this should not be adjusted
         * dest = ``args.step``
         * type = `float`
         * default = `0.25`
         * unit = fractional :math:`\mu \mathrm{Hz}`

    ``--sw, --smoothwidth``
        the width of the box filter that is 
        used to smooth the power spectrum
         * dest = ``args.smooth_width``
         * type = `float`
         * default = `20.0`
         * unit = :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`--sp<--sp, --smoothps>`, :term:`--smoothps<--sp, --smoothps>`


.. warning::

    All parameters are optimized for most star types but some may need adjusting. 
    An example is the smoothing width (``--sw``), which is 20 muHz by default, but 
    may need to be adjusted based on the nyquist frequency and frequency resolution 
    of the input power spectrum.


.. glossary::

    ``--thresh, --threshold``
        the fractional value of the autocorrelation function's full width at half
        maximum (which is important in this scenario because it is used to determine :math:`\Delta\nu`)
         * dest = ``args.threshold``
         * type = `float`
         * default = `1.0`
         * unit = fractional :math:`\mu \mathrm{Hz}`
    
    ``--trials, --ntrials``
        the number of trials used to estimate numax in the first module -- can be bypassed if :term:`--numax`
        is provided.
         * dest = ``args.n_trials``
         * type = `int`
         * default = `3`

    ``--ub, --upperb``
        the upper limit of the power spectrum used in the background-fitting module **Please note:** 
        unless :math:`\nu_{\mathrm{max}}` is known, it is highly recommended that you do *not* fix this beforehand
         * dest = ``args.upper_bg``
         * type = `float`
         * nargs = `'*'`
         * default = `6000.0`
         * unit = :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`--lb<--lb, --lowerb>`, :term:`--lowerb<--lb, --lowerb>`

    ``--ue, --uppere``
        the upper frequency limit of the folded power spectrum used to "whiten" mixed modes before determining
        the correct :math:`\Delta\nu`
         * dest = ``args.upper_ech``
         * type = `float`
         * nargs = `'*'`
         * default = `None`
         * unit = :math:`\mu \mathrm{Hz}`
         * **REQUIRES:** :term:`--le<--le, --lowere>`/:term:`--lowere<--le, --lowere>` *and* :term:`--dnu`

    ``--up, --upperp``
        the upper frequency limit used for the zoomed in power spectrum. In other words, this is an option to
        use a different upper bound than the one determined automatically
         * dest = ``args.upper_ps``
         * type = `float`
         * nargs = `'*'`
         * default = `None`
         * unit = :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`--lp<--lp, --lowerp>`, :term:`--lowerp<--lp, --lowerp>`

    ``--ux, --upperx``
        the upper frequency limit of the power 
        spectrum to use in the first module
         * dest = ``args.upper_ex``
         * type = `float`
         * default = `6000.0`
         * unit = :math:`\mu \mathrm{Hz}`
         * **see also:** :term:`--lx<--lx, --lowerx>`, :term:`--lowerx<--lx, --lowerx>`
    
    ``-v, --verbose``
        turn on the verbose output (also not recommended when running many stars, and
        definitely *not* when in parallel mode) **Check** this but I think it will be
        disabled automatically if the parallel mode is `True`
         * dest = ``args.verbose``
         * type = `bool`
         * default = `False`
         * action = ``store_true``

    ``-x, --stitch, --stitching``
        correct for large gaps in time series data by 'stitching' the light curve
         * dest = ``args.stitch``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
         * **see also:** :term:`--gap<--gap, --gaps>`, :term:`--gaps<--gap, --gaps>`
    
    ``-y, --hey``
        plugin for Daniel Hey's interactive echelle 
        package **but is not currently implemented**
        **TODO**
         * dest = ``args.hey``
         * type = `bool`
         * default = `False`
         * action = ``store_true``
