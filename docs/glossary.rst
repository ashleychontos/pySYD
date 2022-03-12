*******************************
Glossary of documentation terms
*******************************

.. glossary::

    AIC
    Akikdaf Information Criterion
        afjdkajfl
    
    artefact
        *Kepler* short-cadence artefact in the power spectrum from a short-cadence light curve 
        occurring at the nyquist frequency for long-cadence (i.e. ~270muHz)
    
    asteroseismology
        the study of oscillations in stars
    
    ACF
    autocorrelation function
        ajdkfaj;
    
    background
        this basically means any other noise structures present in the power spectrum that are *not* 
        due to the solar-like oscillations. This is traditionally parametrized as:
        
.. math::

    P(\nu) = W + \frac{}{}
    
    
    BCPS
    background-corrected power spectrum
        the power spectrum after removing the best-fit stellar background model. In general, this step
        removes any slopes in power spectra due to correlated red-noise properties

.. note::

    A :term:`background-corrected power spectrum` (:term:`BCPS`) is an umbrella term that has the same
    meanings as a :term:`background-divided power spectrum` (:term:`BDPS`) *and* a
    :term:`background-subtracted power spectrum` (:term:`BSPS`) **but** is good to avoid when possible
    since it does not specify how the power spectrum has been corrected.

    
    BDPS
    background-divided power spectrum
        the power spectrum divided by the best-fit stellar background model. Using this method for data 
        analysis is great for first detecting and identifying any solar-like oscillations since it will
        make the power excess due to stellar oscillations appear higher signal-to-noise
    
    BSPS
    background-subtracted power spectrum
        the best-fit stellar background model is subtracted from the power spectrum. While this method
        appears to give a lower signal-to-noise detection, the amplitudes measured through this analysis
        are physically-motivated and correct (i.e. can be compared with other literature values)
    
    BIC
    Bayesian Information Criterion
        ldkfjaldkjl
    
    critically sampled
        sfdklja

    dnu
    large frequency separation
        the comb patter or regular spacing observed for solar-like oscillations for different modes
         * **variable:** :math:`\Delta\nu`
    
        scales with mean stellar density
    
    FPS
    folded power spectrum
        ldkfjaldkjfa -> used for echelle diagrams and whitening

    numax
    frequency of maximum power
        the frequency corresponding to maximum power, which is roughly the center of the Gaussian-like envelope of oscillations
         * **variable:** :math:`\nu_{\mathrm{max}}`
    
        scales with evolutionary state, logg, acoustic cutoff
        
    FWHM
    full-width half maximum
        kdjfladk

    global properties
        the term 'global' is used to describe the general properties of the observed oscillations and is not associated with
        the detailed frequency analysis of individual oscillation modes, a process referred to as peakbagging. Traditionally
        the two main global asteroseismic properties are :math:`\nu_{\mathrm{max}}` and :math:`\Delta\nu`, both of which are 
        described in more detail in their respective entry.
    
        there are two distinct features of solar-like oscillations that enable the measurement of the two main global 
        properties, numax and dnu. The stochastic nature of convection leads to oscillation modes over a range of frequencies, 
        where the envelope of the observed modes is approximately Gaussian and the frequency corresponding to the middle of
        peak of this Gaussian-like envelope is referred to as numax. The second feature is the comb pattern or regular spacing
        between different modes, which is referred to as the characteristic frequency spacing or dnu. Therefore, the term 'global' 
        is used to describe the general properties of the oscillations, like the center and amplitude of the Gaussian-like envelope. The second 
        distinct feature is the comb pattern or regular spacing between different modes, which is
        referred to as the characteristic frequency spacing or dnu. For purposes of our analyses, global asteroseismic 
        parameters regular spacing or combP-mode oscillations
        In addition 
        to the center of the frequency range (numax), there is a regular spacing or comb pattern between the observed modes 
        that is referred to as the characteristic spacing or dnu.
        
    granulation background
        dlfakjdlakjafld


    *Kepler* legacy sample
        a sample of well-studied *Kepler* stars exhibiting solar-like oscillations (cite)
        
    mesogranulation
        dkjfaldjal
        
    mixed modes
        ldfjadkjf -> what you need to whiten
    
    notching
        a process
        
    nyquist frequency
        the highest frequency that can be sampled, which is set by the cadence of (or time between) 
        observations (1/2*cadence)
         * **variable:** :math:`\rm \nu_{nyq}`
        
    order
        kldjfladkjad
    
    
    oversampling
        ldkjfaljadlak

    p-mode oscillations
    solar-like oscillations
        implied in the name, these oscillations are driven by the same mechanism as that observed in the Sun, which is
        due to turbulent, near-surface convection. They are also sometimes referred to as **p-mode oscillations**, after the
        pressure-driven (or acoustic sound) waves that are resonating in the stellar cavity.
    
    PSD
    power spectral density
        ldjkfalkdajfal :math:`\rm ppm^{2} \,\, \mu Hz^{-1}`
    
    PS
    power spectrum
        dlfajk;adj
        
    resolution
        dkljflajd set by the total length (i.e. time) of the time series 

    ``SYD``
        the well-known IDL-based asteroseismic pipeline created by Dan Huber during his PhD in Sydney (hence SYD). ``SYD``
        has been extensively tested and benchmarked to other closed-source asteroseismic tools on *Kepler* stars.
        
    whitening
        kjdfla;jdlak
