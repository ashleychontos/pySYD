*******************************
Glossary of documentation terms
*******************************

.. glossary::

    AIC
    Akaike Information Criterion
        a common metric for model selection that prevents overfitting of data by penalizing
        models with higher numbers of paramters (:math:`k`)
         * **definition:**
        
.. math::

    \mathrm{AIC} = 2k - 2\mathrm{ln}(\hat{L})
    
.. glossary::
    
    asteroseismology
        the study of oscillations in stars
    
    ACF
    autocorrelation function
        in this context it is a small range of frequencies in the power spectrum surrounding the 
        solar-like oscillations, then the power array is correlated (or convolved) with a copy of
        the power array. This is a helpful diagnostic tool for quantitatively confirming the 
        p-mode oscillations, since they have regular spacings in the frequency domain and therefore
        should create strong peaks at integer and half integer harmonics of :math:`\Delta\nu`
    
    background
        this basically means any other noise structures present in the power spectrum that are *not* 
        due to solar-like oscillations. This is traditionally parametrized as:


.. math::

    P(\nu) = W + \frac{}{}


.. glossary::

    BCPS
    background-corrected power spectrum
        the power spectrum after removing the best-fit stellar background model. In general, this step
        removes any slopes in power spectra due to correlated red-noise properties


.. note::

    A :term:`background-corrected power spectrum` (:term:`BCPS`) is an umbrella term that has the same 
    meanings as a :term:`background-divided power spectrum` (:term:`BDPS`) *and* a 
    :term:`background-subtracted power spectrum` (:term:`BSPS`) **but** is good to avoid when possible 
    since it does not specify how the power spectrum has been corrected.


.. glossary::

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
        a common metric for model selection
    
    critically-sampled power spectrum
        sfdklja
        
    ED
    echelle diagram
        a diagnostic tool to confirm that :term:`dnu` is correct. This is done by folding the power spectrum (:term:`FPS`)
        using :term:`dnu` (you can think of it as the PS modulo the spacing), and if the :term:`large frequency separation`
        is correct, the different oscillation modes will form straight ridges. **Fun fact:** the word 'echelle' is French 
        for ladder
        
    FFT
    fast fourier transform
        a method used in signal analysis to determine the most dominant periodicities present in a :term:`light curve`
    
    FPS
    folded power spectrum
        the power spectrum folded (or stacked) at some frequency, which is typically done with the :term:`large frequency separation`
        to construct an :term:`echelle diagram`

    numax
    frequency of maximum power
        the frequency corresponding to maximum power, which is roughly the center of the Gaussian-like envelope of oscillations
         * **variable:** :math:`\nu_{\mathrm{max}}`
         * **units:** :math:`\rm \mu Hz`
    
        scales with evolutionary state, logg, acoustic cutoff
        
    frequency resolution
        the resolution of a :term:`power spectrum` is set by the total length (i.e. time) of the time series
        
    FWHM
    full-width half maximum
        for a Gaussian-like distribution, the full-width at half maximum (or full-width half max) is
        approximately equal to :math:`\pm 1\sigma`

    global properties
        in asteroseismology, the global asteroseismic parameters or properties refer to :math:`\nu_{\mathrm{max}}` 
        (:term:`numax`) and :math:`\Delta\nu` (:term:`dnu`) 
        
    granulation
        the smallest (i.e. quickest) scale of convective processes
        
    Harvey component
    Harvey model
        named after the person who first person who discovered the relation -- and found it did a good 
        job characterizing granulation amplitudes and time scales in the Sun
        
    *Kepler* artefact
        *Kepler* short-cadence artefact in the power spectrum from a short-cadence light curve 
        occurring at the nyquist frequency for long-cadence (i.e. ~270muHz)

    *Kepler* legacy sample
        a sample of well-studied *Kepler* stars exhibiting solar-like oscillations (cite Lund+2014)
        
    dnu
    large frequency separation
        generally this is the comb pattern or regular spacing observed for solar-like oscillations.
        It is exactly equal to the frequency spacing between modes with the same :term:`spherical degree` 
        and consecutive :term:`radial order`s.
         * **variable:** :math:`\Delta\nu`
         * **units:** :math:`\rm \mu Hz`
         * **definition:**
    
        scales with mean stellar density
        
    light curve
        the measure of an object's brightness with time
        
    mesogranulation
        the intermediate scale of convection
        
    mixed modes
        in special circumstances, pressure (or p-) modes couple with gravity (or g-) modes and make 
        the spectrum of a solar-like oscillator much more difficult to interpret -- in particular,
        for measuring the :term:`large frequency separation`
    
    notching
        a process
        
    nyquist frequency
        the highest frequency that can be sampled, which is set by the cadence of (or time between) 
        observations
         * **variable:** :math:`\rm \nu_{nyq}`
         * **units:** :math:`\rm \mu Hz`
         * **definition:**
         
.. math::

    \mathrm{\nu_{nyq}} = \frac{1}{2*\mathrm{cadence}}
    
.. glossary::
    
    oversampled power spectrum
        if the resolution of the power spectrum is greater than 1/T

    p-mode oscillations
    solar-like oscillations
        implied in the name, these oscillations are driven by the same mechanism as that observed in the Sun, which is
        due to turbulent, near-surface convection. They are also sometimes referred to as **p-mode oscillations**, after the
        pressure-driven (or acoustic sound) waves that are resonating in the stellar cavity.
    
    PSD
    power spectral density
        when the power of a frequency spectrum is normalized s.t. it satisfies Parseval's theorem (which is just a fancy way of 
        saying that the fourier transform is unitary)
         * **unit:** :math:`\rm ppm^{2} \,\, \mu Hz^{-1}`
    
    PS
    power spectrum
        any object that varies in time also has a corresponding frequency (or power) spectrum, which here, is computed by taking 
        the fourier transform of the :term:`light curve`.
        
    radial order
        in asteroseismology, the radial order (:math:`n`) is the number of nodes from the surface to the center of the
        star. For solar-like oscillators, modes are typically characterized by higher radial orders and low spherical
        degree. By definition, modes of the same spherical degree and consecutive radial orders are separated by :term:`dnu`.
         * **variable:** :math:`n`
        
    scaling relations
        these empirical relations are typically scaled with respect to the Sun, since it is the star we know best. These
        are used in many aspects of asteroseismology, but the most common use is to derive fundamental stellar parameters
        mass and radius given the effective temperature of the star and its :term:`global properties`
        
    spherical degree
        in asteroseismology, the spherical degree (:math:`\ell`) is the number of oscillation modes on the surface of
        the star. For unresolved asteroseismology, this is typically very low order degrees and has only been possible
        up to a spherical degree of :math:`\ell = 3`
         * **variable:** :math:`\ell`

    ``SYD``
        the well-known IDL-based asteroseismic pipeline created by Dan Huber during his PhD in Sydney (hence SYD). ``SYD``
        has been extensively tested and benchmarked to other closed-source asteroseismic tools on *Kepler* stars.
        
    whitening
        a process to remove undesired artefacts or effects present in a power spectrum by taking that frequency region 
        and replacing it with white noise. This is typically done for subiants with :term:`mixed modes` in order to better 
        estimate :term:`dnu`. This can also help mitigate the short-cadence :term:`Kepler artefact`.
