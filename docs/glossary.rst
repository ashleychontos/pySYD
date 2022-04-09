.. _glossary:

*******************************
Glossary of documentation terms
*******************************

.. glossary::

    AIC
    Akaike Information Criterion
        a common metric for model selection that prevents overfitting of data by penalizing
        models with higher numbers of parameters (:math:`k`)
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

    B(\nu) = W + \sum_{i=0}^{n} \frac{4\sigma_{i}^{2}\tau_{i}}{1 + (2\pi\nu\tau_{i})^{2} + (2\pi\nu\tau_{i})^{4}}


.. glossary::

    BCPS
    background-corrected power spectrum
        the power spectrum after removing the best-fit stellar background model. In general, this step
        removes any slopes in power spectra due to correlated red-noise properties


.. note::

    A :term:`background-corrected power spectrum` (:term:`BCPS`) is an umbrella term that has the same 
    meanings as a :term:`background-divided power spectrum` (:term:`BDPS`) *and* a 
    :term:`background-subtracted power spectrum` (:term:`BSPS`). Thus it is best ***to avoid*** this
    phrase if at all possible since it does not specify how the power spectrum has been modified.


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


    cadence
        the median absolute difference between consecutive time series observations
         * **variable:** :math:`\Delta t`
         * **units:** :math:`\rm s`
         * **definition:**

    critically-sampled power spectrum
        when the frequency resolution of the power spectrum is exactly equal to the inverse of
        the total duration of the time series data it was calculated from
        
    ED
    echelle diagram
        a diagnostic tool to confirm that :term:`dnu` is correct. This is done by folding the power spectrum (:term:`FPS`)
        using :term:`dnu` (you can think of it as the PS modulo the spacing) -- which if the :term:`large frequency separation`
        is correct -- the different oscillation modes will form straight ridges. **Fun fact:** the word 'echelle'
        is actually French for ladder
        
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
        the resolution of a :term:`power spectrum` is set by the total length of 
        the time series :math:`(\Delta T^{-1})`
        
    FWHM
    full-width half maximum
        for a Gaussian-like distribution, the full-width at half maximum (or full-width half max) is
        approximately equal to :math:`\pm 1\sigma`

    global properties
        in asteroseismology, the global asteroseismic parameters or properties refer to :math:`\nu_{\mathrm{max}}` 
        (:term:`numax`) and :math:`\Delta\nu` (:term:`dnu`) 
        
    granulation
        the smallest (i.e. quickest) scale of convective processes
        
    Harvey-like component
    Harvey-like model
        named after the person who first person who discovered the relation -- and found it did a good 
        job characterizing granulation amplitudes and time scales in the Sun
        
    *Kepler* artefact
        *Kepler* short-cadence artefact in the power spectrum from a short-cadence light curve 
        occurring at the nyquist frequency for long-cadence (i.e. ~270muHz)

    *Kepler* legacy sample
        a sample of well-studied *Kepler* stars exhibiting solar-like oscillations (cite Lund+2014)
        
    dnu
    large frequency separation
        the so-called large frequency separation is the inverse of twice the sound travel time between
        the center of the star and the surface. Even more generally, this is the comb pattern or regular 
        spacing observed for solar-like oscillations. It is exactly equal to the frequency spacing between 
        modes with the same :term:`spherical degree` and consecutive :term:`radial order`s.
         * **variable:** :math:`\Delta\nu`
         * **units:** :math:`\rm \mu Hz`
         * **definition:**
  
.. math::
 
    \Delta\nu = \bigg[2 \int_{0}^{R} \frac{\mathrm{d}r}{c}\bigg]^{-1} \propto \bar{\rho}
 
.. glossary::
        
    light curve
        the measure of an object's brightness with time
        
    mesogranulation
        the intermediate scale of convection
        
    mixed modes
        in special circumstances, pressure (or p-) modes couple with gravity (or g-) modes and make 
        the spectrum of a solar-like oscillator much more difficult to interpret -- in particular,
        for measuring the :term:`large frequency separation`
    
    notching
        a process used to mitigate features in the frequency domain (e.g., mixed modes) by setting
        specific values to the minimum power in the array
        
    nyquist frequency
        the highest frequency that can be sampled, which is set by the :term:`cadence` of observations 
        (:math:`\Delta t`) 
         * **variable:** :math:`\rm \nu_{nyq}`
         * **units:** :math:`\rm \mu Hz`
         * **definition:**
   
.. math::

    \mathrm{\nu_{nyq}} = \frac{1}{2 \Delta t} 


.. note:: *Kepler* example

    *Kepler* short-cadence data has a cadence, :math:`\Delta t \sim 60 \mathrm{s}`. Therefore,
    the nyquist frequency for short-cadence *Kepler* data is:

    .. math::

         \mathrm{\nu_{nyq}} = \frac{1}{2\cdot60\,\mathrm{s}} \times \frac{10^{6}\,\mu\mathrm{Hz}}{1\,\mathrm{Hz}} \approx 8333 \,\mu\mathrm{Hz}


.. glossary::
    
    oversampled power spectrum
        if the resolution of the power spectrum is greater than 1/T

    p-mode oscillations
    solar-like oscillations
        implied in the name, these oscillations are driven by the same mechanism as that observed in the Sun, which is
        due to turbulent, near-surface convection. They are also sometimes referred to as **p-mode oscillations**, after the
        pressure-driven (or acoustic sound) waves that are resonating in the stellar cavity.

    
.. glossary::

    power excess
        the region in the power spectrum believed to show solar-like oscillations is typically characterized by a
        Gaussian-like envelope of oscillations, :math:`G(\nu)`

.. math::

    G(\nu) = A_{\mathrm{osc}} \,\mathrm{exp} \bigg[ - \frac{(\nu-\nu_{\mathrm{max}})^{2}}{2\sigma_{\mathrm{osc}}^{2}} \bigg] 


.. glossary::
    
         * **variables:**
           * :math:`A_{\mathrm{osc}}`: amplitude at frequency of maximum power
           * :math:`\nu_{\mathrm{max}}`: center of the Gaussian-like envelope
           * :math:`\rm \sigma_{osc}`: width of Gaussian

    PSD
    power spectral density
        when the power of a frequency spectrum is normalized s.t. it satisfies Parseval's theorem (which is just a fancy way of 
        saying that the fourier transform is unitary)
         * **unit:** :math:`\rm ppm^{2} \,\, \mu Hz^{-1}`
    
    PS
    power spectrum
        any object that varies in time also has a corresponding frequency (or power) spectrum, which is computed by taking 
        the :term:`fast fourier transform` of the :term:`light curve`. A general model to describe characteristics of a power spectrum is generalized
        by the equation below, where :math:`W` is a constant (frequency-independent) noise term, primarily due to photon noise. :math:`B` and :math:`G`
        correspond to the background and Gaussian-like power excess components, respectively. Finally, :math:`R` corresponds to
        the response function, or the attenuation of signals due to time-averaged observations.

.. math::

    P(\nu) = W + R(\nu) [B(\nu) + G(\nu)]


.. glossary::
        
    scaling relations
        empirical relations for fundamental stellar properties that are scaled with respect to the Sun, since it is the star 
        we know best. In asteroseismology, the most common relations combine :term:`global asteroseismic parameters<global properties>`
        with spectroscopic effective temperatures to derive stellar masses and radii:
        
.. math::

    \frac{R_{\star}}{R_{\odot}} = \bigg( \frac{\nu_{\mathrm{max}}}{\nu_{\mathrm{max,\odot}}} \bigg) \bigg( \frac{\Delta\nu}{\Delta\nu_{\odot}} \bigg)^{-2} \bigg( \frac{T_{\mathrm{eff}}}{T_{\mathrm{eff,\odot}}} \bigg)^{1/2}
    
.. math::

    \frac{M_{\star}}{M_{\odot}} = \bigg( \frac{\nu_{\mathrm{max}}}{\nu_{\mathrm{max,\odot}}} \bigg)^{3} \bigg( \frac{\Delta\nu}{\Delta\nu_{\odot}} \bigg)^{-4} \bigg( \frac{T_{\mathrm{eff}}}{T_{\mathrm{eff,\odot}}} \bigg)^{3/2}
    
.. glossary::

    whiten
    whitening
        a process to remove undesired artefacts or effects present in a frequency spectrum by taking that frequency region 
        and replacing it with simulated white noise. This is typically done for subiants with :term:`mixed modes` in order 
        to better estimate :term:`dnu`. This can also help mitigate the short-cadence :term:`Kepler artefact`.
