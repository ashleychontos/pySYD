import os
import glob
import numpy as np
import pandas as pd
from astropy.stats import mad_std
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle as lomb
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft


from pysyd import models
from pysyd import utils
from pysyd import plots




class Target:

    """

    Main pipeline target object

    """

    def __init__(self, name, args):
        """
        
        A new instance (or star) is created for each target that is processed.
        Initialization copies the relevant star dictionary and then will try to
        load in data for the given star name.

        Deprecated
            Before it used to check for data and then check the Target.ok attribute
            before processing the star but now it will raise an exception (InputError)
            and therefore, there is no need to check the attribute anymore. This can
            be updated.

        Attributes
            ok : bool
                is the star 'ok' to proceed (i.e. does it pass the initial checks)

        Parameters
            name : str
                which target to load in and/or process
            args : utilities.Parameters
                container class of pysyd parameters


        """
        self.name = name
        self.constants = args.constants
        if name in args.params:
            # load star-specific information
            self.params = args.params[name]
        else:
            # if not available, loads defaults
            args.add_stars(stars=[name])
            self.params = args.params[name]
        self.load_star()


    def __repr__(self):
        return "<Star Object {}>".format(self.name)
           

    def process_star(self,):
        """

        Run the ``pySYD`` pipeline on a given star

        Parameters
            results : Dict[]
                dictionary containing the results from the pipeline, with keys corresponding
                to each of the steps (i.e. 'estimates', 'parameters', 'samples')
            plotting : Dict
                same as the results, which saves all the information needed for plotting at 
                the end

        """
        self.params['results'], self.params['plotting'] = {}, {}
        self.estimate_parameters()
        self.derive_parameters()
        self.show_results()


    def show_results(self, show=False, verbose=False,):
        """

        Parameters
            show : bool, optional
                show output figures and text
            verbose : bool, optional
                turn on verbose output

        """
        # Print results
        if self.params['verbose']:
            utils.verbose_output(self)
            if self.params['show']:
                print(' - displaying figures')
        else:
            print('')
        plots.make_plots(self)
        if self.params['cli']:
            input(' - press RETURN to exit')


#########################################################################################
#
# Three main higher-level function calls
#  1. load in data for a given star
#  2. use the data to estimate initial starting points
#  3. use the estimates to derive the full suite of parameters
#
# 

    def load_star(self, ps=False, lc=False, note='',):
        """
        Load data for single star

        Loads all data in for a single star. This is done by first checking for the power
        spectrum and then the time series data, which will compute a power spectrum in the
        event that there is not one.

        Attributes
            note : str, optional
                verbose output
            lc : bool
                `True` if object has light curve
            time : numpy.ndarray
                time array in days
            flux : numpy.ndarray
                relative or normalized flux array
            ps : bool
                `True` if object has power spectrum
            frequency : numpy.ndarray
                copy of original frequency array
            power : numpy.ndarray
                copy of original power spectrum
            freq_os : numpy.ndarray
                final version of oversampled frequency array
            pow_os : numpy.ndarray
                final version of oversampled power spectrum
            freq_cs : numpy.ndarray
                final version of critically-sampled frequency array
            pow_cs : numpy.ndarray
                final version of critically-sampled power spectrum
            cadence : int
                median cadence of time series data (:math:`\Delta t`)
            nyquist : float
                nyquist frequency of the power spectrum
            baseline : float
                total time series duration (:math:`\Delta T`)
            tau_upper : float
                upper limit of the modeled time scales set by baseline/2
            

        """
        self.ps, self.lc, self.ok, self.note = False, False, False, ''
        # Now done at beginning to make sure it only does this once per star
        if glob.glob(os.path.join(self.params['inpdir'],'%s*' % str(self.name))):
            if self.params['verbose']:
                print('\n-----------------------------------------------------------\nTarget: %s\n-----------------------------------------------------------' % str(self.name))
            # Load PS first in case we need to calculate PS from LC
            self.load_power_spectrum()
            # Load light curve
            self.load_time_series()
        # CASE 4: NO LIGHT CURVE AND NO POWER SPECTRUM
        #     ->  cannot process, return user error
        if not self.ps:
            raise InputError("ERROR: no data found for target %s"%self.name)
        self.get_warnings() 


    def estimate_parameters(self, excess=True,):
        """
        Estimate :math:`\\rm \\nu_{max}`

        Estimates the initial starting values for parameters before performing the global
        fit. First it quantifies a crude background model by binning the power spectrum in 
        two steps (in both log and linear space). It will divide this contribution out and
        then use a collapsed autocorrelation technique to identify the power excess due to
        solar-like oscillations.

        Parameters
            excess : bool, optional
                if numax is already known, this can be changed to `False`


        """
        if 'results' not in self.params:
            self.params['results'] = {}
        if 'plotting' not in self.params:
            self.params['plotting'] = {}
        if self.params['excess']:
            # get initial values and fix data
            self.initial_estimates()
            # execute function
            self.estimate_numax()


    def derive_parameters(self, background=True, globe=True, mc_iter=1, converge=False,):
        """
        Global fit

        Estimates the initial starting values for parameters before performing the global
        fit. First it quantifies a crude background model by binning the power spectrum in 
        two steps (in both log and linear space). It will divide this contribution out and
        then use a collapsed autocorrelation technique to identify the power excess due to
        solar-like oscillations.

        Parameters
            background : bool, optional
                if numax is already known, this can be changed to `False`
            globe : bool, optional
                if numax is already known, this can be changed to `False`
            converge : bool
                will return `True` if the background-fitting routine converged on a sensible result
            mc_iter : int
                the number of iterations to run


        """
        if 'results' not in self.params:
            self.params['results'] = {}
        if 'plotting' not in self.params:
            self.params['plotting'] = {}
        # make sure there is an estimate for numax
        self.check_numax()
        # get initial values and fix data
        self.initial_parameters()
        self.first_step()
        # if the first step is ok, carry on
        if self.params['mc_iter'] > 1:
            self.get_samples()
        # Save results
        utils.save_parameters(self)


#########################################################################################
#
# Loading data
#

    def load_power_spectrum(self):
        """
        Load power spectrum
    
        Loads in the power spectrum data in for a given star,
        which will return `False` if unsuccessful and therefore, not run the rest
        of the pipeline

        Parameters
            oversampling_factor : int, optional
                oversampling factor of input power spectrum


        """
        # Try loading the power spectrum
        if os.path.exists(os.path.join(self.params['inpdir'], '%s_PS.txt' % str(self.name))):
            self.ps = True
            self.frequency, self.power = self.load_file(os.path.join(self.params['inpdir'], '%s_PS.txt' % str(self.name)))
            self.note += '# POWER SPECTRUM: %d lines of data read\n'%len(self.frequency)
            # Only use provided oversampling factor if there is no light curve to calculate it from 
            # CASE 1: POWER SPECTRUM AND NO LIGHT CURVE
            #     ->  assume critically-sampled power spectrum
            if not os.path.exists(os.path.join(self.params['inpdir'], '%s_LC.txt' % str(self.name))):
                if self.params['oversampling_factor'] is None:
                    self.note += '# WARNING: using PS with no additional information\n# **assuming critically-sampled PS**\n'
                    if self.params['mc_iter'] > 1:
                        self.note += '# **uncertainties may not be reliable if the PS is not critically-sampled**\n'
                    self.params['oversampling_factor'] = 1
                self.frequency, self.power = self.fix_data(self.frequency, self.power)
                self.freq_os, self.pow_os = np.copy(self.frequency), np.copy(self.power)
                self.freq_cs = np.array(self.frequency[self.params['oversampling_factor']-1::self.params['oversampling_factor']])
                self.pow_cs = np.array(self.power[self.params['oversampling_factor']-1::self.params['oversampling_factor']])
                self.baseline = 1./((self.freq_cs[1]-self.freq_cs[0])*10**-6.)
                self.tau_upper = self.baseline/2.


    def load_time_series(self, nyquist=None):
        """
        Load light curve
        
        If time series data is available, the time series data
        is loaded in, and then it calculates the cadence and nyquist 
        frequency. If time series data is not provided, either the
        cadence or nyquist frequency must be provided via CLI

        Parameters
            save : bool, optional
                save all data products
            kep_corr : bool, optional
                use the module that corrects for known kepler artefacts
            stitch : bool, optional
                use the module that corrects for large gaps in data


        """
        self.nyquist, other = None, ''
        # Try loading the light curve
        if os.path.exists(os.path.join(self.params['inpdir'], '%s_LC.txt' % str(self.name))):
            self.lc = True
            self.time, self.flux = self.load_file(os.path.join(self.params['inpdir'], '%s_LC.txt' % str(self.name)))
            self.time -= min(self.time)
            self.cadence = int(round(np.nanmedian(np.diff(self.time)*24.0*60.0*60.0),0))
            self.nyquist = 10**6./(2.0*self.cadence)
            self.baseline = (max(self.time)-min(self.time))*24.*60.*60.
            self.tau_upper = self.baseline/2.
            note = '# LIGHT CURVE: %d lines of data read\n# Time series cadence: %d seconds\n'%(len(self.time),self.cadence)
            # Stitch light curve together before attempting to compute a PS
            if self.params['stitch']:
                self.stitch_data()
            # Compute a PS if there is not one w/ the option to save to inpdir for next time
            if not self.ps:
                # CASE 2: LIGHT CURVE AND NO POWER SPECTRUM
                #     ->  compute power spectrum and set oversampling factor
                self.ps, self.params['oversampling_factor'] = True, 5
                self.frequency, self.power = self.compute_spectrum(oversampling_factor=self.params['oversampling_factor'])
                if self.params['save']:
                    utils.save_file(self.frequency, self.power, os.path.join(self.params['inpdir'], '%s_PS.txt'%self.name), overwrite=self.params['overwrite'])
                note += '# NEWLY COMPUTED POWER SPECTRUM has length of %d\n'%int(len(self.frequency)/5)
            else:
                # CASE 3: LIGHT CURVE AND POWER SPECTRUM
                #     ->  calculate oversampling factor from time series and compare
                oversampling_factor = (1./((max(self.time)-min(self.time))*0.0864))/(self.frequency[1]-self.frequency[0])
                if self.params['oversampling_factor'] is not None:
                    if int(oversampling_factor) != self.params['oversampling_factor']:
                        other += "WARNING: \ncalculated vs. provided oversampling factor do NOT match"
                else:
                    if not float('%.2f'%oversampling_factor).is_integer():
                        error="\nERROR: the calculated oversampling factor is not an integer\nPlease check the input data and try again"
                        raise InputError(error)
                    else:
                        self.params['oversampling_factor'] = int(oversampling_factor)   
                self.frequency, self.power = self.fix_data(self.frequency, self.power)
            note += self.note
            self.freq_os, self.pow_os = np.copy(self.frequency), np.copy(self.power)
            self.freq_cs = np.array(self.frequency[self.params['oversampling_factor']-1::self.params['oversampling_factor']])
            self.pow_cs = np.array(self.power[self.params['oversampling_factor']-1::self.params['oversampling_factor']])
            if self.params['oversampling_factor'] != 1:
                note += '# PS oversampled by a factor of %d'%self.params['oversampling_factor']
            else:
                note += '# PS is critically-sampled'
            note += '\n# PS resolution: %.6f muHz'%(self.freq_cs[1]-self.freq_cs[0])
        if self.params['verbose']:
            print(note)
            if other != '':
                print(other)


    def load_file(self, path):
        """
        Generic load function from text file
    
        Load a light curve or a power spectrum from a basic 2xN txt file
        and stores the data into the `x` (independent variable) and `y`
        (dependent variable) arrays, where N is the length of the series.

        Parameters
            path : str
                the file path of the data file

        Returns
            x : numpy.array
                the independent variable i.e. the time or frequency array 
            y : numpy.array
                the dependent variable, in this case either the flux or power array

        """

        f = open(path, "r")
        lines = f.readlines()
        f.close()
        # Set values
        x = np.array([float(line.strip().split()[0]) for line in lines])
        y = np.array([float(line.strip().split()[1]) for line in lines])
        return x, y


    def get_warnings(self, long=10**6, note='',):
        """
        Check input data

        Parameters
            note : str, optional
                verbose output
            long : int
                arbitrary number to let user know if a "long" PS was given, as it will
                take pySYD longer to process

        Attributes
            resolution : float
                frequency resolution of input power spectrum

        """
        note = ''
        if len(self.frequency) >= long:
            note += '# WARNING(S):\n             - PS is large and will slow down the software\n'
        if self.params['stitch']:
            note += '#             - using stitch_data module - which is dodgy\n'
        if self.params['kep_corr']:
            note += '#             - used Kepler artefact correction\n'
        if self.params['ech_mask'] is not None:
            note += '#             - whitened PS to help w/ mixed modes**\n'
        if self.params['verbose'] and note != '':
            print(note)
        self.ok = True


    def stitch_data(self, gap=20):
        """
        Stitch light curve

        For computation purposes and for special cases that this does not affect the integrity of the results,
        this module 'stitches' a light curve together for time series data with large gaps. For stochastic p-mode
        oscillations, this is justified if the lifetimes of the modes are smaller than the gap. 
      
        Attributes
            time : numpy.ndarray
                original time series array
            new_time : numpy.ndarray
                corrected time series array

        Parameters
            gap : int
                how many consecutive missing cadences are considered a 'gap'

        Return
            warning : str
                prints a warning when using this method

        .. warning::
            USE THIS WITH CAUTION. This is technically not a great thing to do for primarily
            two reasons:
             #. you lose phase information *and* 
             #. can be problematic if mode lifetimes are shorter than gaps (i.e. more evolved stars)

        .. note::
            temporary solution for handling very long gaps in TESS data -- still need to
            figure out a better way to handle this


        """
        self.new_time = np.copy(self.time)
        for i in range(1,len(self.new_time)):
            if (self.new_time[i]-self.new_time[i-1]) > float(self.params['gap'])*(self.cadence/24./60./60.):
                self.new_time[i] = self.new_time[i-1]+(self.cadence/24./60./60.)
        self.time = np.copy(self.new_time)


    def compute_spectrum(self, oversampling_factor=1):
        """
        Calculate power spectrum

        NEW function to compute a power spectrum given time series data, which will
        normalize the power spectrum to spectral density according to Parseval's theorem

        Parameters
            oversampling_factor : int
                the oversampling factor to compute for the power spectrum 

        Returns
            frequency : numpy.ndarray
                the calculated frequency array in :math:`\\rm \\mu Hz`
            power : numpy.ndarray
                the calculated power density in :math:`\\rm ppm^{2} \\mu Hz^{-1}`

        .. important::

            If you are unsure if your power spectrum is in the proper units, we recommend
            using this new module to compute and normalize for you. This will ensure the
            accuracy of the results.


        """
        freq, pow = lomb(self.time, self.flux).autopower(method='fast', samples_per_peak=oversampling_factor, maximum_frequency=self.nyquist)
        # convert frequency array into proper units
        freq *= (10.**6/(24.*60.*60.))
        # normalize PS according to Parseval's theorem
        psd = 4.*pow*np.var(self.flux*1e6)/(np.sum(pow)*(freq[1]-freq[0]))
        frequency, power = self.fix_data(freq, psd) 
        return frequency, power


    def fix_data(self, frequency, power, kep_corr=False):
        """
        Fix power spectrum

        Runs power spectra through our optional tools like remove_artefact() and 
        white_mixed(). If neither of these options are used, it will return a copy
        of the original arrays

        Parameters
            frequency : numpy.ndarray
                input frequency array to correct
            power : numpy.ndarray
                input power spectrum to correct
	    
        Returns
            frequency : numpy.ndarray
                copy of the corrected frequency array 
            power : numpy.ndarray
                copy of the corrected power spectrum

        """
        if self.params['kep_corr']:
            freq, pow = self.remove_artefact(frequency, power)
        else:
            freq, pow = np.copy(frequency), np.copy(power)
        if self.params['ech_mask'] is not None:
            frequency, power = self.whiten_mixed(freq, pow)
        else:
            frequency, power = np.copy(freq), np.copy(pow)
        return frequency, power


    def set_seed(self, lower=1, upper=10**7):
        """
        Set reproducible seed
    
        For Kepler targets that require a correction via CLI (--kc), a random seed is generated
        from U~[1,10^7] and stored in stars_info.csv for reproducible results in later runs.

        Parameters
            lower : int 
                lower limit for random seed value (default=`1`)
            upper : int
                upper limit for random seed value (default=`10**7`)


        """
        seed = np.random.randint(lower,high=upper)
        df = pd.read_csv(os.path.join(self.params['infdir'], self.params['info']))
        stars = [str(each) for each in df.stars.values.tolist()]
        idx = stars.index(self.name)
        df.loc[idx,'seed'] = int(seed)
        self.params['seed'] = int(seed)
        df.to_csv(os.path.join(self.params['infdir'], self.params['info']), index=False)


    def remove_artefact(self, freq, pow, lcp=1.0/(29.4244*60*1e-6), 
                        lf_lower=[240.0,500.0], lf_upper=[380.0,530.0], 
                        hf_lower = [4530.0,5011.0,5097.0,5575.0,7020.0,7440.0,7864.0],
                        hf_upper = [4534.0,5020.0,5099.0,5585.0,7030.0,7450.0,7867.0],):
        """
        Remove Kepler short-cadence artefact
    
        Module to remove artefacts found in Kepler power spectra by replacing them with noise 
        (using linear interpolation) following a chi-squared distribution. 

        Parameters
            freq : numpy.ndarray
                input frequency array to correct
            pow : numpy.ndarray
                input power spectrum to correct
            lcp : float
                long cadence period in Msec
            lf_lower : List[float]
                lower limit of low frequency artefact
            lf_upper : List[float]
                upper limit of low frequency artefact
            hf_lower : List[float]
                lower limit of high frequency artefact
            hf_upper : List[float]
                upper limit of high frequency artefact
	    
        Returns
            frequency : numpy.ndarray
                copy of the corrected frequency array 
            power : numpy.ndarray
                copy of the corrected power spectrum

        .. note::

            Known artefacts are:
             #. 1./LC harmonics
             #. high frequency artefacts (>5000 muHz)
             #. low frequency artefacts 250-400 muHz (mostly present in Q0 and Q3 data)


        """
        frequency, power = np.copy(freq), np.copy(pow)
        resolution = frequency[1]-frequency[0]
        if self.params['seed'] is None:
            self.set_seed()
        # LC period in Msec -> 1/LC ~muHz
        artefact = (1.0+np.arange(14))*lcp
        # Estimate white noise
        white = np.mean(power[(frequency >= max(frequency)-100.0)&(frequency <= max(frequency)-50.0)])
        # Routine 1: remove 1/LC artefacts by subtracting +/- 5 muHz given each artefact
        np.random.seed(int(self.params['seed']))
        for i in range(len(artefact)):
            if artefact[i] < np.max(frequency):
                mask = np.ma.getmask(np.ma.masked_inside(frequency, artefact[i]-5.0*resolution, artefact[i]+5.0*resolution))
                if np.sum(mask) != 0:
                    power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0
        # Routine 2: fix high frequency artefacts
        np.random.seed(int(self.params['seed']))
        for lower, upper in zip(hf_lower, hf_upper):
            if lower < np.max(frequency):
                mask = np.ma.getmask(np.ma.masked_inside(frequency, lower, upper))
                if np.sum(mask) != 0:
                    power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0
        # Routine 3: remove wider, low frequency artefacts 
        np.random.seed(int(self.params['seed']))
        for lower, upper in zip(lf_lower, lf_upper):
            low = np.ma.getmask(np.ma.masked_outside(frequency, lower-20., lower))
            upp = np.ma.getmask(np.ma.masked_outside(frequency, upper, upper+20.))
            # Coeffs for linear fit
            m, b = np.polyfit(frequency[~(low*upp)], power[~(low*upp)], 1)
            mask = np.ma.getmask(np.ma.masked_inside(self.frequency, lower, upper))
            # Fill artefact frequencies with noise
            power[mask] = ((frequency[mask]*m)+b)*(np.random.chisquare(2, np.sum(mask))/2.0)
        return np.copy(frequency), np.copy(power)


    def whiten_mixed(self, freq, pow, dnu=None, lower_ech=None, upper_ech=None, notching=False,):
        """
        Remove mixed modes
    
        Module to help reduce the effects of mixed modes random white noise in place of 
        :math:`\ell=1` for subgiants with mixed modes to better constrain the large 
        frequency separation

        Parameters
            freq : numpy.ndarray
                input frequency array to correct
            pow : numpy.ndarray
                input power spectrum to correct
            folded_freq : numpy.ndarray
                frequency array modulo dnu (i.e. folded to the large separation, :math:`\Delta\nu`)
            lower_ech : float, optional
                lower frequency limit of mask to "whiten"
            upper_ech : float, optional
                upper frequency limit of mask to "whiten"
            notching : bool, optional
                if `True`, uses notching instead of generating white noise

        Returns
            frequency : numpy.ndarray
                copy of the corrected frequency array 
            power : numpy.ndarray
                copy of the corrected power array

        """
        frequency, power = np.copy(freq), np.copy(pow)
        if self.params['seed'] is None:
            self.set_seed()
        # Estimate white noise
        if not self.params['notching']:
            white = np.mean(power[(frequency >= max(frequency)-100.0)&(frequency <= max(frequency)-50.0)])
        else:
            white = min(power[(frequency >= max(frequency)-100.0)&(frequency <= max(frequency)-50.0)])
        # Take the provided dnu and "fold" the power spectrum
        folded_freq = np.copy(frequency)%self.params['force']
        mask = np.ma.getmask(np.ma.masked_inside(folded_freq, self.params['ech_mask'][0], self.params['ech_mask'][1]))
        # Set seed for reproducibility purposes
        np.random.seed(int(self.params['seed']))
        # Makes sure the mask is not empty
        if np.sum(mask) != 0:
            if self.params['notching']:
                power[mask] = white
            else:
                power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0
        # switch "force" dnu value back
        self.params['force'] = None

        return np.copy(frequency), np.copy(power)


#########################################################################################
#
# Estimating parameters
#

    def initial_estimates(self, lower_ex=1.0, upper_ex=8000.0, max_trials=6):
        """
    
        Parameters associated with the first module, which is an automated method 
        to identify power excess due to solar-like oscillations and then estimate
        the center of that region

        Parameters
            lower_ex : float, optional
                the lower frequency limit of the PS used to estimate numax
            upper_ex : float, optional
                the upper frequency limit of the PS used to estimate numax
            max_trials : int, optional
	               the number of "guesses" or trials to perform to estimate numax


        """
        self.module = 'estimates'
        # If running the first module, mask out any unwanted frequency regions
        self.frequency, self.power = np.copy(self.freq_os), np.copy(self.pow_os)
        self.params['resolution'] = self.frequency[1]-self.frequency[0]
        # mask out any unwanted frequencies
        if self.params['lower_ex'] is not None:
            lower = self.params['lower_ex']
        else:
            lower = min(self.frequency)
        if self.params['upper_ex'] is not None:
            upper = self.params['upper_ex']
        else:
            upper = max(self.frequency)
        if self.nyquist is not None and self.nyquist < upper:
            upper = self.nyquist
        self.freq = self.frequency[(self.frequency >= lower)&(self.frequency <= upper)]
        self.pow = self.power[(self.frequency >= lower)&(self.frequency <= upper)]
        if self.params['n_trials'] > max_trials:
            self.params['n_trials'] = max_trials
        if (self.params['numax'] is not None and self.params['numax'] <= 500.) or (self.nyquist is not None and self.nyquist <= 300.):
            self.params['boxes'] = np.logspace(np.log10(0.5), np.log10(25.), self.params['n_trials'])
        else:
            self.params['boxes'] = np.logspace(np.log10(50.), np.log10(500.), self.params['n_trials'])
        self.params['plotting']['estimates'], self.params['results']['estimates'] = {}, {}


    def estimate_numax(self, n_trials=3, binning=0.005, bin_mode='mean', smooth_width=20.0, ask=False):
        """
        Estimate :math:`\nu_{\mathrm{max}}`

        Automatically finds power excess due to solar-like oscillations using a
        frequency-resolved, collapsed autocorrelation function (ACF)

        Parameters
            n_trials : int, optional
                the number of trials. Default value is `3`.
            binning : float, optional
                logarithmic binning width. Default value is `0.005`.
            bin_mode : {'mean', 'median', 'gaussian'}
                mode to use when binning
            smooth_width: float, optional
                box filter width (in :math:`\rm \mu Hz`) to smooth power spectrum
            ask : bool, optional
                If `True`, it will ask which trial to use as the estimate for numax.

        Attributes
            bin_freq
            bin_pow
            bin_pow_err
            smooth_freq
            smooth_pow
            smooth_pow_err
            interp_pow
            bgcorr_pow


        """
        # Smooth the power in log-space
        self.bin_freq, self.bin_pow, self.bin_pow_err = utils.bin_data(self.freq, self.pow, width=self.params['binning'], log=True, mode=self.params['bin_mode'])
        # Smooth the power in linear-space
        self.smooth_freq, self.smooth_pow, self.smooth_pow_err = utils.bin_data(self.bin_freq, self.bin_pow, width=self.params['smooth_width'])
        if self.params['verbose']:
            print('-----------------------------------------------------------\nPS binned to %d datapoints\n\nNumax estimates\n---------------' % len(self.smooth_freq))
        # Mask out frequency values that are lower than the smoothing width to avoid weird looking fits
        mask = (self.smooth_freq >= (min(self.freq)+self.params['smooth_width'])) & (self.smooth_freq <= (max(self.freq)-self.params['smooth_width']))
        s = InterpolatedUnivariateSpline(self.smooth_freq[mask], self.smooth_pow[mask], k=1)
#        s = InterpolatedUnivariateSpline(self.smooth_freq, self.smooth_pow, k=1)
        # Interpolate and divide to get a crude background-corrected power spectrum
        self.interp_pow = s(self.freq)
        self.bgcorr_pow = self.pow/self.interp_pow
        # Collapsed ACF to find numax
        self.collapsed_acf()
        # Select trial that resulted with the highest SNR detection
        if not self.params['ask']:
            self.params['best'] = self.params['compare'].index(max(self.params['compare']))+1
            if self.params['verbose']:
                print('Selecting model %d' % self.params['best'])
        # Or ask which estimate to use
        else:
            self = utils.save_plotting(self)
            plots.plot_estimates(self, ask=True)
            value = utils.ask_int('Which estimate would you like to use? ', self.params['n_trials'])
            if isinstance(value, int):
                self.params['best'] = value
                print('Selecting model %d' % value)
            else:
                self.params['numax'] = value
                self.params['dnu'] = utils.delta_nu(value)
                print('Using numax of %.2f muHz as an initial guess' % value)
        self = utils.save_estimates(self)



    def collapsed_acf(self, step=0.25, max_snr=100.0):
        """
        Collapsed ACF

        Computes a collapsed autocorrelation function (ACF).

        Parameters
            b : int
                the trial number
            step : float
                fractional step size to use for the collapsed ACF calculation

        Attributes
            compare : List[float]
                list of SNR results from the different trials


        """
        self.params['compare'] = []
        # Computes a collapsed ACF using different "box" (or bin) sizes
        for b, box in enumerate(self.params['boxes']):
            self.params['results']['estimates'][b+1], self.params['plotting']['estimates'][b] = {}, {}
            start, cumsum, md = 0, [], []
            subset = np.ceil(box/self.params['resolution'])
            steps = np.ceil((box*self.params['step'])/self.params['resolution'])
            # Iterates through entire power spectrum using box width
            while True:
                if (start+subset) > len(self.freq):
                    break
                p = self.bgcorr_pow[int(start):int(start+subset)]
                auto = np.real(np.fft.fft(np.fft.ifft(p)*np.conj(np.fft.ifft(p))))
                cumsum.append(np.sum(np.absolute(auto-np.mean(auto))))
                md.append(np.mean(self.freq[int(start):int(start+subset)]))
                start += steps
            # subtract/normalize the summed ACF and take the max
            cumsum = np.array(cumsum)-min(cumsum)
            csum = list(cumsum/max(cumsum))
            # Pick the maximum value as an initial guess for numax
            idx = csum.index(max(csum))
            self.params['plotting']['estimates'].update({b:{'x':np.array(md),'y':np.array(csum),'maxx':md[idx],'maxy':csum[idx]}})
            # Fit Gaussian to get estimate value for numax
            try:
                best_vars, _ = curve_fit(models.gaussian, np.array(md), np.array(csum), p0=[np.median(csum), 1.0-np.median(csum), md[idx], self.constants['width_sun']*(md[idx]/self.constants['numax_sun'])], maxfev=5000, bounds=((-np.inf,-np.inf,1,-np.inf),(np.inf,np.inf,np.inf,np.inf)),)
            except Exception as _:
                self.params['plotting']['estimates'][b].update({'good_fit':False,})
                snr = 0.
            else:
                self.params['plotting']['estimates'][b].update({'good_fit':True,})
                fitx = np.linspace(min(md), max(md), 10000)
                self.params['plotting']['estimates'][b].update({'fitx':fitx,'fity':models.gaussian(fitx, *best_vars)})
                snr = max(self.params['plotting']['estimates'][b]['fity'])/np.absolute(best_vars[0])
                if snr > max_snr:
                    snr = max_snr
                self.params['results']['estimates'][b+1].update({'numax':best_vars[2], 'dnu':utils.delta_nu(best_vars[2]), 'snr':snr})
                self.params['plotting']['estimates'][b].update({'numax':best_vars[2], 'dnu':utils.delta_nu(best_vars[2]), 'snr':snr})
                if self.params['verbose']:
                    print('Numax estimate %d: %.2f +/- %.2f'%(b+1, best_vars[2], np.absolute(best_vars[3])/2.0))
                    print('S/N: %.2f' % snr)
            self.params['compare'].append(snr)


    def check_numax(self, columns=['numax', 'dnu', 'snr']):
        """
    
        Checks if there is a starting value for numax

        Parameters
            columns : List[str]
                saved columns if the estimate_numax() function was run

        Returns
            return : bool
                will return `True` if there is prior value for numax otherwise `False`.

        """
        # THIS MUST BE FIXED TOO
        # Check if numax was provided as input
        if self.params['numax'] is not None:
            if np.isnan(float(self.params['numax'])):
                raise ProcessingError("ERROR: invalid value for numax")
        else:
            # If not, checks if estimate_numax module was run
            if glob.glob(os.path.join(self.params['path'],'estimates*')):
                if not self.params['overwrite']:
                    list_of_files = glob.glob(os.path.join(self.params['path'],'estimates*'))
                    file = max(list_of_files, key=os.path.getctime)
                else:
                    file = os.path.join(self.params['path'],'estimates.csv')
                df = pd.read_csv(file)
                for col in columns:
                    self.params[col] = df.loc[0, col]
                if np.isnan(self.params['numax']):
                    raise ProcessingError("ERROR: invalid value for numax")
            else:
                # Raise error
                raise ProcessingError("ERROR: no numax provided for global fit")


#########################################################################################
#
# Deriving parameters
#

    def initial_parameters(self, lower_bg=1.0, upper_bg=8000.0,):
        """
    
        Gets initial guesses for granulation components (i.e. timescales and amplitudes) using
        solar scaling relations. This resets the power spectrum and has its own independent
        filter (i.e. [lower,upper] mask) to use for this subroutine.

        Parameters
            lower_bg : float, optional
                minimum frequency to use for the background-fitting 
            upper_bg : float, optional
                maximum frequency to use for the background-fitting

        .. warning::

            This is typically sufficient for most stars but may affect evolved stars and
            need to be adjusted!


        """
        self.module = 'parameters'
        self.frequency, self.power = np.copy(self.freq_os), np.copy(self.pow_os)
        self.params['resolution'] = self.frequency[1]-self.frequency[0]
        if self.params['lower_bg'] is not None:
            lower = self.params['lower_bg']
        else:
            lower = min(self.frequency)
        if self.params['upper_bg'] is not None:
            upper = self.params['upper_bg']
        else:
            upper = max(self.frequency)
        if self.nyquist is not None and self.nyquist < upper:
            upper = self.nyquist
        self.params['bg_mask']=[lower,upper]

        # Mask power spectrum for main module
        mask = np.ma.getmask(np.ma.masked_inside(self.frequency, self.params['bg_mask'][0], self.params['bg_mask'][1]))
        self.frequency, self.power = np.copy(self.frequency[mask]), np.copy(self.power[mask])
        self.random_pow = np.copy(self.power)
        # Get other relevant initial conditions
        self.i = 0
        self.params['results']['parameters'] = {'numax_smooth':[],'A_smooth':[],'numax_gauss':[],'A_gauss':[],'FWHM':[],'dnu':[]}
        # Use scaling relations from sun to get starting points
        self.solar_scaling()
        self.params['plotting']['parameters'] = {'exp_numax':self.params['exp_numax'],'nlaws_orig':len(self.params['mnu']),'mnu_orig':np.copy(self.params['mnu']),'b_orig':np.copy(self.params['b'])}
        if self.params['verbose']:
            print('-----------------------------------------------------------\nGLOBAL FIT\n-----------------------------------------------------------')


    def solar_scaling(self, numax=None, scaling='tau_sun_single', max_laws=3, ex_width=1.0,
                      lower_ps=None, upper_ps=None,):
        """
        Solar scaling relation
    
        Uses scaling relations from the Sun to:
         #. estimate the width of the region of oscillations using numax
         #. guess starting values for granulation time scales

        Parameters
	           scaling : str
	               which scaling relation to use
            max_laws : int
                the maximum number of resolvable Harvey-like components
           	ex_width : float
                fractional width of the power excess to use (w.r.t. solar)

        Attributes
            b : List[float]
                list of starting points for
            b_orig : List[float]
                copy of the list of starting points for
            mnu : List[float]
                list of starting points for 
            mnu_orig : List[float]
                copy of list of starting points for 
            nlaws : int
                estimated number of Harvey-like components 
	           nlaws_orig : int
                copy of the estimated number of Harvey-like components 

        """
        self.params['exp_numax'] = self.params['numax']
        # Use scaling relations to estimate width of oscillation region to mask out of the background fit
        width = self.constants['width_sun']*(self.params['exp_numax']/self.constants['numax_sun'])
        maxpower = [self.params['exp_numax']-(width*self.params['ex_width']), self.params['exp_numax']+(width*self.params['ex_width'])]
        if self.params['lower_ps'] is not None:
            maxpower[0] = self.params['lower_ps']
        if self.params['upper_ps'] is not None:
            maxpower[1] = self.params['upper_ps']
        self.params['ps_mask'] = [maxpower[0], maxpower[1]]
        # Use scaling relation for granulation timescales from the sun to get starting points
        scale = self.constants['numax_sun']/self.params['exp_numax']
        # make sure interval is not empty
        if not list(self.frequency[(self.frequency>=self.params['ps_mask'][0])&(self.frequency<=self.params['ps_mask'][1])]):
            raise InputError("ERROR: frequency region for power excess is null\nPlease specify an appropriate numax and/or frequency limits for the power excess (via --lp/--up)")
        # Estimate granulation time scales
        if scaling == 'tau_sun_single':
            taus = np.array(self.constants['tau_sun_single'])*scale
        else:
            taus = np.array(self.constants['tau_sun'])*scale
        taus = taus[taus <= self.baseline]
        b = taus*10**-6.
        mnu = (1.0/taus)*10**5.
        self.params['b'] = b[mnu >= min(self.frequency)]
        self.params['mnu'] = mnu[mnu >= min(self.frequency)]
        if len(self.params['mnu']) == 0:
            self.params['b'] = b[mnu >= 10.] 
            self.params['mnu'] = mnu[mnu >= 10.]
        if len(self.params['mnu']) > max_laws:
            self.params['b'] = self.params['b'][-max_laws:]
            self.params['mnu'] = self.params['mnu'][-max_laws:]
        # Save copies for plotting after the analysis
        self.params['nlaws'], self.params['a'] = len(self.params['mnu']), []


    def get_samples(self,):
        """

        """
        # Switch to critically-sampled PS if sampling
        mask = np.ma.getmask(np.ma.masked_inside(self.freq_cs, self.params['bg_mask'][0], self.params['bg_mask'][1]))
        self.frequency, self.power = np.copy(self.freq_cs[mask]), np.copy(self.pow_cs[mask])
        self.params['resolution'] = self.frequency[1]-self.frequency[0]
        if self.params['verbose']:
            from tqdm import tqdm 
            self.pbar = tqdm(total=self.params['mc_iter'])
            self.pbar.update(1)
        # iterate for x steps
        while self.i < self.params['mc_iter']:
            # TODO: SET SEED
            if self.single_step():
                self.i += 1
                if self.params['verbose']:
                    self.pbar.update(1)
                    if self.i == self.params['mc_iter']:
                        self.pbar.close()


    def first_step(self, converge=False, background=True, globe=True,):
        """

        """
        # Background corrections
        self.estimate_background()
        self.model_background()
        # Global fit
        if self.params['globe']:
            # global fit
            self.fit_global()
            if self.params['verbose'] and self.params['mc_iter'] > 1:
                print('-----------------------------------------------------------\nSampling routine:')


    def single_step(self, converge=False,):
        """

        """
        self.random_pow = (np.random.chisquare(2, len(self.frequency))*self.power)/2.
        # Background corrections
        self.estimate_background()
        converge = self.get_background()
        # Requires bg fit to converge before moving on
        if not converge:
            return False
        if self.params['globe']:
            self.fit_global()
            return True


    def fit_global(self, acf_mask=None):
        """
        Fit global properties

        The main pySYD pipeline routine that:
         #. fits the granulation background and correct for it
         #. then measures numax and dnu 


        """
        # get numax
        self.get_numax_smooth()
        self.get_numax_gaussian()
        # get dnu
        self.compute_acf()
        if self.i == 0:
            self.initial_dnu()
            self.get_acf_cutout()
            self.get_ridges()
        else:
            self.estimate_dnu()


    def estimate_background(self, ind_width=20.0,):
        """

        Estimates initial guesses for the stellar background 

        Parameters
            ind_width : float
                the independent average smoothing width (default = `20.0` :math:`\rm \mu Hz`)

        Attributes
            bin_freq : numpy.ndarray
                binned frequency array 
            bin_pow : numpy.ndarray
                binned power array using a bin size set by ind_width
            bin_err : numpy.ndarray
                binned power error array 

        Returns
            return : bool
                will return `True` if the model converged

        """
        # Bin power spectrum to model stellar background/correlated red noise components
        self.bin_freq, self.bin_pow, self.bin_err = utils.bin_data(self.frequency, self.random_pow, width=self.params['ind_width'], mode=self.params['bin_mode'])
        # Mask out region with power excess
        mask = np.ma.getmask(np.ma.masked_outside(self.bin_freq, self.params['ps_mask'][0], self.params['ps_mask'][1]))
        self.bin_freq, self.bin_pow, self.bin_err = self.bin_freq[mask], self.bin_pow[mask], self.bin_err[mask]
        # Estimate white noise level
        self.get_white_noise()
        # Get initial guesses for the optimization of the background model
        self.estimate_initial_red()


    def get_white_noise(self):
        """
        Estimate white noise

        Estimate the white noise level (in muHz) by taking the mean of
        the last 10% of the power spectrum.

        Attributes
            noise : float
                estimate of white or frequency-independent noise level

        """
        mask = (self.frequency > (max(self.frequency)-0.1*max(self.frequency)))&(self.frequency < max(self.frequency))
        self.params['noise'] = np.mean(self.random_pow[mask])


    def estimate_initial_red(self, box_filter=1.0, n_rms=20,):
        """
        Estimate red noise

        Estimates amplitude of red noise components by using a smoothed version of the power
        spectrum with the power excess region masked out. This will take the mean of a specified 
        number of points (via -nrms, default=20) for each Harvey-like component.

        Parameters
            box_filter : float
                the size of the 1D box smoothing filter (default = `1.0` :math:`\rm \mu Hz`)
            n_rms : int
                number of data points to estimate red noise contributions (default = `20`)

        Attributes
            smooth_pow : numpy.ndarray
                smoothed power spectrum after applying the box filter
            guesses : numpy.ndarray
                initial guesses for background model fitting
            a : List[float]
                initial guesses for the amplitudes of all Harvey components
            a_orig : numpy.ndarray
                copy of the original guesses for Harvey amplitudes


        """
        # Exclude region with power excess and smooth to estimate red noise components
        boxkernel = Box1DKernel(int(np.ceil(self.params['box_filter']/self.params['resolution'])))
        mask = (self.frequency >= self.params['ps_mask'][0])&(self.frequency <= self.params['ps_mask'][1])
        self.smooth_pow = convolve(self.random_pow, boxkernel)
        # Temporary array for inputs into model optimization
        self.params['guesses'] = np.zeros((self.params['nlaws']*2+1))
        # Estimate amplitude for each harvey component
        for n, nu in enumerate(self.params['mnu']):
            diff = list(np.absolute(self.frequency-nu))
            idx = diff.index(min(diff))
            if idx < self.params['n_rms']:
                self.params['guesses'][2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][:self.params['n_rms']]))/(4.*self.params['b'][n]))
            elif (len(self.smooth_pow[~mask])-idx) < self.params['n_rms']:
                self.params['guesses'][2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][-self.params['n_rms']:]))/(4.*self.params['b'][n]))
            else:
                self.params['guesses'][2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][idx-int(self.params['n_rms']/2):idx+int(self.params['n_rms']/2)]))/(4.*self.params['b'][n]))
            self.params['guesses'][2*n] = self.params['b'][n]
            self.params['a'].append(self.params['guesses'][2*n+1])
        self.params['guesses'][-1] = self.params['noise']


    def model_background(self, n_laws=None, fix_wn=False, basis='tau_sigma',):
        """
        Determines the best-fit model for the stellar granulation background in the power spectrum
        by iterating through several models, where the initial guess for the number of Harvey-like 
        component(s) to model is estimated from a solar scaling relation.

        Parameters
            n_laws : int
                force number of Harvey-like components in background fit (default = `None`)
            fix_wn : bool
                fix the white noise level in the background fit (default = `False`)
            basis : str
                which basis to use for background fitting, e.g. {a,b} parametrization (default = `tau_sigma`)

        Attributes
            bounds : list
                the bounds on the Harvey parameters for a given model
            bic : list
                the BIC statistic
            aic : list
                the AIC statistic
            paras : list
                the fitted parameters for each model that was explored

        Returns
            return : bool
                will return `True` if fitting failed and the iteration must be repeated, otherwise `False`.

        """
        if self.params['background']:
            if self.params['verbose']:
                print('PS binned to %d data points\n\nBackground model\n----------------' % len(self.bin_freq))    
            # Get best-fit model
            self.params['bounds'], self.params['bic'], self.params['aic'], self.params['paras'] = [], [], [], []
            if self.params['n_laws'] is not None:
                if not self.params['fix_wn']:
                    self.params['models'] = [self.params['n_laws']*2, self.params['n_laws']*2+1]
                else:
                    self.params['models'] = [self.params['n_laws']*2]
            else:
                if self.params['fix_wn']:
                    self.params['models'] = np.arange(0,(self.params['nlaws']+1)*2,2)
                else:
                    self.params['models'] = np.arange((self.params['nlaws']+1)*2)
            if self.params['verbose'] and len(self.params['models']) > 1:
                print('Comparing %d different models:' % len(self.params['models']))
            for n, n_free in enumerate(self.params['models']):
                note = 'Model %d: %d Harvey-like component(s) + '%(n, n_free//2)
                if self.params['basis'] == 'a_b':
                    bounds = ([0.0,0.0]*(n_free//2), [np.inf,self.tau_upper]*(n_free//2))
                else:
                    bounds = ([0.0,0.0]*(n_free//2), [np.inf,np.inf]*(n_free//2))
                if not n_free%2:
                    note += 'white noise fixed'
                    *guesses, = self.params['guesses'][-int(2*(n_free//2)+1):-1]
                else:
                    note += 'white noise term'
                    bounds[0].append(10.**-2)
                    bounds[1].append(np.inf)
                    *guesses, = self.params['guesses'][-int(2*(n_free//2)+1):]
                self.params['bounds'].append(bounds)
                if n_free == 0:
                    self.params['paras'].append([])
                    model = np.ones_like(self.bin_pow)*self.params['noise']
                    b, a = models.compute_bic(self.bin_pow, model, n_parameters=n_free), models.compute_aic(self.bin_pow, model, n_parameters=n_free)
                    self.params['bic'].append(b)
                    self.params['aic'].append(a)
                else:
                    try:
                        if not n_free%2:
                            # If white noise is fixed, pass "noise" estimate to lambda operator
                            pars, _ = curve_fit(self.params['functions'][n_free](self.params['noise']), self.bin_freq, self.bin_pow, p0=guesses, sigma=self.bin_err, bounds=bounds)
                        else:
                            # If white noise is a free parameter, good to go!
                            pars, _ = curve_fit(self.params['functions'][n_free], self.bin_freq, self.bin_pow, p0=guesses, sigma=self.bin_err, bounds=bounds)
                    except RuntimeError as _:
                        self.params['paras'].append([])
                        self.params['bic'].append(np.inf)
                        self.params['aic'].append(np.inf)
                    else:
                        self.params['paras'].append(pars)
                        model = models.background(self.bin_freq, pars, noise=self.params['noise'])
                        b, a = models.compute_bic(self.bin_pow, model, n_parameters=n_free), models.compute_aic(self.bin_pow, model, n_parameters=n_free)
                        self.params['bic'].append(b)
                        self.params['aic'].append(a)
                if self.params['verbose']:
                    note += '\n BIC = %.2f | AIC = %.2f'%(b, a)
                    print(note)
            # Did the fit converge 
            if np.isfinite(min(self.params[self.params['metric']])):
                self.correct_background()
            # Otherwise raise error that fit did not converge
            else:
                raise ProcessingError("Background fit failed to converge for any models.\n\n We recommend disabling this feature using our boolean background flag ('-b' )")
        else:
            if self.params['verbose']:
                print('-----------------------------------------------------------\nWARNING: estimating global parameters from raw PS:')
            self.bg_corr = np.copy(self.random_pow)/self.params['noise']
            self.params['pars'] = ([self.params['noise']])


    def correct_background(self, metric='aic'):
        """
        Saves information re: the selected best-fit model (for the stellar background).
        
        Parameters
            metric : str
                which metric to use (i.e. bic or aic) for model selection (default = `'bic'`)

        Attributes
            bg_corr : numpy.ndarray
                background-corrected power spectrum -> currently background-DIVIDED
            model : int
                selected best-fit background model
            pars : numpy.ndarray
                derived parameters for best-fit background model


        """
        idx = self.params[self.params['metric']].index(min(self.params[self.params['metric']]))
        self.params['selected'] = self.params['models'][idx]
        self.params['bounds'] = self.params['bounds'][idx]
        self.params['pars'] = self.params['paras'][idx]
        # If model with fixed white noise is preferred, change 'fix_wn' option
        if not int(self.params['selected']%2):
            self.params['fix_wn'] = True
        # Store model results for plotting
        if self.params['nlaws'] != self.params['selected']//2:
            self.params['nlaws'] = self.params['selected']//2
            self.params['b'] = self.params['b'][:(self.params['nlaws'])]
            self.params['mnu'] = self.params['mnu'][:(self.params['nlaws'])]
        if self.params['verbose'] and len(self.params['models']) > 1:
            print('Based on %s statistic: model %d'%(self.params['metric'].upper(),idx))
        # Compare different model results
        self.bg_corr = self.random_pow/models.background(self.frequency, self.params['pars'], noise=self.params['noise'])
        # Save background-corrected power spectrum
        if self.params['save']:
            utils.save_file(self.frequency, self.bg_corr, os.path.join(self.params['path'], '%s_bg_corr.txt'%self.name), overwrite=self.params['overwrite'])
        # Create appropriate keys for star based on best-fit model
        for n in range(self.params['nlaws']):
            self.params['results']['parameters']['tau_%d'%(n+1)] = []
            self.params['results']['parameters']['sigma_%d'%(n+1)] = []
        if not self.params['fix_wn']:
            self.params['results']['parameters']['white'] = []
        # Save the final values
        for n in range(self.params['nlaws']):
            self.params['results']['parameters']['tau_%d'%(n+1)].append(self.params['pars'][2*n]*10.**6)
            self.params['results']['parameters']['sigma_%d'%(n+1)].append(self.params['pars'][2*n+1])
        if not self.params['fix_wn']:
            self.params['results']['parameters']['white'].append(self.params['pars'][-1])


    def get_background(self):
        """
        Calculate red noise

        Calculates the red noise levels in a power spectrum due to the background 
        stellar contribution

        Returns
            return : bool
                returns `True` if model converges
   
     
        """
        if self.params['background']:
            # Use as initial guesses for the optimized model
            try:
                if self.params['fix_wn']:
                    # If white noise is fixed, pass "noise" estimate to lambda operator
                    self.params['pars'], _ = curve_fit(self.params['functions'][self.params['selected']](self.params['noise']), self.bin_freq, self.bin_pow, p0=self.params['guesses'][:-1], sigma=self.bin_err, bounds=self.params['bounds'])
                else:
                    # If white noise is a free parameter, good to go!
                    self.params['pars'], _ = curve_fit(self.params['functions'][self.params['selected']], self.bin_freq, self.bin_pow, p0=self.params['guesses'], sigma=self.bin_err, bounds=self.params['bounds'])
            except RuntimeError as _:
                return False
            else:
                self.bg_corr = self.random_pow/models.background(self.frequency, self.params['pars'], noise=self.params['noise'])
                # save final values for Harvey components
                for n in range(self.params['nlaws']):
                    self.params['results']['parameters']['tau_%d'%(n+1)].append(self.params['pars'][2*n]*10.**6)
                    self.params['results']['parameters']['sigma_%d'%(n+1)].append(self.params['pars'][2*n+1])
                if not self.params['fix_wn']:
                    self.params['results']['parameters']['white'].append(self.params['pars'][-1])
        else:
            self.bg_corr = np.copy(self.random_pow)/self.params['noise']
            self.params['pars'] = ([self.params['noise']])
        return True


    def get_numax_smooth(self, sm_par=None,):
        """
        Smooth :math:`\nu_{\mathrm{max}}`

        Estimate numax taking the peak of the smoothed power spectrum

        Attributes
            pssm
            pssm_bgcorr
            obs_numax
            exp_dnu


        """
        # Smoothing width for determining numax
        if self.params['sm_par'] is not None:
            sm_par = self.params['sm_par']
        else:
            sm_par = 4.*(self.params['numax']/self.constants['numax_sun'])**0.2
        if sm_par < 1.:
            sm_par = 1.
        sig = (sm_par*(self.params['dnu']/self.params['resolution']))/np.sqrt(8.0*np.log(2.0))
        self.pssm = convolve_fft(np.copy(self.random_pow), Gaussian1DKernel(int(sig)))
        self.pssm_bgcorr = self.pssm-models.background(self.frequency, self.params['pars'], noise=self.params['noise'])
        mask = np.ma.getmask(np.ma.masked_inside(self.frequency, self.params['ps_mask'][0], self.params['ps_mask'][1]))
        self.region_freq, self.region_pow = self.frequency[mask], self.pssm_bgcorr[mask]
        idx = utils.return_max(self.region_freq, self.region_pow, index=True)
        self.params['results']['parameters']['numax_smooth'].append(self.region_freq[idx])
        self.params['results']['parameters']['A_smooth'].append(self.region_pow[idx])
        self.params['obs_numax'] = self.params['results']['parameters']['numax_smooth'][0]
        self.params['exp_dnu'] = utils.delta_nu(self.params['obs_numax'])



    def get_numax_gaussian(self):
        """
        Gaussian :math:`\nu_{\mathrm{max}}`

        Estimate numax by fitting a Gaussian to the power spectrum and adopting the center value
    
        Attributes
            new_freq
            numax_fit


        """
        guesses = [0.0, np.absolute(max(self.region_pow)), self.params['obs_numax'], (max(self.region_freq)-min(self.region_freq))/np.sqrt(8.0*np.log(2.0))]
        bb = ([-np.inf,0.0,0.01,0.01],[np.inf,np.inf,np.inf,np.inf])
        gauss, _ = curve_fit(models.gaussian, self.region_freq, self.region_pow, p0=guesses, bounds=bb, maxfev=5000)
        # Save values
        self.params['results']['parameters']['numax_gauss'].append(gauss[2])
        self.params['results']['parameters']['A_gauss'].append(gauss[1])
        self.params['results']['parameters']['FWHM'].append(gauss[3])
        if self.i == 0:
            # Create an array with finer resolution for plotting
            new_freq = np.linspace(min(self.region_freq), max(self.region_freq), 10000)
            self.params['plotting']['parameters'].update({'new_freq':new_freq, 'numax_fit':models.gaussian(new_freq, *gauss)})


    def compute_acf(self, fft=True, smooth_ps=2.5, ps_mask=None,):
        """
        Compute the ACF of the smooth background corrected power spectrum.

        Parameters
            fft : bool
                if `True`, uses FFTs to compute the ACF, otherwise it will use
                ``numpy.correlate`` (default = `True`)
            smooth_ps : float, optional
                convolve the background-corrected PS with a box filter of this width (:math:`\rm \mu Hz`)

        Attributes
            bgcorr_smooth : numpy.ndarray
                
            lag
            auto

        """
        if self.params['dnu'] is not None:
            self.guess = self.params['dnu']
        else:
            self.guess = self.params['exp_dnu']
        # Optional smoothing of PS to remove fine structure before computing ACF
        if int(self.params['smooth_ps']) != 0:
            boxkernel = Box1DKernel(int(np.ceil(self.params['smooth_ps']/self.params['resolution'])))
            self.bgcorr_smooth = convolve(self.bg_corr, boxkernel)
        else:
            self.bgcorr_smooth = np.copy(self.bg_corr)
        # Use only power near the expected numax to reduce additional noise in ACF
        power = self.bgcorr_smooth[(self.frequency >= self.params['ps_mask'][0])&(self.frequency <= self.params['ps_mask'][1])]
        lag = np.arange(0.0, len(power))*self.params['resolution']
        if fft:
            auto = np.real(np.fft.fft(np.fft.ifft(power)*np.conj(np.fft.ifft(power))))
        else:
            auto = np.correlate(power-np.mean(power), power-np.mean(power), "full")
            auto = auto[int(auto.size/2):]
        mask = np.ma.getmask(np.ma.masked_inside(lag, self.guess/4., 2.*self.guess+self.guess/4.))
        lag = lag[mask]
        auto = auto[mask]
        auto -= min(auto)
        auto /= max(auto)
        self.lag, self.auto = np.copy(lag), np.copy(auto)


    def initial_dnu(self, force=None, method='D', n_peaks=10,):
        """
        More modular functions to estimate dnu on the first iteration given
        different methods. By default, we have been using a Gaussian weighting
        centered on the expected value for dnu (determine from the pipeline).
        One can also "force" or provide a value for dnu.

        Parameters
            method : str
                which method to use, where: 
                 - 'M' == Maryum == scipy's find_peaks module
                 - 'A' == Ashley == Ashley's module from the functions script
                 - 'D' == Dennis == weighting technique
            n_peaks : int
                the number of peaks to identify in the ACF
            force : float, optional
                option to "force" a dnu value, but is `None` by default.

        Attributes
            peaks_l
            peaks_a
            best_lag
            best_auto

        """
        if self.params['method'] == 'M':
            # Get peaks from ACF using scipy package
            peak_idx, _ = find_peaks(self.auto)
            peaks_l0, peaks_a0 = self.lag[peak_idx], self.auto[peak_idx]
            # Pick n highest peaks
            self.peaks_l, self.peaks_a = utils.max_elements(peaks_l0, peaks_a0, npeaks=self.params['n_peaks'])
        elif self.params['method'] == 'A':
            # Get peaks from ACF using Ashley's module 
            self.peaks_l, self.peaks_a = utils.max_elements(self.lag, self.auto, npeaks=self.params['n_peaks'])
        elif self.params['method'] == 'D':
            # Get peaks from ACF by providing dnu to weight the array (aka Dennis' routine)
            self.peaks_l, self.peaks_a = utils.max_elements(self.lag, self.auto, npeaks=self.params['n_peaks'], exp_dnu=self.guess)
        else:
            pass
        # Pick "best" peak in ACF (i.e. closest to expected dnu)
        idx = utils.return_max(self.peaks_l, self.peaks_a, index=True, exp_dnu=self.guess)
        self.params['best_lag'], self.params['best_auto'] = self.peaks_l[idx], self.peaks_a[idx]


    def get_acf_cutout(self, threshold=1.0, acf_mask=None,):
        """

        Gets the region in the ACF centered on the correct peak to prevent pySYD
        from getting stuck in a local maximum (i.e. fractional harmonics)

        Parameters
            threshold : float
                the threshold is multiplied by the full-width half-maximum value, centered on the peak 
                in the ACF to determine the width of the cutout region
            acf_mask : List[float,float]
                limits (i.e. lower, upper) to use for ACF "cutout"

        Attributes
            zoom_lag
            zoom_auto
            acf_guesses
            acf_bb
            obs_dnu
            new_lag
            dnu_fit
            obs_acf
       

        """
        # Calculate FWHM
        if list(self.lag[(self.lag<self.params['best_lag'])&(self.auto<=self.params['best_auto']/2.)]):
            left_lag = self.lag[(self.lag<self.params['best_lag'])&(self.auto<=self.params['best_auto']/2.)][-1]
            left_auto = self.auto[(self.lag<self.params['best_lag'])&(self.auto<=self.params['best_auto']/2.)][-1]
        else:
            left_lag = self.lag[0]
            left_auto = self.auto[0]
        if list(self.lag[(self.lag>self.params['best_lag'])&(self.auto<=self.params['best_auto']/2.)]):
            right_lag = self.lag[(self.lag>self.params['best_lag'])&(self.auto<=self.params['best_auto']/2.)][0]
            right_auto = self.auto[(self.lag>self.params['best_lag'])&(self.auto<=self.params['best_auto']/2.)][0]
        else:
            right_lag = self.lag[-1]
            right_auto = self.auto[-1]
        # Lag limits to use for ACF mask or "cutout"
        self.params['acf_mask']=[self.params['best_lag']-(self.params['best_lag']-left_lag)*self.params['threshold'], self.params['best_lag']+(right_lag-self.params['best_lag'])*self.params['threshold']]
        self.zoom_lag = self.lag[(self.lag>=self.params['acf_mask'][0])&(self.lag<=self.params['acf_mask'][1])]
        self.zoom_auto = self.auto[(self.lag>=self.params['acf_mask'][0])&(self.lag<=self.params['acf_mask'][1])]
        # Boundary conditions and initial guesses stay the same for all iterations
        self.params['acf_guesses'] = [np.mean(self.zoom_auto), self.params['best_auto'], self.params['best_lag'], self.params['best_lag']*0.01*2.]
        self.params['acf_bb'] = ([-np.inf,0.,min(self.zoom_lag),10**-2.],[np.inf,np.inf,max(self.zoom_lag),2.*(max(self.zoom_lag)-min(self.zoom_lag))]) 
        # Fit a Gaussian function to the selected peak in the ACF to get dnu
        gauss, _ = curve_fit(models.gaussian, self.zoom_lag, self.zoom_auto, p0=self.params['acf_guesses'], bounds=self.params['acf_bb'])
        self.params['results']['parameters']['dnu'].append(gauss[2])
        self.params['obs_dnu'] = gauss[2]
        # Save for plotting
        self.params['plotting']['parameters'].update({'obs_dnu':gauss[2], 
          'new_lag':np.linspace(min(self.zoom_lag),max(self.zoom_lag),2000), 
          'dnu_fit':models.gaussian(np.linspace(min(self.zoom_lag),max(self.zoom_lag),2000), *gauss),})


    def get_ridges(self, clip_value=3.0,):
        """

        Determine the best frequency spacing by determining which forms the "best"
        ridges -- TODO: still under development

        Parameters
            clip_value : float
                lower limit of distance modulus. Default value is `0.0`

        Attributes
            xax
            yax
            zz
            z


        """
        self.echelle()
        copy = self.z.flatten()
        n = int(np.ceil(self.params['obs_dnu']/self.params['resolution']))
        xax = np.linspace(0.0, self.params['obs_dnu'], n)
        yax = np.zeros_like(xax)
        modx = self.frequency%self.params['obs_dnu']
        for k in range(n-1):
            mask = (modx >= xax[k])&(modx < xax[k+1])
            if self.bg_corr[(modx >= xax[k])&(modx < xax[k+1])] != []:
                xax[k] = np.median(modx[(modx >= xax[k])&(modx < xax[k+1])])
                yax[k] = np.sum(self.bg_corr[(modx >= xax[k])&(modx < xax[k+1])])
        mask = np.ma.getmask(np.ma.masked_where(yax == 0.0, yax))
        xax, yax = xax[~mask], yax[~mask]
        self.xax = np.array(xax.tolist()+list(xax+self.params['obs_dnu']))
        self.yax = np.array(list(yax)+list(yax))-min(yax)
        # Clip the lower bound (`clip_value`)
        if int(self.params['clip_value']) != 0:
            cut = np.nanmedian(copy)+(self.params['clip_value']*np.nanmedian(copy))
            copy[copy >= cut] = cut
        self.zz = copy
        self.z = copy.reshape((self.z.shape[0], self.z.shape[1]))
        self = utils.save_plotting(self)
        self.i += 1


    def echelle(self, smooth_ech=None, nox=50, noy=0, hey=False,):
        """
        Echelle diagram

        Creates required arrays to plot an echelle diagram in the final figure

        Parameters
            smooth_ech : float, optional
                value to smooth (i.e. convolve) ED by
            nox : int
                number of grid points in x-axis of echelle diagram (not implemented yet)
            noy : int
                number of orders (y-axis) to show in echelle diagram (not implemented yet)
            hey : bool, optional
                plugin for Dan Hey's echelle package (not currently implemented)

        Attributes
            x : numpy.ndarray
                folded frequencies (x-axis) for echelle diagram
            y : numpy.ndarray
                frequency array (y-axis) for echelle diagram
            z : numpy.meshgrid
                smoothed + summed 2d power for echelle diagram
            extent : List[float]
                extent == [min(x), max(x), min(y), max(y)]

        Returns
            smoothed : numpy.meshgrid
                resulting echelle diagram based on the observed $\delta \nu$ 
            gridx : numpy.ndarray
                grid of x-axis measurements (distance modulus) for echelle diagram
            gridy : numpy.ndarray
                grid of y-axis measurements (frequency) for echelle diagram
            extent : List[float]
                The bounding box in data coordinates that the image will fill. 
                The image is stretched individually along x and y to fill the box.

        """
        if self.params['smooth_ech'] is not None:
            boxkernel = Box1DKernel(int(np.ceil(self.params['smooth_ech']/self.params['resolution'])))
            smooth_y = convolve(self.bg_corr, boxkernel)
        else:
            smooth_y = np.copy(self.bg_corr)
        # If the number of desired orders is not provided
        if self.params['noy'] == 0:
            self.params['noy'] = int(self.params['obs_numax']/self.params['obs_dnu']//2)
        # Make sure n_across isn't finer than the actual resolution grid
        if self.params['nox'] >= int(np.ceil(self.params['obs_dnu']/self.params['resolution'])):
            self.params['nox'] = int(np.ceil(self.params['obs_dnu']/self.params['resolution']/3.))
        nx, ny = (self.params['nox'], self.params['noy'])
        self.x = np.linspace(0.0, 2*self.params['obs_dnu'], 2*nx+1)
        yy = np.arange(self.params['obs_numax']%(self.params['obs_dnu']/2.),max(self.frequency),self.params['obs_dnu'])
        lower, upper = self.params['obs_numax']-3*self.params['obs_dnu']/2.-(self.params['obs_dnu']*(ny//2)), self.params['obs_numax']+self.params['obs_dnu']/2.+(self.params['obs_dnu']*(ny//2))
        self.y = yy[(yy >= lower)&(yy <= upper)]
        z = np.zeros((ny+1,2*nx))
        for i in range(1,ny+1):
            y_mask = ((self.frequency >= self.y[i-1]) & (self.frequency < self.y[i]))
            for j in range(nx):
                x_mask = ((self.frequency%(self.params['obs_dnu']) >= self.x[j]) & (self.frequency%(self.params['obs_dnu']) < self.x[j+1]))
                if smooth_y[x_mask & y_mask] != []:
                    z[i][j] = np.sum(smooth_y[x_mask & y_mask])
                else:
                    z[i][j] = np.nan
        z[0][:nx], z[-1][nx:] = np.nan, np.nan
        for k in range(ny):
            z[k][nx:] = z[k+1][:nx]
        self.z = np.copy(z)
        self.extent = [min(self.x),max(self.x),min(self.y),max(self.y)]


    def estimate_dnu(self):
        # define the peak in the ACF
        self.zoom_lag = self.lag[(self.lag>=self.params['acf_mask'][0])&(self.lag<=self.params['acf_mask'][1])]
        self.zoom_auto = self.auto[(self.lag>=self.params['acf_mask'][0])&(self.lag<=self.params['acf_mask'][1])]
        # fit a Gaussian function to the selected peak in the ACF
        gauss, _ = curve_fit(models.gaussian, self.zoom_lag, self.zoom_auto, p0=self.params['acf_guesses'], bounds=self.params['acf_bb'])
        # the center of that Gaussian is our estimate for Dnu
        self.params['results']['parameters']['dnu'].append(gauss[2]) 


class InputError(Exception):
    def __repr__(self):
        return "InputError"
    __str__ = __repr__


class InputError(Exception):
    def __repr__(self):
        return "ProcessingError"
    __str__ = __repr__