import os
import glob
import numpy as np
import pandas as pd
from astropy.stats import mad_std
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle as lomb
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft


# Package mode
from . import utils
from . import plots
from . import models



class LightCurve:

    def __init__(self, star):
        """

        """
        self.name = star.name

    def __repr__(self):
        return "<Star {} LC>".format(self.name)


class PowerSpectrum:

    def __init__(self, star):
        """

        """
        self.name = star.name

    def __repr__(self):
        return "<Star {} PS>".format(self.name)


class Target:

    def __init__(self, name, args):
        """Main pipeline target object

        .. deprecated:: 1.6.0
                  `Target.ok` will be removed in pySYD 6.0.0, it is replaced by
                  new error handling, that will instead raise exceptions or warnings 
        
        A new instance (or star) is created for each target that is processed.
        Instantiation copies the relevant, individual star dictionary (and the inherited 
        constants) and will then load in data using the provided star `name`

        Parameters
            name : str
                which target to load in and/or process
            args : :mod:`pysyd.utils.Parameters`
                container class of pysyd parameters

        Attributes
            params : Dict
                copy of args.params[name] dictionary with pysyd parameters and options

        """
        self.name = name
        self.constants = args.constants
        if name in args.params:
            # load star-specific information
            self.params = args.params[name]
        else:
            # if not available, loads defaults
            self.params = utils.Parameters()


    def __repr__(self):
        return "pysyd.target.Target(name=%r, params=%r)" % (self.name, self.params)


    def __str__(self):
        return "<Star {}>".format(self.name)


    def _adjust_parameters(self, adjust=True, defaults=None,):
        """Adjust low/high numax
    
        Adjusts default parameters for low vs high numax configurations (not tested yet -
        need to do)

        Parameters
            star : str
                individual star ID
            adjust : bool, optional
                maximum number of resolvable Harvey components
            defaults : str, optional
                option for when numax is not known but can differentiate between "low" vs. "high" frequencies

        if self.params['numax'] is not None:
            if self.params['numax'] < 500.:
                self.params['boxes'] = np.logspace(np.log10(0.5), np.log10(25.), self.params['n_trials'])
                self.params['smooth_width'] = 5.
                self.params['ind_width'] = 5.
                self.params['smooth_ps'] = 1.0
            elif:
        elif self.nyquist is not None:
            if (self.params['numax'] is not None and self.params['numax'] < 500.) or \
              (self.params['defaults'] is not None and self.params['defaults'] == 'low'):
                self.params['lower_ex'], self.params['lower_bg'] = 1., 1.
                self.params['upper_ex'], self.params['upper_bg'] = 1000., 1000.
            elif (self.params['numax'] is not None and self.params['numax'] >= 500.) or \
              (self.params['defaults'] is not None and self.params['defaults'] == 'high'):
                self.params['boxes'] = np.logspace(np.log10(50.), np.log10(500.), self.params['n_trials'])
                self.params['smooth_width'] = 20.
                self.params['ind_width'] = 20.
                self.params['smooth_ps'] = 2.5
                self.params['lower_ex'], self.params['lower_bg'] = 100., 100.
                self.params['upper_ex'], self.params['upper_bg'] = 8000., 8000.
            else:
                self.params['boxes'] = np.logspace(np.log10(0.5), np.log10(500.), self.params['n_trials'])
                self.params['smooth_width'], self.params['ind_width'] = 10., 10.
                self.params['lower_ex'], self.params['lower_bg'] = 1., 1.
                self.params['upper_ex'], self.params['upper_bg'] = 8000., 8000.
                self.params['smooth_ps'] = 1.5           
        """


    def load_data(self,):
        try:
            self._load_star()
        except utils.InputWarning as warning:
            print(warning.msg)
        except utils.InputError as error:
            print(error.msg)
            if os.path.exists(self.params['outdir']):
                try:
                    os.rmdir(self.params['outdir'])
                except OSError:
                    pass
            return False
        return True


    def process_star(self,):
        """Run pipeline

        Processes a given star with `pySYD`

        Methods
            - :mod:`pysyd.target.Target.estimate_parameters`
            - :mod:`pysyd.target.Target.derive_parameters`
            - :mod:`pysyd.target.Target.show_results`

        """
        self.params['results'], self.params['plotting'] = {}, {}
        self.estimate_parameters()
        self.derive_parameters()
        if self.params['test']:
            return self.params['results'].pop('parameters')
        self.show_results()


    def show_results(self, show=False, verbose=False,):
        """Show results

        Parameters
            show : bool, optional
                show output figures and text
            verbose : bool, optional
                turn on verbose output

        """
        # Print results
        if self.params['verbose']:
            utils._verbose_output(self)
            if self.params['show']:
                print('- displaying figures -')
        plots.make_plots(self)
        if self.params['cli'] and self.params['verbose'] and self.params['show']:
            input('- press [RETURN] to exit -\n')


##########################################################################################


    def _load_star(self, ps=False, lc=False,):
        """Input star data

        Load data in for a single star by first checking to see if the power spectrum exists
        and then loads in the time series data, which will compute a power spectrum in the
        event that there is not one

        Attributes
            lc : bool
                `True` if object has light curve
            ps : bool
                `True` if object has power spectrum

        Methods
            - :mod:`pysyd.target.Target.load_power_spectrum`
            - :mod:`pysyd.target.Target.load_time_series`

        Raises
            :mod:`pysyd.utils.InputError`
                if no data is found for a given target   

        """
        self.ps, self.lc, self.note, self.note2, self.warnings = False, False, '', '', '\n##### OTHER WARNING(S): #####\n'
        self.params['data'], self.params['plotting'], self.params['results'] = {}, {}, {}
        # Now done at beginning to make sure it only does this once per star
        if glob.glob(os.path.join(self.params['inpdir'],'%s*' % str(self.name))):
            self.note += '\n-----------------------------------------------------------\nTarget: %s\n-----------------------------------------------------------\n' % str(self.name)
            # Load PS first in case we need to calculate PS from LC
            self.load_power_spectrum()
            # Load light curve
            self.load_time_series()
        # CASE 4: NO LIGHT CURVE AND NO POWER SPECTRUM
        #     ->  cannot process, return user error
        if not self.ps:
            error = "\n\nERROR: no data found for target %s\n     -> please make sure you are in the correct\n        directory and try again!\n"%self.name
            raise utils.InputError(error)
        if self.params['verbose']:
            if self.warnings != '\n##### OTHER WARNING(S): #####\n' and self.params['warnings']:
                print(self.warnings)
            print(self.note)


    def load_power_spectrum(self, long=10**6,):
        r"""Load power spectrum
    
        Loads in available power spectrum and computes relevant information -- also checks
        for time series data and will raise a warning if there is none since it will have
        to assume a :term:`critically-sampled power spectrum`

        Attributes
            note : str, optional
                verbose output
            ps : bool
                `True` if star ID has an available (or newly-computed) power spectrum

        Yields
            frequency, power : numpy.ndarray, numpy.ndarray
                input power spectrum
            freq_os, pow_os : numpy.ndarray, numpy.ndarray
                copy of the oversampled power spectrum (i.e. `frequency` & `power`)
            freq_cs, pow_cs : numpy.ndarray, numpy.ndarray
                copy of the critically-sampled power spectrum (i.e. `frequency` & `power`) 
                iff the :term:`oversampling_factor<--of, --over, --oversample>` is provided, 
                otherwise these arrays are just copies of `freq_os` & `pow_os` since this factor
                isn't known and needs to be assumed

        Raises
            :mod:`pysyd.utils.InputWarning`
                if no information or time series data is provided (i.e. *has* to assume the PS is critically-sampled) 

        """
        # Try loading the power spectrum
        if os.path.exists(os.path.join(self.params['inpdir'], '%s_PS.txt' % str(self.name))):
            self.ps = True
            self.frequency, self.power = self.load_file(os.path.join(self.params['inpdir'], '%s_PS.txt' % str(self.name)))
            self.params['data'].update({'freq_orig':np.copy(self.frequency),'pow_orig':np.copy(self.power)})
            self.note2 += '# POWER SPECTRUM: %d lines of data read\n'%len(self.frequency)
            if len(self.frequency) >= long:
                self.warnings += '#             - PS is large and will slow down the software\n'
            # Only use provided oversampling factor if there is no light curve to calculate it from 
            # CASE 3: POWER SPECTRUM AND NO LIGHT CURVE
            #     ->  assume critically-sampled power spectrum
            if not os.path.exists(os.path.join(self.params['inpdir'], '%s_LC.txt' % str(self.name))):
                if self.params['oversampling_factor'] is None:
                    self.warnings += '#             - using PS with no additional information\n# **assuming critically-sampled PS**\n'
                    if self.params['mc_iter'] > 1:
                        self.warnings += '#              **uncertainties may not be reliable if the PS is not critically-sampled**\n'
                    self.params['oversampling_factor'] = 1
                self.frequency, self.power = self.fix_data(self.frequency, self.power)
                self.params['data'].update({'freq_fin':np.copy(self.frequency),'pow_fin':np.copy(self.power)})
                self.freq_os, self.pow_os = np.copy(self.frequency), np.copy(self.power)
                self.freq_cs = np.array(self.frequency[self.params['oversampling_factor']-1::self.params['oversampling_factor']])
                self.pow_cs = np.array(self.power[self.params['oversampling_factor']-1::self.params['oversampling_factor']])
                self.baseline = 1./((self.freq_cs[1]-self.freq_cs[0])*10**-6.)
                self.tau_upper = self.baseline/2.
                self.params['data'].update({'freq_over':np.copy(self.freq_os),'pow_over':np.copy(self.pow_os),
                                            'freq_crit':np.copy(self.freq_cs),'pow_crit':np.copy(self.pow_cs),})


    def load_time_series(self, save=True, stitch=False, oversampling_factor=None,):
        r"""Load light curve
        
        Loads in time series data and calculates relevant parameters like the 
        cadence and nyquist frequency

        Parameters
            save : bool
                save all data products
            stitch : bool
                "stitches" together time series data with large "gaps"
            oversampling_factor : int, optional
                oversampling factor of input power spectrum

        Attributes
            note : str, optional
                verbose output
            lc : bool
                `True` if star ID has light curve data available
            cadence : int
                median cadence of time series data (:math:`\\Delta t`)
            nyquist : float
                nyquist frequency of the power spectrum (calculated from time series cadence)
            baseline : float
                total time series duration (:math:`\\Delta T`)
            tau_upper : float
                upper limit of the granulation time scales, which is set by the total duration
                of the time series (divided in half)

        Yields
            time, flux : numpy.ndarray, numpy.ndarray
                input time series data
            frequency, power : numpy.ndarray, numpy.ndarray
                newly-computed frequency array using the time series array (i.e. `time` & `flux`)
            freq_os, pow_os : numpy.ndarray, numpy.ndarray
                copy of the oversampled power spectrum (i.e. `frequency` & `power`)
            freq_cs, pow_cs : numpy.ndarray, numpy.ndarray
                copy of the critically-sampled power spectrum (i.e. `frequency` & `power`)

        Raises
            :mod:`pysyd.utils.InputWarning`
                if the oversampling factor provided is different from that computed from the
                time series data and power spectrum
            :mod:`pysyd.utils.InputError`
                if the oversampling factor calculated from the time series data and power 
                spectrum is not an integer


        """
        self.nyquist = None
        # Try loading the light curve
        if os.path.exists(os.path.join(self.params['inpdir'], '%s_LC.txt' % str(self.name))):
            self.lc = True
            self.time, self.flux = self.load_file(os.path.join(self.params['inpdir'], '%s_LC.txt' % str(self.name)))
            self.params['data'].update({'time_orig':np.copy(self.time),'flux_orig':np.copy(self.flux),})
            self.time -= min(self.time)
            self.cadence = int(round(np.nanmedian(np.diff(self.time)*24.0*60.0*60.0),0))
            self.nyquist = 10**6./(2.0*self.cadence)
            self.baseline = (max(self.time)-min(self.time))*24.*60.*60.
            self.tau_upper = self.baseline/2.
            self.note += '# LIGHT CURVE: %d lines of data read\n# Time series cadence: %d seconds\n'%(len(self.time),self.cadence)
            # Stitch light curve together before attempting to compute a PS
            if self.params['stitch']:
                self.stitch_data()
            self.params['data'].update({'time_fin':np.copy(self.time),'flux_fin':np.copy(self.flux)})
            # Compute a PS if there is not one w/ the option to save to inpdir for next time
            if not self.ps:
                # CASE 2: LIGHT CURVE AND NO POWER SPECTRUM
                #     ->  compute power spectrum and set oversampling factor
                self.ps, self.params['oversampling_factor'] = True, 5
                self.frequency, self.power = self.compute_spectrum(oversampling_factor=self.params['oversampling_factor'], store=True)
                if self.params['save']:
                    utils._save_file(self.frequency, self.power, os.path.join(self.params['inpdir'], '%s_PS.txt'%self.name), overwrite=self.params['overwrite'])
                self.note += '# NEWLY COMPUTED POWER SPECTRUM has length of %d\n'%int(len(self.frequency)/5)
            else:
                # CASE 1: LIGHT CURVE AND POWER SPECTRUM
                #     ->  calculate oversampling factor from time series and compare
                oversampling_factor = int(np.ceil((1./((max(self.time)-min(self.time))*0.0864))/(self.frequency[1]-self.frequency[0])))
                if self.params['oversampling_factor'] is not None:
                    if oversampling_factor != self.params['oversampling_factor'] and self.params['warnings']:
                        raise utils.InputWarning("\nWARNING: \ncalculated vs. provided oversampling factor do NOT match\n")
                else:
                    if not float('%.2f'%oversampling_factor).is_integer():
                        raise utils.InputError("\nERROR: \nthe calculated oversampling factor is not an integer\nPlease check the input data and try again\n")
                    else:
                        self.params['oversampling_factor'] = oversampling_factor  
                self.frequency, self.power = self.fix_data(self.frequency, self.power)
                self.params['data'].update({'freq_fin':np.copy(self.frequency),'pow_fin':np.copy(self.power)})
            self.note += self.note2
            self.freq_os, self.pow_os = np.copy(self.frequency), np.copy(self.power)
            self.freq_cs = np.array(self.frequency[self.params['oversampling_factor']-1::self.params['oversampling_factor']])
            self.pow_cs = np.array(self.power[self.params['oversampling_factor']-1::self.params['oversampling_factor']])
            self.params['data'].update({'freq_over':np.copy(self.freq_os),'pow_over':np.copy(self.pow_os),
                                        'freq_crit':np.copy(self.freq_cs),'pow_crit':np.copy(self.pow_cs),})
            if self.params['oversampling_factor'] != 1:
                self.note += '# PS oversampled by a factor of %d'%self.params['oversampling_factor']
            else:
                self.note += '# PS is critically-sampled'
            self.note += '\n# PS resolution: %.6f muHz'%(self.freq_cs[1]-self.freq_cs[0])


    def load_file(self, path):
        r"""Load text file
    
        Load a light curve or a power spectrum from a basic 2xN txt file
        and stores the data into the `x` (independent variable) and `y`
        (dependent variable) arrays, where N is the length of the series

        Parameters
            path : str
                the file path of the data file

        Returns
            x, y : numpy.ndarray, numpy.ndarray
                the independent and dependent variables, respectively

        """
        # Open file
        with open(path, "r") as f:
            lines = f.readlines()
        # Set values
        x = np.array([float(line.strip().split()[0]) for line in lines])
        y = np.array([float(line.strip().split()[1]) for line in lines])
        return x, y


    def stitch_data(self, gap=20):
        r"""Stitch light curve

        For computation purposes and for special cases that this does not affect the integrity of the results,
        this module 'stitches' a light curve together for time series data with large gaps. For stochastic p-mode
        oscillations, this is justified if the lifetimes of the modes are smaller than the gap. 

        Parameters
            gap : int
                how many consecutive missing cadences are considered a 'gap'
      
        Attributes
            time : numpy.ndarray
                original time series array to correct
            new_time : numpy.ndarray
                the corrected time series array

        Raises
            :mod:`pysyd.utils.InputWarning`
                when using this method since it's technically not a great thing to do

        .. warning::
            USE THIS WITH CAUTION. This is technically not a great thing to do for primarily
            two reasons:
             #. you lose phase information *and* 
             #. can be problematic if mode lifetimes are shorter than gaps (i.e. more evolved stars)


        .. note::
            temporary solution for handling very long gaps in TESS data -- still need to
            figure out a better way to handle this


        """
        self.warnings += '#             - using stitch_data module - which is dodgy\n'
        self.new_time = np.copy(self.time)
        for i in range(1,len(self.new_time)):
            if (self.new_time[i]-self.new_time[i-1]) > float(self.params['gap'])*(self.cadence/24./60./60.):
                self.new_time[i] = self.new_time[i-1]+(self.cadence/24./60./60.)
        self.time = np.copy(self.new_time)


    def compute_spectrum(self, oversampling_factor=1, store=False):
        """Compute power spectrum

        **NEW** function to calculate a power spectrum given time series data, which will
        normalize the power spectrum to spectral density according to Parseval's theorem

        Parameters
            oversampling_factor : int, default=1
                the oversampling factor to use when computing the power spectrum 
            store : bool, default=False
                if `True`, it will store the original data arrays for plotting purposes later

        Yields
            frequency, power : numpy.ndarray, numpy.ndarray
                power spectrum computed from the input time series data (i.e. `time` & `flux`)
                using the :mod:`astropy.timeseries.LombScargle` module

        Returns
            frequency, power : numpy.ndarray, numpy.ndarray
                the newly-computed and normalized power spectrum (in units of :math:`\\rm \\mu Hz` vs. :math:`\\rm ppm^{2} \\mu Hz^{-1}`)


        .. important::

            If you are unsure if your power spectrum is in the proper units, we recommend
            using this new module to compute and normalize for you. This will ensure the
            accuracy of the results.

        .. todo::

           add equation for conversion


        """
        freq, pow = lomb(self.time, self.flux).autopower(method='fast', samples_per_peak=oversampling_factor, maximum_frequency=self.nyquist)
        # convert frequency array into proper units
        freq *= (10.**6/(24.*60.*60.))
        # normalize PS according to Parseval's theorem
        psd = 4.*pow*np.var(self.flux*1e6)/(np.sum(pow)*(freq[1]-freq[0]))
        frequency, power = self.fix_data(freq, psd)
        if store:
            self.params['data'].update({'freq_orig':np.copy(freq),'pow_orig':np.copy(psd),
                                        'freq_fin':np.copy(frequency),'pow_fin':np.copy(power)})
        return frequency, power


    def fix_data(self, frequency, power, save=True, kep_corr=False, ech_mask=None, lower_ech=None,
                 upper_ech=None):
        r"""Fix frequency domain data

        Applies frequency-domain tools to power spectra to "fix" (i.e. manipulate) the data. 
        If no available options are used, it will simply return copies of the original arrays

        Parameters
            save : bool
                save all data products
            kep_corr : bool
                correct for known *Kepler* short-cadence artefacts
            ech_mask : List[lower_ech,upper_ech]
                corrects for mixed modes if not `None`
            lower_ech : float
                folded lower frequency limit (~[0,dnu])
            upper_ech : float
                folded upper frequency limit (~[0,dnu])
            frequency, power : numpy.ndarray, numpy.ndarray
                input power spectrum to be corrected 

        Methods
            - :mod:`pysyd.target.Target.remove_artefact`
                mitigate known *Kepler* artefacts
            - :mod:`pysyd.target.Target.whiten_mixed` 
                mitigate mixed modes
	   
        Returns
            frequency, power : numpy.ndarray, numpy.ndarray
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

    def _get_seed(self):
        if self.params['seed'] is None:
            if os.path.exists(self.params['info']):
                df = pd.read_csv(self.params['info'])
                stars = [str(each) for each in df.star.values.tolist()]
                # check if star is in file and if not, adds it and the seed for reproducibility
                if str(self.name) in stars and 'seed' in df.columns.values.tolist():
                    idx = stars.index(str(self.name))
                    if not np.isnan(float(df.loc[idx,'seed'])):
                        self.params['seed'] = int(df.loc[idx,'seed'])
            # if still None, generate new seed
            if self.params['seed'] is None:
                self._new_seed()
        if self.params['save']:
            self._save_seed()

    def _new_seed(self, lower=1, upper=10**7):
        """Set seed
    
        For *Kepler* targets that require a correction via CLI (--kc), a random seed is generated
        from U~[1,10^7] and stored in stars_info.csv for reproducible results in later runs.

        Parameters
            lower : int, default=1
                lower limit for random seed value
            upper : int, default=:math:`10^{7}`
                arbitrary upper limit for random seed value 

        .. important:: this is now implemented for every star for every run (no longer used
                        just for correcting)


        """
        self.params['seed'] = int(np.random.randint(lower,high=upper))

    def _save_seed(self):
        if os.path.exists(self.params['info']):
            df = pd.read_csv(self.params['info'])
            stars = [str(each) for each in df.star.values.tolist()]
            if str(self.name) in stars:
                idx = stars.index(str(self.name))
                df.loc[idx,'seed'] = int(self.params['seed'])
            else:
                idx = len(df)
                df.loc[idx,'star'], df.loc[idx,'seed'] = str(self.name), int(self.params['seed'])
        else:
            df = pd.DataFrame(columns=['star','seed'])
            df.loc[0,'star'], df.loc[0,'seed'] = str(self.name), int(self.params['seed'])
        df.to_csv(self.params['info'], index=False)


    def remove_artefact(self, freq, pow, lcp=1.0/(29.4244*60*1e-6), 
                        lf_lower=[240.0,500.0], lf_upper=[380.0,530.0], 
                        hf_lower = [4530.0,5011.0,5097.0,5575.0,7020.0,7440.0,7864.0],
                        hf_upper = [4534.0,5020.0,5099.0,5585.0,7030.0,7450.0,7867.0],):
        """Remove *Kepler* artefacts
    
        Module to remove artefacts found in *Kepler* data by replacing known frequency 
        ranges with simulated noise 

        Parameters
            lcp : float
                long cadence period (in Msec)
            lf_lower : List[float]
                lower limits of low-frequency artefacts
            lf_upper : List[float]
                upper limits of low-frequency artefacts
            hf_lower : List[float]
                lower limit of high frequency artefact
            hf_upper : List[float]
                upper limit of high frequency artefact
            freq, pow : numpy.ndarray, numpy.ndarray
                input data that needs to be corrected 

	    
        Returns
            frequency, power : numpy.ndarray, numpy.ndarray
                copy of the corrected power spectrum

        .. note::

            Known *Kepler* artefacts include:
             #. long-cadence harmonics
             #. sharp, high-frequency artefacts (:math:`\\rm >5000 \\mu Hz`)
             #. low frequency artefacts 250-400 muHz (mostly present in Q0 and Q3 data)


        """
        frequency, power = np.copy(freq), np.copy(pow)
        resolution = frequency[1]-frequency[0]
        self._get_seed()
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
        """Whiten mixed modes
    
        Module to help reduce the effects of mixed modes random white noise in place of 
        :math:`\\ell=1` for subgiants with mixed modes to better constrain the large 
        frequency separation

        Parameters
            dnu : float, default=None
                the so-called large frequency separation to fold the PS to
            lower_ech : float, default=None
                lower frequency limit of mask to "whiten"
            upper_ech : float, default=None
                upper frequency limit of mask to "whiten"
            notching : bool, default=False
                if `True`, uses notching instead of generating white noise
            freq, pow : numpy.ndarray, numpy.ndarray
                input data that needs to be corrected 
            folded_freq : numpy.ndarray
                frequency array modulo dnu (i.e. folded to the large separation, :math:`\\Delta\\nu`)

        Returns
            frequency, power : numpy.ndarray, numpy.ndarray
                copy of the corrected power spectrum

        """
        frequency, power = np.copy(freq), np.copy(pow)
        self._get_seed()
        # Estimate white noise
        if not self.params['notching']:
            white = np.mean(power[(frequency >= max(frequency)-100.0)&(frequency <= max(frequency)-50.0)])
        else:
            white = min(power[(frequency >= max(frequency)-100.0)&(frequency <= max(frequency)-50.0)])
        # Take the provided dnu and "fold" the power spectrum
        folded_freq = np.copy(frequency)%self.params['dnu']
        mask = np.ma.getmask(np.ma.masked_inside(folded_freq, self.params['ech_mask'][0], self.params['ech_mask'][1]))
        # Set seed for reproducibility purposes
        np.random.seed(int(self.params['seed']))
        # Makes sure the mask is not empty
        if np.sum(mask) != 0:
            if self.params['notching']:
                power[mask] = white
            else:
                power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0
        return np.copy(frequency), np.copy(power)


##########################################################################################


    def estimate_parameters(self, estimate=True,):
        """Estimate parameters

        Calls all methods related to the first module 

        Parameters
            estimate : bool, default=True
                if numax is already known, this will automatically be skipped since it is not needed

        Methods
            - :mod:`pysyd.target.Target.initial_estimates`
            - :mod:`pysyd.target.Target.estimate_numax`
            - :mod:`pysyd.utils._save_estimates`

        """
        if 'results' not in self.params:
            self.params['results'] = {}
        if 'plotting' not in self.params:
            self.params['plotting'] = {}
        if self.params['estimate']:
            # get initial values and fix data
            self.initial_estimates()
            # execute function
            self.estimate_numax()
            # save results
            self = utils._save_estimates(self)


    def initial_estimates(self, lower_ex=1.0, upper_ex=8000.0, max_trials=6,):
        """Initial estimates
    
        Prepares data and parameters associated with the first module that identifies 
        solar-like oscillations and estimates :term:`numax`

        Parameters
            lower_ex : float
                the lower frequency limit of the PS used to estimate numax
            upper_ex : float
                the upper frequency limit of the PS used to estimate numax
            max_trials : int
	               (arbitrary) maximum number of "guesses" or trials to perform to estimate numax

        Attributes
            frequency, power : numpy.ndarray, numpy.ndarray
                copy of the entire oversampled (or critically-sampled) power spectrum (i.e. `freq_os` & `pow_os`) 
            freq, pow : numpy.ndarray, numpy.ndarray
                copy of the entire oversampled (or critically-sampled) power spectrum (i.e. `freq_os` & `pow_os`) after applying the mask~[lower_ex,upper_ex]
            module : str, default='parameters'
                which part of the pipeline is currently being used


        """
        self.module = 'estimates'
        self.params['plotting'][self.module], self.params['results'][self.module] = {}, {}
        if self.lc:
            self.params['plotting'][self.module].update({'time':np.copy(self.params['data']['time_fin']),
                                                         'flux':np.copy(self.params['data']['flux_fin'])})
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
        self.params['plotting'][self.module].update({'freq':np.copy(self.freq),'pow':np.copy(self.pow)})
        if self.params['n_trials'] > max_trials:
            self.params['n_trials'] = max_trials
        if (self.params['numax'] is not None and self.params['numax'] <= 500.) or (self.nyquist is not None and self.nyquist <= 300.) or (max(self.frequency) < 300.):
            self.params['boxes'] = np.logspace(np.log10(0.5), np.log10(25.), self.params['n_trials'])
        else:
            self.params['boxes'] = np.logspace(np.log10(50.), np.log10(500.), self.params['n_trials'])


    def estimate_numax(self, binning=0.005, bin_mode='mean', smooth_width=20.0, ask=False,):
        """Estimate numax

        Automated routine to identify power excess due to solar-like oscillations and estimate
        an initial starting point for :term:`numax` (:math:`\\nu_{\\mathrm{max}}`)

        Parameters
            binning : float
                logarithmic binning width (i.e. evenly spaced in log space)
            bin_mode : {'mean', 'median', 'gaussian'}
                mode to use when binning
            smooth_width: float
                box filter width (in :math:`\\rm \\mu Hz`) to smooth power spectrum
            ask : bool
                If `True`, it will ask which trial to use as the estimate for numax

        Attributes
            bin_freq, bin_pow : numpy.ndarray, numpy.ndarray
                copy of the power spectrum (i.e. `freq` & `pow`) binned equally in logarithmic space
            smooth_freq, smooth_pow : numpy.ndarray, numpy.ndarray
                copy of the binned power spectrum (i.e. `bin_freq` & `bin_pow`) binned equally in linear space -- *yes, this is doubly binned intentionally*
            freq, interp_pow : numpy.ndarray, numpy.ndarray
                the smoothed power spectrum (i.e. `smooth_freq` & `smooth_pow`) interpolated back to the original frequency array (also referred to as "crude background model")
            freq, bgcorr_pow : numpy.ndarray, numpy.ndarray
                approximate :term:`background-corrected power spectrum` computed by dividing the original PS (`pow`) by the interpolated PS (`interp_pow`) 

        Methods
            - :mod:`pysyd.target.Target.collapsed_acf`


        """
        # Smooth the power in log-space
        self.bin_freq, self.bin_pow, _ = utils._bin_data(self.freq, self.pow, width=self.params['binning'], log=True, mode=self.params['bin_mode'])
        # Smooth the power in linear-space
        self.smooth_freq, self.smooth_pow, _ = utils._bin_data(self.bin_freq, self.bin_pow, width=self.params['smooth_width'])
        if self.params['verbose']:
            print('-----------------------------------------------------------\nPS binned to %d datapoints\n\nNumax estimates\n---------------' % len(self.smooth_freq))
        # Mask out frequency values that are lower than the smoothing width to avoid weird looking fits
        if min(self.freq) < self.params['smooth_width']:
            mask = (self.smooth_freq >= self.params['smooth_width'])
            self.smooth_freq, self.smooth_pow = self.smooth_freq[mask], self.smooth_pow[mask]
        s = InterpolatedUnivariateSpline(self.smooth_freq, self.smooth_pow, k=1)
        # Interpolate and divide to get a crude background-corrected power spectrum
        self.interp_pow = s(self.freq)
        self.bgcorr_pow = self.pow/self.interp_pow
        self.params['plotting'][self.module].update({'bin_freq':np.copy(self.bin_freq),
                                                     'bin_pow':np.copy(self.bin_pow),
                                                     'interp_pow':np.copy(self.interp_pow),
                                                     'bgcorr_pow':np.copy(self.bgcorr_pow)})
        # Collapsed ACF to find numax
        self.collapsed_acf()
        self.params['best'] = self.params['compare'].index(max(self.params['compare']))+1
        # Select trial that resulted with the highest SNR detection
        if not self.params['ask']:
            if self.params['verbose']:
                print('Selecting model %d' % self.params['best'])
        # Or ask which estimate to use
        else:
            self = plots.select_trial(self)


    def collapsed_acf(self, n_trials=3, step=0.25, max_snr=100.0,):
        """Collapsed ACF

        Computes a collapsed autocorrelation function (ACF) using n different box sizes in
        n different trials (i.e. `n_trials`)

        Parameters
            n_trials : int
                the number of trials to run
            step : float
                fractional step size to use for the collapsed ACF calculation
            max_snr : float
                the maximum signal-to-noise of the estimate (this is primarily for plot formatting)


        """
        self.params['compare'] = []
        # Computes a collapsed ACF using different "box" (or bin) sizes
        for b, box in enumerate(self.params['boxes']):
            self.params['results'][self.module][b+1], self.params['plotting'][self.module][b] = {}, {}
            start, snr, cumsum, md = 0, 0.0, [], []
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
            self.params['plotting'][self.module].update({b:{'x':np.array(md),'y':np.array(csum),'maxx':md[idx],'maxy':csum[idx]}})
            # Fit Gaussian to get estimate value for numax
            try:
                best_vars, _ = curve_fit(models.gaussian, np.array(md), np.array(csum), p0=[np.median(csum), 1.0-np.median(csum), md[idx], self.constants['width_sun']*(md[idx]/self.constants['numax_sun'])], maxfev=5000, bounds=((-np.inf,-np.inf,1,-np.inf),(np.inf,np.inf,np.inf,np.inf)),)
            except Exception as _:
                self.params['plotting'][self.module][b].update({'good_fit':False,'fitx':np.linspace(min(md), max(md), 10000)})
            else:
                self.params['plotting'][self.module][b].update({'good_fit':True,})
                self.params['plotting'][self.module][b].update({'fitx':np.linspace(min(md), max(md), 10000),'fity':models.gaussian(np.linspace(min(md), max(md), 10000), *best_vars)})
                snr = max(self.params['plotting'][self.module][b]['fity'])/np.absolute(best_vars[0])
                if snr > max_snr:
                    snr = max_snr
                self.params['results'][self.module][b+1].update({'value':best_vars[2],'snr':snr})
                self.params['plotting'][self.module][b].update({'value':best_vars[2],'snr':snr})
                if self.params['verbose']:
                    print('Estimate %d: %.2f +/- %.2f'%(b+1, best_vars[2], np.absolute(best_vars[3])/2.0))
                    print('S/N: %.2f' % snr)
            self.params['compare'].append(snr)


    def check_numax(self, columns=['numax', 'dnu', 'snr']):
        r"""Check :math:`\\rm \\nu_{max}`
    
        Checks if there is an initial starting point or estimate for :term:`numax`

        Parameters
            columns : List[str]
                saved columns if the estimate_numax() function was run

        Raises
            utils.InputError
                if an invalid value was provided as input for numax
            utils.ProcessingError
                if it still cannot find any estimate for :term:`numax`


        """
        # THIS MUST BE FIXED TOO
        # Check if numax was provided as input
        if self.params['numax'] is not None:
            if np.isnan(float(self.params['numax'])):
                raise utils.InputError("ERROR: invalid value for numax")
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
                    raise utils.ProcessingError("\nERROR: invalid value for numax\n")
            else:
                # Raise error
                raise utils.ProcessingError("\nERROR: no numax provided for global fit\n")


##########################################################################################


    def derive_parameters(self, mc_iter=1,):
        """Derive parameters

        Main function to derive the background and global asteroseismic parameters (including
        uncertainties when relevant), which does everything from finding the initial estimates 
        to plotting/saving results

        Parameters
            mc_iter : int, default=1
                the number of iterations to run

        Methods
            :mod:`pysyd.target.Target.check_numax`
                first checks to see if there is a valid estimate or input value provides
                for numax
            :mod:`pysyd.target.Target.initial_parameters`
                if so, it will estimate the rest of the initial guesses required for the
                background and global fitting (primarily using solar scaling relations)
            :mod:`pysyd.target.Target.first_step`
                the first iteration determines the best-fit background model and global
                properties
            :mod:`pysyd.target.Target.get_samples`
                bootstrap uncertainties by attempting to recover the parameters from the
                first step


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
        utils._save_parameters(self)


    def initial_parameters(self, module='parameters', lower_bg=1.0, upper_bg=8000.0,):
        r"""Initial guesses
    
        Estimates initial guesses for background components (i.e. timescales and amplitudes) using
        solar scaling relations. This resets the power spectrum and has its own independent filter 
        or bounds (via [lower_bg, upper_bg]) to use for this subroutine

        Parameters
            lower_bg : float
                lower frequency limit of PS to use for the background fit
            upper_bg : float
                upper frequency limit of PS to use for the background fit

        Attributes
            frequency, power : numpy.ndarray, numpy.ndarray
                copy of the entire oversampled (or critically-sampled) power spectrum (i.e. `freq_os` & `pow_os`) 
            frequency, random_pow : numpy.ndarray, numpy.ndarray
                copy of the entire oversampled (or critically-sampled) power spectrum (i.e. `freq_os` & `pow_os`) after applying the mask~[lower_bg,upper_bg]
            module : str
                which part of the pipeline is currently being used
            i : int
                iteration number, starts at 0

        Methods
            :mod:`pysyd.target.Target.solar_scaling`
                uses multiple solar scaling relations to determnie accurate initial guesses for
                many of the derived parameters 

        .. warning::

            This is typically sufficient for most stars but may affect evolved stars and
            need to be adjusted!

        """
        self.module = 'parameters'
        self.params['plotting'][self.module], self.params['results'][self.module] = {}, {}
        if self.lc:
            self.params['plotting'][self.module].update({'time':np.copy(self.params['data']['time_fin']),
                                                         'flux':np.copy(self.params['data']['flux_fin'])})
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
        self.params['plotting'][self.module].update({'frequency':np.copy(self.frequency),
                                                     'random_pow':np.copy(self.power)})
        # Get other relevant initial conditions
        self.i = 0
        for parameter in utils.get_dict(type="columns")['params']:
            self.params['results'][self.module].update({parameter:[]})
        # Use scaling relations from sun to get starting points
        self.solar_scaling()
        if self.params['verbose']:
            print('-----------------------------------------------------------\nGLOBAL FIT\n-----------------------------------------------------------')


    def solar_scaling(self, numax=None, scaling='tau_sun_single', max_laws=3, ex_width=1.0,
                      lower_ps=None, upper_ps=None,):
        r"""Initial values
        
        Using the initial starting value for :math:`\\rm \\nu_{max}`, estimates the rest of
        the parameters needed for *both* the background and global fits. Uses scaling relations 
        from the Sun to:
         #. estimate the width of the region of oscillations using numax
         #. guess starting values for granulation time scales

        Parameters
            numax : float, default=None
                provide initial value for numax to bypass the first module
	           scaling : str, default='tau_sun_single'
	               which solar scaling relation to use
            max_laws : int, default=3
                the maximum number of resolvable Harvey-like components
            ex_width : float, default=1.0
                fractional width to use for power excess centered on :term:`numax`
            lower_ps : float, default=None
                lower bound of power excess to use for :term:`ACF` [in :math:`\\rm \mu Hz`]
            upper_ps : float, default=None
                upper bound of power excess to use for :term:`ACF` [in :math:`\\rm \mu Hz`]

        Attributes
            converge : bool, default=True
                `True` if all fitting converges


        """
        if self.params['dnu'] is None:
            self.params['dnu'] = utils.delta_nu(self.params['numax'])
        # Use scaling relations to estimate width of oscillation region to mask out of the background fit
        width = self.constants['width_sun']*(self.params['numax']/self.constants['numax_sun'])
        maxpower = [self.params['numax']-(width*self.params['ex_width']), self.params['numax']+(width*self.params['ex_width'])]
        if self.params['lower_ps'] is not None:
            maxpower[0] = self.params['lower_ps']
        if self.params['upper_ps'] is not None:
            maxpower[1] = self.params['upper_ps']
        self.params['ps_mask'] = [maxpower[0], maxpower[1]]
        # Use scaling relation for granulation timescales from the sun to get starting points
        scale = self.constants['numax_sun']/self.params['numax']
        # make sure interval is not empty
        if not list(self.frequency[(self.frequency>=self.params['ps_mask'][0])&(self.frequency<=self.params['ps_mask'][1])]):
            raise utils.PySYDInputError("ERROR: frequency region for power excess is null\nPlease specify an appropriate numax and/or frequency limits for the power excess (via --lp/--up)")
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
        self.converge = True
        # Save copies for plotting after the analysis
        self.params['nlaws'], self.params['a'] = len(self.params['mnu']), []
        self.params['plotting'][self.module].update({'exp_numax':self.params['numax'],
                                                     'nlaws_orig':len(self.params['mnu']),
                                                     'b_orig':np.copy(self.params['b'])})


    def first_step(self, background=True, globe=True,):
        r"""First step

        Processes a given target for the first step, which has extra steps for each of the two 
        main parts of this method (i.e. background model and global fit):
         #. **background model:** the automated best-fit model selection is only performed in the
            first step, the results which are saved for future purposes (including the 
            background-corrected power spectrum)
         #. **global fit:** while the :term:`ACF` is computed for every iteration, a mask is
            created in the first step to prevent the estimate for dnu to latch on to a different 
            (i.e. incorrect) peak, since this is a multi-modal parameter space

        Parameters
            background : bool
                run the automated background-fitting routine
            globe : bool
                perform global asteroseismic analysis (really only relevant if interested in the background model *only*)

        Methods
            :mod:`pysyd.target.Target.estimate_background`
                estimates the amplitudes/levels of both correlated and frequency-independent noise
                properties from the input power spectrum
            :mod:`pysyd.target.Target.model_background`
                automated best-fit background model selection that is a summed contribution of
                various white + red noise componenets
            :mod:`pysyd.target.Target.global_fit`
                after correcting for the best-fit background model, this derives the :term:`global asteroseismic parameters`

        .. seealso:: :mod:`pysyd.target.Target.single_step`

        """
        # Background corrections
        self.estimate_background()
        self.model_background()
        # Global fit
        if self.params['globe']:
            # global fit
            self.global_fit()
            if self.params['verbose'] and self.params['mc_iter'] > 1:
                # Set seed for reproducibility
                self._get_seed()
                print('-----------------------------------------------------------\nSampling routine (using seed=%d):' % int(self.params['seed']))


    def single_step(self,):
        r"""Single step

        Similar to the first step, this function calls the same methods but uses the selected best-fit
        background model from the first step to estimate the parameters

        Attributes
            converge : bool
                removes any saved parameters if any fits did not converge (i.e. `False`)

        Returns
            converge : bool
                returns `True` if all relevant fits converged

        Methods
            - :mod:`pysyd.target.Target.estimate_background`
                estimates the amplitudes/levels of both correlated and frequency-independent noise
                properties from the input power spectrum
            - :mod:`pysyd.target.Target.get_background`
                unlike the first step, which iterated through several models and performed a
                best-fit model comparison, this only fits parameters from the selected model 
                in the first step
            - :mod:`pysyd.target.Target.global_fit`
                after correcting for the background model, this derives the :term:`global asteroseismic parameters`

        """
        self.converge = True
        self.random_pow = (np.random.chisquare(2, len(self.frequency))*self.power)/2.
        # Background corrections
        self.estimate_background()
        self.get_background()
        # Requires bg fit to converge before moving on
        if self.converge and self.params['globe']:
            self.global_fit()
        if not self.converge:
            for parameter in self.params['results'][self.module]:
                if len(self.params['results'][self.module][parameter]) > (self.i+1):
                    p = self.params['results'][self.module][parameter].pop(-1)
        return self.converge


    def get_samples(self,):
        r"""Get samples

        Estimates uncertainties for parameters by randomizing the power spectrum and
        attempting to recover the same parameters by calling the :mod:`pysyd.target.Target.single_step`

        Attributes
            frequency, power : numpy.ndarray, numpy.ndarray
                copy of the critically-sampled power spectrum (i.e. `freq_cs` & `pow_cs`) after applying the mask~[lower_bg,upper_bg]
            pbar : tqdm.tqdm, optional
                optional progress bar used with verbose output when running multiple iterations 

        .. note:: 

           all iterations except for the first step are applied to the :term:`critically-sampled power spectrum`
           and *not* the :term:`oversampled power spectrum`

        .. important:: if the verbose option is enabled, the `tqdm` package is required


        """
        # Switch to critically-sampled PS if sampling
        mask = np.ma.getmask(np.ma.masked_inside(self.freq_cs, self.params['bg_mask'][0], self.params['bg_mask'][1]))
        self.frequency, self.power = np.copy(self.freq_cs[mask]), np.copy(self.pow_cs[mask])
        self.params['resolution'] = self.frequency[1]-self.frequency[0]
        np.random.seed(int(self.params['seed']))
        if self.params['verbose']:
            from tqdm import tqdm 
            self.pbar = tqdm(total=self.params['mc_iter'])
            self.pbar.update(1)
        # iterate for x steps
        while self.i < self.params['mc_iter']:
            if self.single_step():
                self.i += 1
                if self.params['verbose']:
                    self.pbar.update(1)
                    if self.i == self.params['mc_iter']:
                        self.pbar.close()


    def estimate_background(self, ind_width=20.0,):
        """Background estimates

        Estimates initial guesses for the stellar background contributions for both the
        red and white noise components

        Parameters
            ind_width : float
                the independent average smoothing width (:math:`\\rm \\mu Hz`)

        Attributes
            bin_freq, bin_pow, bin_err : numpy.ndarray, numpy.ndarray, numpy.ndarray
                binned power spectrum using the :term:`ind_width<--iw, --indwidth>` bin size   

        Methods

        """
        # Bin power spectrum to model stellar background/correlated red noise components
        self.bin_freq, self.bin_pow, self.bin_err = utils._bin_data(self.frequency, self.random_pow, width=self.params['ind_width'], mode=self.params['bin_mode'])
        # Mask out region with power excess
        mask = np.ma.getmask(np.ma.masked_outside(self.bin_freq, self.params['ps_mask'][0], self.params['ps_mask'][1]))
        self.bin_freq, self.bin_pow, self.bin_err = self.bin_freq[mask], self.bin_pow[mask], self.bin_err[mask]
        # Estimate white noise level
        self.white_noise()
        # Get initial guesses for the optimization of the background model
        self.red_noise()


    def white_noise(self):
        """Estimate white noise

        Estimate the white noise level by taking the mean of the last 10% of the power spectrum

        """
        mask = (self.frequency > (max(self.frequency)-0.1*max(self.frequency)))&(self.frequency < max(self.frequency))
        self.params['noise'] = np.mean(self.random_pow[mask])


    def red_noise(self, box_filter=1.0, n_rms=20,):
        """Estimate red noise

        Estimates amplitudes of red noise components by using a smoothed version of the power
        spectrum with the power excess region masked out -- which will take the mean of a specified 
        number of points (via -nrms, default=20) for each Harvey-like component

        Parameters
            box_filter : float
                the size of the 1D box smoothing filter
            n_rms : int
                number of data points to average over to estimate red noise amplitudes 

        Attributes
            smooth_pow : numpy.ndarray
                smoothed power spectrum after applying the box filter


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
        """Model stellar background

        If nothing is fixed, this method iterates through :math:`2\\dot(n_{\\mathrm{laws}}+1)` 
        models to determine the best-fit background model due to stellar granulation processes,
        which uses a solar scaling relation to estimate the number of Harvey-like component(s) 
        (or `n_laws`)

        Parameters
            n_laws : int
                specify number of Harvey-like components to use in background fit 
            fix_wn : bool
                option to fix the white noise instead of it being an additional free parameter 
            basis : str
                which basis to use for background fitting, e.g. {a,b} parametrization **TODO: not yet operational**

        Methods
            - :mod:`pysyd.models.background`
            - :mod:`scipy.curve_fit`
            - :mod:`pysyd.models._compute_aic`
            - :mod:`pysyd.models._compute_bic`
            - :mod:`pysyd.target.Target.correct_background`

        Returns
            converge : bool
                returns `False` if background model fails to converge

        Raises
            utils.ProcessingError
                if this failed to converge on a single model during the first iteration


        """
        # save initial guesses for plotting purposes
        self.params['plotting'][self.module].update({'bin_freq':np.copy(self.bin_freq),'bin_pow':np.copy(self.bin_pow),
               'bin_err':np.copy(self.bin_err),'a_orig':np.copy(self.params['a']),
               'smooth_pow':np.copy(self.smooth_pow),'noise':self.params['noise'],})
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
                b, a = np.inf, np.inf
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
                    bounds[0].append(0.)
                    bounds[1].append(np.inf)
                    *guesses, = self.params['guesses'][-int(2*(n_free//2)+1):]
                self.params['bounds'].append(bounds)
                if n_free == 0:
                    self.params['paras'].append([])
                    b = models.Likelihood(self.bin_pow, np.ones_like(self.bin_pow)*self.params['noise'], n_free).compute_bic()
                    a = models.Likelihood(self.bin_pow, np.ones_like(self.bin_pow)*self.params['noise'], n_free).compute_aic()
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
                    else:
                        self.params['paras'].append(pars)
                        model = models.background(self.bin_freq, pars, noise=self.params['noise'])
                        b, a = models.Likelihood(self.bin_pow, model, n_free).compute_bic(), models.Likelihood(self.bin_pow, model, n_free).compute_aic()
                self.params['bic'].append(b)
                self.params['aic'].append(a)
                if self.params['verbose'] and np.isfinite(a) and np.isfinite(b):
                    note += '\n BIC = %.2f | AIC = %.2f'%(b, a)
                    print(note)
            # Did the fit converge 
            if np.isfinite(min(self.params[self.params['metric']])):
                self.correct_background()
            # Otherwise raise error that fit did not converge
            else:
                self.converge = False
                raise utils.ProcessingError("Background fit failed to converge for any models.\n\n We recommend disabling this feature using our boolean background flag ('-b' )")
        else:
            if self.params['verbose']:
                print('-----------------------------------------------------------\nWARNING: estimating global parameters from raw PS:')
            self.bg_corr = np.copy(self.random_pow)/self.params['noise']
            self.params['pars'] = ([self.params['noise']])
        # save final guesses for plotting purposes
        self.params['plotting'][self.module].update({'pars':self.params['pars'],})


    def correct_background(self, metric='bic'):
        r"""Correct background

        Corrects for the stellar background contribution in the power spectrum by *both*
        dividing and subtracting this out, which also saves copies of each (i.e. `bg_div`
        :term:`background-divided power spectrum` to :ref:`ID_BDPS.txt <library-output-files-bdps>`
        and `bg_sub` :term:`background-subtracted power spectrum to 
        :ref:`ID_BSPS.txt <library-output-files-bsps>`). After this is done, a copy of the
        BDPS is saved to `bg_corr` and used for dnu calculations and the echelle diagram.
        
        Parameters
            metric : str, default='bic'
                which metric to use (i.e. bic or aic) for model selection

        Attributes
            frequency, bg_div : numpy.ndarray, numpy.ndarray
                background-divded power spectrum (BDPS -> higher S/N for echelle diagram)
            frequency, bg_sub : numpy.ndarray, numpy.ndarray
                background-subtracted power spectrum (BSPS -> preserves mode amplitudes)
            frequency, bg_corr : numpy.ndarray, numpy.ndarray
                background-corrected power spectrum, which is a copy of the :term:`BDPS`


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
        # Save background-corrected power spectrum
        self.bg_div = self.random_pow/models.background(self.frequency, self.params['pars'], noise=self.params['noise'])
        if self.params['save']:
            utils._save_file(self.frequency, self.bg_div, os.path.join(self.params['path'], '%s_BDPS.txt'%self.name), overwrite=self.params['overwrite'])
        self.bg_sub = self.random_pow-models.background(self.frequency, self.params['pars'], noise=self.params['noise'])
        if self.params['save']:
            utils._save_file(self.frequency, self.bg_sub, os.path.join(self.params['path'], '%s_BSPS.txt'%self.name), overwrite=self.params['overwrite'])
        self.params['plotting'][self.module].update({'models':self.params['models'],
                                                     'model':self.params['selected'],
                                                     'paras':self.params['paras'],
                                                     'aic':self.params['aic'],
                                                     'bic':self.params['bic'],})
        # For the rest of the calculations, we'll use the background-divided power spectrum
        self.bg_corr = np.copy(self.bg_div)
        # Create appropriate keys for star based on best-fit model
        for n in range(self.params['nlaws']):
            self.params['results'][self.module]['tau_%d'%(n+1)] = []
            self.params['results'][self.module]['sigma_%d'%(n+1)] = []
        if not self.params['fix_wn']:
            self.params['results'][self.module]['white'] = []
        # Save the final values
        for n in range(self.params['nlaws']):
            self.params['results'][self.module]['tau_%d'%(n+1)].append(self.params['pars'][2*n]*10.**6)
            self.params['results'][self.module]['sigma_%d'%(n+1)].append(self.params['pars'][2*n+1])
        if not self.params['fix_wn']:
            self.params['results'][self.module]['white'].append(self.params['pars'][-1])


    def get_background(self):
        """Get background

        Attempts to recover background model parameters in later iterations by using the
        :mod:`scipy.curve_fit` module using the same best-fit background model settings 

        Returns
            converge : bool
                returns `False` if background model fails to converge
     
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
                self.converge = False
            else:
                self.bg_corr = self.random_pow/models.background(self.frequency, self.params['pars'], noise=self.params['noise'])
                # save final values for Harvey components
                for n in range(self.params['nlaws']):
                    self.params['results'][self.module]['tau_%d'%(n+1)].append(self.params['pars'][2*n]*10.**6)
                    self.params['results'][self.module]['sigma_%d'%(n+1)].append(self.params['pars'][2*n+1])
                if not self.params['fix_wn']:
                    self.params['results'][self.module]['white'].append(self.params['pars'][-1])
        else:
            self.bg_corr = np.copy(self.random_pow)/self.params['noise']
            self.params['pars'] = ([self.params['noise']])


    def global_fit(self):
        """Global fit

        Fits global asteroseismic parameters :math:`\\rm \\nu{max}` and :math:`\\Delta\\nu`,
        where the former is estimated two different ways.

        Methods
            :mod:`pysyd.target.Target.numax_smooth`
            :mod:`pysyd.target.Target.numax_gaussian`
            :mod:`pysyd.target.Target.compute_acf`
            :mod:`pysyd.target.Target.frequency_spacing`


        """
        # get numax
        self.numax_smooth()
        self.numax_gaussian()
        # get dnu
        self.compute_acf()
        self.frequency_spacing()


    def numax_smooth(self, sm_par=None):
        """Smooth :math:`\\nu_{\\mathrm{max}}`

        Estimate numax by taking the peak of the smoothed power spectrum

        Parameters
            sm_par : float, optional
                smoothing width for power spectrum calculated from solar scaling relation (typically ~1-4)

        Attributes
            frequency, pssm : numpy.ndarray, numpy.ndarray
                smoothed power spectrum
            frequency, pssm_bgcorr : numpy.ndarray, numpy.ndarray
                smoothed :term:`background-subtracted power spectrum`
            region_freq, region_pow : numpy.ndarray, numpy.ndarray
                oscillation region of the power spectrum ("zoomed in") by applying the mask~[lower_ps,upper_ps]
            numax_smoo : float
                the 'observed' numax (i.e. the peak of the smoothed power spectrum)
            dnu_smoo : float
                the 'expected' dnu based on a scaling relation using the `numax_smoo`

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
        idx, max_freq, max_pow = utils._return_max(self.region_freq, self.region_pow)
        self.params['results'][self.module]['numax_smooth'].append(max_freq)
        self.params['results'][self.module]['A_smooth'].append(max_pow)
        self.params['numax_smoo'] = self.params['results'][self.module]['numax_smooth'][0]
        self.params['exp_dnu'] = utils.delta_nu(self.params['numax_smoo'])


    def numax_gaussian(self):
        """Gaussian :math:`\\nu_{\\mathrm{max}}`

        Estimate numax by fitting a Gaussian to the "zoomed-in" power spectrum (i.e. `region_freq`
        and `region_pow`) using :mod:`scipy.curve_fit`

        Returns
            converge : bool
                returns `False` if background model fails to converge

        Raises
            utils.ProcessingError
                if the Gaussian fit does not converge for the first step


        """
        guesses = [0.0, np.absolute(max(self.region_pow)), self.params['numax_smoo'], (max(self.region_freq)-min(self.region_freq))/np.sqrt(8.0*np.log(2.0))]
        bb = ([-np.inf,0.0,0.01,0.01],[np.inf,np.inf,np.inf,np.inf])
        try:
            gauss, _ = curve_fit(models.gaussian, self.region_freq, self.region_pow, p0=guesses, bounds=bb, maxfev=1000)
        except RuntimeError as _:
            self.converge = False
            if self.i == 0:
                raise utils.ProcessingError("Gaussian fit for numax failed to converge.\n\nPlease check your power spectrum and try again.")
        else:
            if self.i == 0:
                # Create an array with finer resolution for plotting
                new_freq = np.linspace(min(self.region_freq), max(self.region_freq), 10000)
                self.params['plotting'][self.module].update({'pssm':np.copy(self.pssm),
                                                             'obs_numax':self.params['numax_smoo'],
                                                             'new_freq':new_freq,
                                                             'numax_fit':models.gaussian(new_freq, *gauss),
                                                             'exp_dnu':self.params['exp_dnu'],
                                                             'region_freq':np.copy(self.region_freq),
                                                             'region_pow':np.copy(self.region_pow),})
            # Save values
            self.params['results'][self.module]['numax_gauss'].append(gauss[2])
            self.params['results'][self.module]['A_gauss'].append(gauss[1])
            self.params['results'][self.module]['FWHM'].append(gauss[3])


    def compute_acf(self, fft=True, smooth_ps=2.5,):
        """ACF

        Compute the autocorrelation function (:term:`ACF`) of the background-divided power 
        spectrum (i.e. `bg_corr`), with an option to smooth the :term:`BCPS` first

        Parameters
            fft : bool, default=True
                if `True`, uses FFTs to compute the ACF, otherwise it will use :mod:`numpy.correlate`
            smooth_ps : float, optional
                convolve the background-corrected PS with a box filter of this width (:math:`\\rm \\mu Hz`)

        Attributes
            bgcorr_smooth : numpy.ndarray
                smoothed background-corrected power spectrum if `smooth_ps != 0` else copy of `bg_corr`     
            lag, auto : numpy.ndarray, numpy.ndarray
                the autocorrelation of the "zoomed-in" power spectrum

        """
        # Optional smoothing of PS to remove fine structure before computing ACF
        if self.params['smooth_ps'] == 0.0:
            self.bgcorr_smooth = np.copy(self.bg_corr)
        else:
            boxkernel = Box1DKernel(int(np.ceil(self.params['smooth_ps']/self.params['resolution'])))
            self.bgcorr_smooth = convolve(self.bg_corr, boxkernel)
        # Use only power near the expected numax to reduce additional noise in ACF
        power = self.bgcorr_smooth[(self.frequency >= self.params['ps_mask'][0])&(self.frequency <= self.params['ps_mask'][1])]
        lag = np.arange(0.0, len(power))*self.params['resolution']
        if self.params['fft']:
            auto = np.real(np.fft.fft(np.fft.ifft(power)*np.conj(np.fft.ifft(power))))
        else:
            auto = np.correlate(power-np.mean(power), power-np.mean(power), "full")
            auto = auto[int(auto.size/2):]
        mask = np.ma.getmask(np.ma.masked_inside(lag, self.params['exp_dnu']/4., 2.*self.params['exp_dnu']+self.params['exp_dnu']/4.))
        lag, auto = lag[mask], auto[mask]
        auto = (auto - min(auto))/(max(auto) - min(auto))
        self.lag, self.auto = np.copy(lag), np.copy(auto)


    def frequency_spacing(self, n_peaks=10,):
        """Estimate :math:`\\Delta\\nu`

        Estimates the large frequency separation (or :math:`\\Delta\\nu`) by fitting a 
        Gaussian to the peak of the ACF "cutout" using :mod:`scipy.curve_fit`. 

        Parameters
            n_peaks : int, default=10
                the number of peaks to identify in the ACF

        Attributes
            peaks_l, peaks_a : numpy.ndarray
                the n highest peaks (`n_peaks`) in the ACF
            zoom_lag, zoom_auto : numpy.ndarray
                cutout from the ACF of the peak near dnu

        Returns
            converge : bool
                returns `False` if a Gaussian could not be fit within the `1000` iterations

        Raises
            utils.ProcessingError
                if a Gaussian could not be fit to the provided peak

        .. seealso:: :mod:`pysyd.target.Target.acf_cutout`, :mod:`pysyd.target.Target.optimize_ridges`,
                     :mod:`pysyd.target.Target.echelle_diagram`

        .. note:: 

           For the first step, a Gaussian weighting (centered on the expected value for dnu, or `exp_dnu`) is 
           automatically computed and applied by the pipeline to prevent the fit from latching 
           on to a peak that is a harmonic and not the actual spacing


        """
        if self.i == 0:
            # Get actual peaks from ACF for plotting purposes before any weighting
            pl, pa, _ = utils._max_elements(self.lag, self.auto, npeaks=self.params['n_peaks'], distance=self.params['exp_dnu']/4.)
            # Get peaks from ACF by providing dnu to weight the array 
            peaks_l, peaks_a, weights = utils._max_elements(self.lag, self.auto, npeaks=self.params['n_peaks'], distance=self.params['exp_dnu']/4., exp_dnu=self.params['exp_dnu'])
            # Pick "best" peak in ACF (i.e. closest to expected dnu)
            idx , self.params['best_lag'], self.params['best_auto'] = utils._return_max(peaks_l, peaks_a, exp_dnu=self.params['exp_dnu'])
            self._acf_cutout()
        self.zoom_lag = self.lag[(self.lag >= self.params['acf_mask'][0]) & (self.lag <= self.params['acf_mask'][1])]
        self.zoom_auto = self.auto[(self.lag >= self.params['acf_mask'][0]) & (self.lag <= self.params['acf_mask'][1])]
        # fit a Gaussian to the peak to estimate dnu
        try:
            gauss, _ = curve_fit(models.gaussian, self.zoom_lag, self.zoom_auto, p0=self.params['acf_guesses'], bounds=self.params['acf_bb'], maxfev=1000)
        # did the fit converge
        except RuntimeError:
            self.converge = False
            if self.i == 0:
            # Raise error if it's the first step
                raise utils.ProcessingError("Gaussian fit for dnu failed to converge.\n\nPlease check your power spectrum and try again.")
        # if fit converged, save appropriate results
        else:
            self.params['results'][self.module]['dnu'].append(gauss[2]) 
            if self.i == 0:
                self.params['obs_dnu'] = gauss[2]
                idx, _, _ = utils._return_max(pl, pa, exp_dnu=self.params['obs_dnu'])
                l, a = pl.pop(idx), pa.pop(idx)
                self.params['plotting'][self.module].update({'obs_dnu':gauss[2], 
                  'peaks_l':np.copy(pl),'peaks_a':np.copy(pa),'best_lag':self.params['best_lag'],
                  'best_auto':self.params['best_auto'],'weights':np.copy(weights/max(weights)),
                  'new_lag':np.linspace(min(self.zoom_lag),max(self.zoom_lag),2000), 
                  'dnu_fit':models.gaussian(np.linspace(min(self.zoom_lag),max(self.zoom_lag),2000), *gauss),})
                self.echelle_diagram()


    def _acf_cutout(self, threshold=1.0,):
        """ACF cutout

        Gets the region in the ACF centered on the correct peak to prevent pySYD
        from getting stuck in a local maximum (i.e. fractional and integer harmonics)

        Parameters
            threshold : float, default=1.0
                the threshold is multiplied by the full-width half-maximum value, centered on the peak 
                in the ACF to determine the width of the cutout region


        """
        mask = (self.frequency >= self.params['ps_mask'][0]) & (self.frequency <= self.params['ps_mask'][1])
        self.zoom_freq = self.frequency[mask]
        self.zoom_pow = self.bgcorr_smooth[mask]
        self.params['plotting'][self.module].update({'zoom_freq':np.copy(self.zoom_freq),
                  'zoom_pow':np.copy(self.zoom_pow),'lag':np.copy(self.lag),'auto':np.copy(self.auto),})
        # Literally calculate FWHM
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
        self.params['acf_mask'] = [self.params['best_lag']-(self.params['best_lag']-left_lag)*self.params['threshold'], self.params['best_lag']+(right_lag-self.params['best_lag'])*self.params['threshold']]
        zoom_lag = self.lag[(self.lag>=self.params['acf_mask'][0])&(self.lag<=self.params['acf_mask'][1])]
        zoom_auto = self.auto[(self.lag>=self.params['acf_mask'][0])&(self.lag<=self.params['acf_mask'][1])]
        self.params['plotting'][self.module].update({'zoom_lag':np.copy(zoom_lag),'zoom_auto':np.copy(zoom_auto)})
        # Boundary conditions and initial guesses stay the same for all iterations
        self.params['acf_guesses'] = [np.mean(zoom_auto), self.params['best_auto'], self.params['best_lag'], self.params['best_lag']*0.01*2.]
        self.params['acf_bb'] = ((-np.inf,0.,min(zoom_lag),10**-2.),(np.inf,np.inf,max(zoom_lag),2.*(max(zoom_lag)-min(zoom_lag)))) 


    def echelle_diagram(self, smooth_ech=None, nox=None, noy='0+0', hey=False, npb=10, nshift=0, clip_value=3.0,):
        """Echelle diagram

        Calculates everything required to plot an :term:`echelle diagram` **Note:** this does not
        currently have the `get_ridges` method attached (i.e. not optimizing the spacing or stepechelle)

        Parameters
            smooth_ech : float, default=None
                value to smooth (i.e. convolve) ED by
            nox : int, default=0
                number of grid points in x-axis of echelle diagram 
            noy : str, default='0+0'
                number of orders (y-axis) to plot in echelle diagram
            npb : int, default=10
                option to provide the number of points per bin as opposed to an arbitrary value (calculated from spacing and frequency resolution)
            nshift : int, default=0
                number of orders to shift echelle diagram (i.e. + is up, - is down)
            hey : bool, default=False
                plugin for Dan Hey's echelle package **(not currently implemented)**
            clip_value : float, default=3.0
                to clip any peaks higher than Nx the median value

        Attributes
            ed : numpy.meshgrid
                smoothed + summed 2d power for echelle diagram
            extent : List[float]
                bounding box for echelle diagram

        """
        self.optimize_ridges()
        use_dnu = self.params['plotting'][self.module]['use_dnu']
        if self.params['smooth_ech'] is not None:
            boxkernel = Box1DKernel(int(np.ceil(self.params['smooth_ech']/self.params['resolution'])))
            smooth_y = convolve(self.bg_corr, boxkernel)
        else:
            smooth_y = np.copy(self.bg_corr)
        # If the number of desired orders is not provided
        if self.params['noy'] == "0+0" or self.params['noy'] == "0-0":
            width = self.constants['width_sun']*(self.params['numax_smoo']/self.constants['numax_sun'])
            ny = int(np.ceil(width/use_dnu))
            nshift = 0
        else:
            if '+' in self.params['noy']:
                ny, nshift = int(self.params['noy'].split('+')[0]), int(self.params['noy'].split('+')[-1])
            if '-' in self.params['noy']:
                ny, nshift = int(self.params['noy'].split('-')[0]), int(self.params['noy'].split('-')[-1])
                nshift *= -1            
        # Make sure n_across isn't finer than the actual resolution grid
        if self.params['nox'] is None or (self.params['nox'] >= int(np.ceil(use_dnu/self.params['resolution']))):
            # add function to check that the resolution isn't ridiculous
            nx = int(np.ceil(use_dnu/self.params['resolution']/self.params['npb']))
        else:
            nx = int(self.params['nox'])
        x = np.linspace(0.0, 2*use_dnu, 2*nx+1)
        yy = np.arange(min(self.frequency),max(self.frequency),use_dnu)
        lower = self.params['numax_smoo']-(use_dnu*(ny/2.))+(use_dnu*(nshift+0))
        upper = self.params['numax_smoo']+(use_dnu*(ny/2.))+(use_dnu*(nshift+1))
        y = yy[(yy >= lower)&(yy <= upper)]
        z = np.zeros((ny+1,2*nx))
        for i in range(1,ny+1):
            y_mask = ((self.frequency >= y[i-1]) & (self.frequency < y[i]))
            for j in range(nx):
                x_mask = ((self.frequency%(use_dnu) >= x[j]) & (self.frequency%(use_dnu) < x[j+1]))
                if np.sum(x_mask & y_mask) > 0:
                    z[i][j] = np.sum(smooth_y[x_mask & y_mask])
                else:
                    z[i][j] = np.nan
        z[0][:nx], z[-1][nx:] = np.nan, np.nan
        for k in range(ny):
            z[k][nx:] = z[k+1][:nx]
        self.ed = np.copy(z)
        self.extent = [min(x),max(x),min(y),max(y)]
        # make copy of ED to flatten and clip outliers
        ed_copy = self.ed.flatten()
        if self.params['clip_value'] > 0:
            cut = np.nanmedian(ed_copy)+(self.params['clip_value']*np.nanmedian(ed_copy))
            ed_copy[ed_copy >= cut] = cut
        self.ed = ed_copy.reshape((self.ed.shape[0], self.ed.shape[1]))
        self.collapse_ed()


    def collapse_ed(self, n_trials=3):
        """Get ridges

        Optimizes the large frequency separation by determining which spacing creates the
        "best" ridges (but is currently under development) think similar to a step-echelle
        but quicker and more hands off?

        Attributes
            x : numpy.ndarray
                x-axis for the collapsed ED ~[0, :math:`2\\times\\Delta\\nu`]
            y : numpy.ndarray
                marginalized power along the y-axis (i.e. collapsed on to the x-axis)

        .. important:: need to optimize this - currently does nothing really

        """
        n = int(np.ceil(self.params['plotting'][self.module]['use_dnu']/self.params['resolution']))
        xx = np.linspace(0.0, self.params['plotting'][self.module]['use_dnu'], n)
        yy = np.zeros_like(xx)
        modx = self.frequency%self.params['plotting'][self.module]['use_dnu']
        for k in range(n-1):
            mask = (modx >= xx[k])&(modx < xx[k+1])
            if np.sum(mask) > 0:
                xx[k] = np.median(modx[mask])
                yy[k] = np.sum(self.bg_corr[mask])
        mask = np.ma.getmask(np.ma.masked_where(yy == 0.0, yy))
        xx, yy = xx[~mask], yy[~mask]
        self.x = np.array(xx.tolist()+list(xx+self.params['plotting'][self.module]['use_dnu']))
        self.y = np.array(list(yy)+list(yy))-min(yy)
        self.params['plotting'][self.module].update({'ed':np.copy(self.ed),'extent':np.copy(self.extent),'x':np.copy(self.x),'y':np.copy(self.y),})
        self.i += 1


    def get_epsilon(self, n_trials=3):
        """Get ridges

        Optimizes the large frequency separation by determining which spacing creates the
        "best" ridges (but is currently under development) think similar to a step-echelle
        but quicker and more hands off?

        Attributes
            x : numpy.ndarray
                x-axis for the collapsed ED ~[0, :math:`2\\times\\Delta\\nu`]
            y : numpy.ndarray
                marginalized power along the y-axis (i.e. collapsed on to the x-axis)

        .. important:: need to optimize this - currently does nothing really

        """
        self.params['results']['trial'] = []
        if not self.params['force']:
            self.params['boxes'] = np.logspace(np.log10(0.01*self.params['obs_dnu']), np.log10(0.1*self.params['obs_dnu']), self.params['n_trials'])
            self.freq, self.bgcorr_pow = np.copy(self.frequency%self.params['obs_dnu']), np.copy(self.bg_corr)
            self._collapsed_acf()
            print(self.params['results']['trial'])
            best = self.params['compare'].index(max(self.params['compare']))+1
            return self.params['results']['trial'][best]['value']
        else:
            if self.params['dnu'] is None:
                return self.params['obs_dnu']
            else:
                return self.params['dnu']


    def _collapsed_dnu(self, n_trials=3, step=0.25, max_snr=1000.0,):
        """Collapsed ACF

        Computes a collapsed autocorrelation function (ACF) using n different box sizes in
        n different trials (i.e. `n_trials`)

        Parameters
            n_trials : int, default=3
                the number of trials to run
            step : float, default=0.25
                fractional step size to use for the collapsed ACF calculation
            max_snr : float, default=100.0
                the maximum signal-to-noise of the estimate (this is primarily for plot formatting)


        """
        self.params['compare2'], self.params['step2'], self.params['n_trials2'], self.module = [], 0.1, 6, 'trial'
        self.params['results'][self.module], self.params['plotting'][self.module] = {}, {}
        self.params['boxes2'] = np.logspace(np.log10(self.params['obs_dnu']/10.0), np.log10(2.*self.params['obs_dnu']), self.params['n_trials2'])
        new_freq = np.copy(self.frequency%self.params['obs_dnu'])
        s = np.argsort(new_freq)
        freq, bgcorr_pow = new_freq[s], self.bg_corr[s]
        # Computes a collapsed ACF using different "box" (or bin) sizes
        for b, box in enumerate(self.params['boxes2']):
            self.params['results'][self.module][b+1], self.params['plotting'][self.module][b] = {}, {}
            start, snr, cumsum, md = 0, 0.0, [], []
            subset = np.ceil(box/self.params['resolution'])
            steps = np.ceil((box*self.params['step2'])/self.params['resolution'])
            print(b, box, subset, steps)
            # Iterates through entire power spectrum using box width
            while True:
                if (start+subset) > len(freq):
                    break
                p = bgcorr_pow[int(start):int(start+subset)]
                auto = np.real(np.fft.fft(np.fft.ifft(p)*np.conj(np.fft.ifft(p))))
                cumsum.append(np.sum(np.absolute(auto-np.mean(auto))))
                md.append(np.mean(freq[int(start):int(start+subset)]))
                start += steps
            # subtract/normalize the summed ACF and take the max
            csum = list(np.copy(cumsum))
            # Pick the maximum value as an initial guess for dnu
            idx = csum.index(max(csum))
            self.params['plotting'][self.module].update({b:{'x':np.array(md),'y':np.array(csum),'maxx':md[idx],'maxy':csum[idx]}})
            # Fit Gaussian to get estimate value for numax
            try:
                best_vars, _ = curve_fit(models.gaussian, np.array(md), np.array(csum), p0=[np.median(csum), 1.0-np.median(csum), md[idx], self.params['acf_guesses'][-1]], maxfev=5000, bounds=self.params['acf_bb'],)
            except Exception as _:
                self.params['plotting'][self.module][b].update({'good_fit':False,'fitx':np.linspace(min(md), max(md), 10000)})
            else:
                self.params['plotting'][self.module][b].update({'good_fit':True,})
                self.params['plotting'][self.module][b].update({'fitx':np.linspace(min(md), max(md), 10000),'fity':models.gaussian(np.linspace(min(md), max(md), 10000), *best_vars)})
                snr = max(self.params['plotting'][self.module][b]['fity'])/np.absolute(best_vars[0])
                if snr > max_snr:
                    snr = max_snr
                self.params['results'][self.module][b+1].update({'value':best_vars[2],'snr':snr})
                self.params['plotting'][self.module][b].update({'value':best_vars[2],'snr':snr})
                if self.params['verbose']:
                    print('Estimate %d: %.2f +/- %.2f'%(b+1, best_vars[2], np.absolute(best_vars[3])/2.0))
                    print('S/N: %.2f' % snr)
            self.params['compare2'].append(snr)
#        self = plots.select_trial2(self)
        print(self.params['compare2'])
        best = self.params['compare2'].index(max(self.params['compare2']))+1
        print(best)
        print(self.params['results'][self.module][best]['value'])
        self.module = 'parameters'


    def optimize_ridges(self, n=50, res=0.01):
        """Get ridges

        Optimizes the large frequency separation by determining which spacing creates the
        "best" ridges (but is currently under development) think similar to a step-echelle
        but quicker and more hands off?

        Attributes
            x : numpy.ndarray
                x-axis for the collapsed ED ~[0, :math:`2\\times\\Delta\\nu`]
            y : numpy.ndarray
                marginalized power along the y-axis (i.e. collapsed on to the x-axis)

        .. important:: need to optimize this - currently does nothing really

        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.collections import LineCollection
        if self.params['ridges']:
            fig, ax = plt.subplots(figsize=(10,6))
            dnu = float('%.2f'%self.params['obs_dnu'])
            lower, upper, nox = dnu-1.0, dnu+1.0, int(np.floor(self.params['obs_dnu']/self.params['resolution']))
            colors = plots.get_colors(np.arange(lower,upper,res))
            if self.params['nox'] is not None and self.params['nox'] <= np.ceil(self.params['obs_dnu']/self.params['resolution']):
                nox = self.params['nox']
            note, sigma = '', 1.0
            note_formats = [">10s",">10s",">10s",">10s",">10s"]
            note_cols = ['dnu','med','std','peak','sig']
            text = '{:{}}'*len(note_cols)+'\n'
            fmt = sum(zip(note_cols,note_formats),())
            note += text.format(*fmt)
            for n, spacing in enumerate(np.arange(lower,upper,res)):
                bins = np.linspace(0.0, spacing, nox)
                x, y = self.frequency%spacing, np.copy(self.bg_corr)
                digitized = np.digitize(self.frequency%spacing, bins)
                x2 = np.array([x[digitized == i].mean() for i in range(len(bins)) if len(x[digitized == i]) > 0])
                y2 = np.array([y[digitized == i].sum() for i in range(len(bins)) if len(x[digitized == i]) > 0])
                ye = np.std(y2)
                _, peaksy, _ = utils.max_elements(x2, y2, npeaks=1)
                if (peaksy[0]-np.median(y2))/ye > sigma:
                    sigma = (peaksy[0]-np.median(y2))/ye
                    use_dnu = spacing
                ax.plot(x2, y2, color=colors[n], alpha=0.75, linewidth=0.75)
                values = ['%.2f'%spacing, '%.2f'%np.median(y2), '%.2f'%ye, '%.2f'%peaksy[0], '%.2f'%((peaksy[0]-np.median(y2))/ye)]
                text = '{:{}}'*len(values)+'\n'
                fmt = sum(zip(values,note_formats),())
                note += text.format(*fmt)
            if self.params['save']:
                path = os.path.join(self.params['path'],'test_ridges.txt')
                if not self.params['overwrite']:
                    path = utils._get_next(path)
                with open(path,"w") as f:
                    f.write(note)
            ax.set_xlabel(r'$\rm Folded \,\, Frequency \,\, [\mu Hz]$', fontsize=28)
            ax.set_ylabel(r'$\rm Collapsed \,\, ED \,\, [power]$', fontsize=28)
            ax.set_xlim([0.0,upper])
            plt.tight_layout()
            if self.params['save']:
                path = os.path.join(self.params['path'],'1d_ed.png')
                if not self.params['overwrite']:
                    path = utils._get_next(path)
                plt.savefig(path, dpi=300)
            plt.close()
        else:
            if self.params['force'] and self.params['dnu'] is not None:
                use_dnu = self.params['dnu']
            else:
                use_dnu = self.params['obs_dnu']
        self.params['plotting'][self.module]['use_dnu'] = use_dnu
