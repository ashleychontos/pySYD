import os
import glob
import numpy as np
import pandas as pd
from astropy.stats import mad_std
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle as lomb
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft



import utils
import plots
import models



class LightCurve(object):
    """Light curve data
    """
    def __init__(self):
        """
        """
        for each in ['notes', 'warnings', 'errors']:
            if not hasattr(self, each):
                self.__setattr__(each, [])

    def __repr__(self):
        return "pysyd.target.LightCurve(name=%r)" % (self.name, os.path.exists(os.path.join(self.inpdir, '%s_LC.txt' % str(self.name))))

    def __str__(self):
        return "<%s LC>" % str(self.name)

    def load_data(self, long=10**6):
        """Load LC
        """
        # Try loading the light curve
        if os.path.exists(os.path.join(self.inpdir, '%s_LC.txt' % str(self.name))):
            self.time, self.flux = self.read_file(os.path.join(self.inpdir, '%s_LC.txt' % str(self.name)))
            self.time -= min(self.time)
            self.notes.append('LIGHT CURVE (LC): %d lines of data read' % len(self.time))
            if len(self.time) >= long:
                self.warnings.append('LC is long and may slow down the software')
            self.cadence = int(round(np.nanmedian(np.diff(self.time)*24.0*60.0*60.0),0))
            self.nyquist = 10**6./(2.0*self.cadence)
            self.baseline = (max(self.time)-min(self.time))*24.*60.*60.
            self.tau_upper = self.baseline/2.
            self.notes.append('LC cadence: %d seconds' % self.cadence)
            self.stitch_data()
        else:
            self.warnings.append('no time series data found')

    def stitch_data(self, stitch=False, gap=20):
        """Stitch light curve

        For computation purposes and for special cases that this does not affect the integrity of the results,
        this module 'stitches' a light curve together for time series data with large gaps. For stochastic p-mode
        oscillations, this is justified if the lifetimes of the modes are smaller than the gap. 

        Parameters
            gap : int
                how many consecutive missing cadences to be considered a 'gap'
      
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
        if self.stitch and hasattr(self, 'time'):
            self.new_time = np.copy(self.time)
            for i in range(1,len(self.time)):
                if (self.new_time[i]-self.new_time[i-1]) > float(self.gap)*(self.cadence/24./60./60.):
                    self.new_time[i] = self.new_time[i-1]+(self.cadence/24./60./60.)
            self.time = np.copy(self.new_time)
            self.warnings.append('using stitch data module (which is dodgy and strongly discouraged)')

    def compute_spectrum(self, oversampling_factor=1, long=10**6):
        r"""Compute power spectrum

        **NEW** function to calculate a power spectrum given time series data, which will
        normalize the power spectrum to spectral density according to Parseval's theorem

        Parameters
            oversampling_factor : int
                the oversampling factor to use when computing the power spectrum 

        Attributes
            frequency, power : numpy.ndarray, numpy.ndarray
                power spectrum computed from the 'time' and 'flux' attributes
                using the :mod:`astropy.timeseries.LombScargle` module

        .. important::
        
            the newly-computed and normalized power spectrum is in units of :math:`\\rm \\mu Hz` 
            vs. :math:`\\rm ppm^{2} \\mu Hz^{-1}`. IF you are unsure if your power spectrum is in 
            the correct units, we recommend using this new module to compute and normalize the PS
            for you -- this will ensure accurate results.

        .. todo::

           add equation for conversion
           add in unit conversions

        """
        freq, pow = lomb(self.time, self.flux).autopower(method='fast', samples_per_peak=self.oversampling_factor, maximum_frequency=self.nyquist)
        # normalize PS according to Parseval's theorem
        self.power = 4.*pow*np.var(self.flux*1e6)/(np.sum(pow)*(freq[1]-freq[0]))
        self.frequency = freq*(10.**6/(24.*60.*60.))
        self.notes.append('NEWLY-COMPUTED PS has length of %d' % len(self.frequency))
        if len(self.frequency) >= long:
            self.warnings.append('PS is large and may slow down software')

        
class PowerSpectrum(object):
    """Power spectrum data
    """
    def __init__(self):
        """
        """
        for each in ['notes', 'warnings', 'errors']:
            if not hasattr(self, each):
                self.__setattr__(each, [])

    def __repr__(self):
        return "pysyd.target.PowerSpectrum(name=%r, PS=%r)" % (self.name, os.path.exists(os.path.join(self.params['inpdir'], '%s_PS.txt' % str(self.name))))

    def __str__(self):
        return "<%s PS>" % str(self.name)

    def load_data(self, long=10**6):
        r"""Load PS
    
        Loads in available power spectrum and computes relevant information -- also checks
        for time series data and will raise a warning if there is none since it will have
        to assume a :term:`critically-sampled power spectrum`

        Attributes
            note : str, optional
                verbose output
            ps : bool
                `True` if star ID has an available (or newly-computed) power spectrum

        Raises
            :mod:`pysyd.utils.InputWarning`
                if no information or time series data is provided (i.e. *has* to assume the PS is critically-sampled) 

        """
        # Try loading the power spectrum
        if os.path.exists(os.path.join(self.inpdir, '%s_PS.txt' % str(self.name))):
            self.frequency, self.power = self.read_file(os.path.join(self.inpdir, '%s_PS.txt' % str(self.name)))
            self.notes.append('POWER SPECTRUM (PS): %d lines of data read' % len(self.frequency))
            if len(self.frequency) >= long:
                self.warnings.append('PS is large and may slow down software')
            self.fix_data()
        else:
            self.warnings.append('no power spectrum found')

    def fix_data(self):
        """Fix frequency domain data
        """
        self.remove_artefact()
        self.whiten_mixed()
        if self.kep_corr or self.ech_mask is not None:
            self.warnings.append('modifying the PS with optional module(s)')

    def _get_seed(self):
        if self.seed is None:
            if os.path.exists(self.info):
                df = pd.read_csv(self.info)
                stars = [str(each) for each in df.star.values.tolist()]
                # check if star is in file and if not, adds it and the seed for reproducibility
                if str(self.name) in stars and 'seed' in df.columns.values.tolist():
                    idx = stars.index(str(self.name))
                    if not np.isnan(float(df.loc[idx,'seed'])):
                        self.seed = int(df.loc[idx,'seed'])
            # if still None, generate new seed
            if self.seed is None:
                self._new_seed()
        if self.save:
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
        self.seed = int(np.random.randint(lower,high=upper))

    def _save_seed(self):
        """Save seed
        """
        if os.path.exists(self.info):
            df = pd.read_csv(self.info)
            stars = [str(each) for each in df.star.values.tolist()]
            if str(self.name) in stars:
                idx = stars.index(str(self.name))
                df.loc[idx,'seed'] = int(self.seed)
            else:
                idx = len(df)
                df.loc[idx,'star'], df.loc[idx,'seed'] = str(self.name), int(self.seed)
        else:
            df = pd.DataFrame(columns=['star','seed'])
            df.loc[0,'star'], df.loc[0,'seed'] = str(self.name), int(self.seed)
        df.to_csv(self.info, index=False)


    def remove_artefact(self, kep_corr=False, lcp=1.0/(29.4244*60*1e-6), 
                        lf_lower=[240.0,500.0], lf_upper=[380.0,530.0], 
                        hf_lower = [4530.0,5011.0,5097.0,5575.0,7020.0,7440.0,7864.0],
                        hf_upper = [4534.0,5020.0,5099.0,5585.0,7030.0,7450.0,7867.0],):
        """Remove Kepler (frequency-domain) artefacts
        """
        if self.kep_corr and hasattr(self, 'frequency'):
            resolution = self.frequency[1]-self.frequency[0]
            if self.seed is None:
                self._get_seed()
            # LC period in Msec -> 1/LC ~muHz
            artefact = (1.0+np.arange(14))*lcp
            # Estimate white noise
            white = np.mean(self.power[(self.frequency >= max(self.frequency)-100.0)&(self.frequency <= max(self.frequency)-50.0)])
            # Routine 1: remove 1/LC artefacts by subtracting +/- 5 muHz given each artefact
            np.random.seed(int(self.seed))
            for i in range(len(artefact)):
                if artefact[i] < np.max(self.frequency):
                    mask = np.ma.getmask(np.ma.masked_inside(self.frequency, artefact[i]-5.0*resolution, artefact[i]+5.0*resolution))
                    if np.sum(mask) != 0:
                        self.power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0
            # Routine 2: fix high frequency artefacts
            np.random.seed(int(self.seed))
            for lower, upper in zip(hf_lower, hf_upper):
                if lower < np.max(self.frequency):
                    mask = np.ma.getmask(np.ma.masked_inside(self.frequency, lower, upper))
                    if np.sum(mask) != 0:
                        self.power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0
            # Routine 3: remove wider, low frequency artefacts 
            np.random.seed(int(self.seed))
            for lower, upper in zip(lf_lower, lf_upper):
                low = np.ma.getmask(np.ma.masked_outside(self.frequency, lower-20., lower))
                upp = np.ma.getmask(np.ma.masked_outside(self.frequency, upper, upper+20.))
                # Coeffs for linear fit
                m, b = np.polyfit(self.frequency[~(low*upp)], self.power[~(low*upp)], 1)
                mask = np.ma.getmask(np.ma.masked_inside(self.frequency, lower, upper))
                # Fill artefact frequencies with noise
                self.power[mask] = ((self.frequency[mask]*m)+b)*(np.random.chisquare(2, np.sum(mask))/2.0)


    def whiten_mixed(self, ech_mask=None, dnu=None, lower_ech=None, upper_ech=None, notching=False,):
        """Whiten mixed modes
        """
        if self.ech_mask is not None and hasattr(self, 'frequency'):
            if self.seed is None:
                self._get_seed()
            # Estimate white noise
            if not self.notching:
                white = np.mean(self.power[(self.frequency >= max(self.frequency)-100.0)&(self.frequency <= max(self.frequency)-50.0)])
            else:
                white = min(self.power[(self.frequency >= max(self.frequency)-100.0)&(self.frequency <= max(self.frequency)-50.0)])
            # Take the provided dnu and "fold" the power spectrum
            mask = np.ma.getmask(np.ma.masked_inside((self.frequency%self.dnu), self.ech_mask[0], self.ech_mask[1]))
            # Set seed for reproducibility purposes
            np.random.seed(int(self.seed))
            # Makes sure the mask is not empty
            if np.sum(mask) != 0:
                if self.notching:
                    self.power[mask] = white
                else:
                    self.power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0
    

class Target(LightCurve, PowerSpectrum):
    r"""Target object
    
    A new instance is created for each processed star, which copies relevant (and
    individualistic aka fully customizable) dictonaries and parameters. This used to
    automatically load in available data too but this is now called separately with
    the upgraded LightCurve and PowerSpectrum classes
    
    """
    def __init__(self, name, args=None, width=65):
        r"""pySYD target

        Parameters
            name : str
                which target of interest to run
            args : :mod:`pysyd.utils.Parameters`
                container class of pysyd parameters

        Attributes
            notes : List[str]
                if verbose is `True`, this saves information about the input data
            warnings : List[str]
                akin to the notes list, this saves all warnings related to loading in of
                data, which will only raise a :mod:`pysyd.utils.InputWarning` iff verbose
                and warnings are both `True`
            errors : List[str]
                similar to 'notes' and 'warnings' except that these are dealbreakers --
                this will raise a :mod:`pysyd.utils.InputError` any time this isn't empty
            params : Dict
                copy of :mod:`pysyd.utils.Parameters` dictionary with target-specific 
                parameters and options
            constants : Dict
                copy of :mod:`pysyd.utils.Constants` container class

        """
        self.name, self.width = name, width
        for label in ['notes', 'warnings', 'errors']:
            if not hasattr(self, label):
                self.__setattr__(label, [])
        if args is None:
            params = utils.Parameters()
            params.add_targets(stars=name)
        else:
            params = utils.Parameters(args)
        for key, value in params.__dict__[self.name].items():
            self.__dict__[key] = value
        self.load_input()

    def __repr__(self):
        return "pysyd.target.Target(name=%r, LC=%r, PS=%r)" % (self.name, os.path.exists(os.path.join(self.inpdir, '%s_LC.txt' % str(self.name))), os.path.exists(os.path.join(self.inpdir, '%s_PS.txt' % str(self.name))))

    def __str__(self):
        return "<Star %s>" % (self.name)

    def load_input(self):
        r"""Load input data

        Load data in for a single star by initiating both the LightCurve and
        PowerSpectrum objects (via their 'load_data' methods - it has to be this
        way otherwise init will create everything twice - still not sure how to 
        fix this -- there's probably a better way)

        Methods
            :mod:`pysyd.target.LightCurve.load_data`
            :mod:`pysyd.target.PowerSpectrum.load_data`
            :mod:`pysyd.target.Target.get_properties`  

        """
        line = '  TARGET: %s  ' % str(self.name)
        for msg in ['*'*self.width, line.center(self.width, '*'), '*'*self.width]:
            self.notes.append(msg)
        LightCurve.load_data(self)
        print(self.__dict__)
        PowerSpectrum.load_data(self)
        print(self.__dict__)
        self.get_properties()
        print(self.__dict__)

    def get_properties(self):
        r"""Get data properties

        After loading in available data, it determines what methods need to be called
        or what needs to be corrected in order for the pipeline to run. For example,
        after trying to load in the light curve and power spectrum, it will compute a
        new power spectrum if there isn't one already.

        Methods
            :mod:`pysyd.target.LightCurve.compute_spectrum`
            :mod:`pysyd.utils._save_file`
            :mod:`pysyd.target.PowerSpectrum.fix_data`

        Attributes
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
            :mod:`pysyd.utils.InputError`
                if no data is found for a given target
            :mod:`pysyd.utils.InputWarning`, optional
                if warnings are turned on, it will raise whatever warnings popped up

        """
        # CASE 1: LIGHT CURVE AND POWER SPECTRUM
        if hasattr(self, 'time') and hasattr(self, 'frequency'):
            ofactor = (1./((max(self.time)-min(self.time))*0.0864))/(self.frequency[1]-self.frequency[0])
            if self.oversampling_factor is not None and ofactor != self.oversampling_factor:
                self.warnings.append('the input ofactor (=%d) and calculated value (=%d) do NOT match' % (int(self.oversampling_factor), int(ofactor)))
            else:
                if not float(ofactor).is_integer():
                    self.errors.append('calculated an ofactor=%.1f but MUST be int type for array indexing' % float(ofactor))
                self.oversampling_factor = int(float('%.0f' % ofactor))
        # CASE 2: LIGHT CURVE AND NO POWER SPECTRUM
        elif hasattr(self, 'time') and not hasattr(self, 'frequency'):
            if self.oversampling_factor is None:
                self.oversampling_factor = 1
            LightCurve.compute_spectrum(self)
            if self.save:
                self.save_file(self.frequency, self.power, os.path.join(self.inpdir, '%s_PS.txt'%self.name), overwrite=self.overwrite)
            PowerSpectrum.fix_data(self)
        # CASE 3: NO LIGHT CURVE AND POWER SPECTRUM
        elif not hasattr(self, 'time') and hasattr(self, 'frequency'):
            if self.oversampling_factor is None:
                self.warnings.append('using PS with no additional information (e.g., assuming critically-sampled)')
                if self.mc_iter > 1:
                    self.warnings.append('uncertainties may not be reliable if PS is not critically-sampled')
                self.oversampling_factor = 1
        # CASE 4: NO LIGHT CURVE AND NO POWER SPECTRUM
        else:
            self.errors.append('no data found for target %s' % str(self.name))
        errors, message = self.get_errors()
        if errors:
            print(message)
            return
        self.freq_os, self.pow_os = np.copy(self.frequency), np.copy(self.power)
        self.freq_cs = np.array(self.frequency[self.oversampling_factor-1::self.oversampling_factor])
        self.pow_cs = np.array(self.power[self.oversampling_factor-1::self.oversampling_factor])
        if self.oversampling_factor != 1:
            self.notes.append('PS is oversampled by a factor of %d' % self.oversampling_factor)
        else:
            self.notes.append('PS is critically-sampled')
        self.notes.append('PS resolution: %.6f muHz' % (self.freq_cs[1]-self.freq_cs[0]))
        warnings, message = self.get_warnings()
        if self.verbose:
            if warnings:
                print(message)
            output = ''
            for note in self.notes:
                output += '\n%s' % note
            print(output)

    def read_file(self, path):
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

    def save_file(self, x, y, path, overwrite=False, formats=[">15.8f", ">18.10e"]):
        r"""Save text file
    
        After determining the best-fit stellar background model, this module
        saved the background-subtracted power spectrum

        Parameters
            x : numpy.ndarray
                the independent variable i.e. the time or frequency array 
            y : numpy.ndarray
                the dependent variable, in this case either the flux or power array
            path : str
                absolute path to save file to
            overwrite : bool, optional
                whether to overwrite existing files or not
            formats : List[str], optional
                2x1 list of formats to save arrays as

        """
        if os.path.exists(path) and not overwrite:
            path = self._get_next(path)
        with open(path, "w") as f:
            for xx, yy in zip(x, y):
                values = [xx, yy]
                text = '{:{}}'*len(values) + '\n'
                fmt = sum(zip(values, formats), ())
                f.write(text.format(*fmt))

    def _get_next(self, path, count=1):
        """Get next integer
    
        When the overwriting of files is disabled, this module determines what
        the last saved file was 

        Parameters
            path : str
                absolute path to file name that already exists
            count : int
                starting count, which is incremented by 1 until a new path is determined

        Returns
            new_path : str
                unused path name

        """
        path, fname = os.path.split(path)[0], os.path.split(path)[-1]
        new_fname = '%s_%d.%s'%(fname.split('.')[0], count, fname.split('.')[-1])
        new_path = os.path.join(path, new_fname)
        if os.path.exists(new_path):
            while os.path.exists(new_path):
                count += 1
                new_fname = '%s_%d.%s'%(fname.split('.')[0], count, fname.split('.')[-1])
                new_path = os.path.join(path, new_fname)
        return new_path
    
    def get_errors(self, delim='#'):
        r"""Return errors
    
        Concatenates all the various flags/messages/warnings during the
        load into a single string for easy readability
        (defaults -> '#' for errors, '-' for warnings, and ' ' for notes)

        Parameters
            witch : str
                which attribute to concatenate
            delim : str
                delimiter for pretty formatting 

        Returns
            message : str
                the final concatenated message

        """
        if self.errors != []:
            errors = "%s\n ERROR(S):" % delim*self.width
            for message in self.__dict__['errors']:
                errors += '\n  - %s' % message
            errors += "\n%s" % (delim*self.width)
            return True, errors
        return False, ''

    def get_warnings(self, delim='-'):
        r"""Return warnings
    
        Concatenates all the various flags/messages/warnings during the
        load into a single string for easy readability
        (defaults -> '#' for errors, '-' for warnings, and ' ' for notes)

        Parameters
            witch : str
                which attribute to concatenate
            delim : str
                delimiter for pretty formatting 

        Returns
            message : str
                the final concatenated message

        """
        if self.warnings != []:
            warnings = "%s\n WARNING(S):" % delim*self.width
            for message in self.__dict__['warnings']:
                warnings += '\n  - %s' % message
            warnings += "\n%s" % (delim*self.width)
            return True, warnings
        return False, ''

    def get_notes(self, delim=' '):
        r"""Return warnings
    
        Concatenates all the various flags/messages/warnings during the
        load into a single string for easy readability
        (defaults -> '#' for errors, '-' for warnings, and ' ' for notes)

        Parameters
            witch : str
                which attribute to concatenate
            delim : str
                delimiter for pretty formatting 

        Returns
            message : str
                the final concatenated message

        """
        notes = "%s\n Note(s):" % delim*self.width
        for message in self.__dict__['notes']:
            notes += '\n  - %s' % message
        notes += "\n%s" % (delim*self.width)
        return notes

    def output_parameters(self, note=''):
        """Verbose output

        Prints verbose output from the global fit 

        """
        params = utils.get_dict()
        if not self.params['overwrite']:
            list_of_files = glob.glob(os.path.join(star.params['path'], 'global*'))
            file = max(list_of_files, key=os.path.getctime)
        else:
            file = os.path.join(star.params['path'], 'global.csv')
        df = pd.read_csv(file)
        note += '%s\nOutput parameters\n%s' % ('-'*self.width, '-'*self.width)
        if star.params['mc_iter'] > 1:
            line = '\n%s: %.2f +/- %.2f %s'
            for idx in df.index.values.tolist():
                note += line%(df.loc[idx,'parameter'], df.loc[idx,'value'], df.loc[idx,'uncertainty'], params[df.loc[idx,'parameter']]['unit'])
        else:
            line = '\n%s: %.2f %s'
            for idx in df.index.values.tolist():
                note += line%(df.loc[idx,'parameter'], df.loc[idx,'value'], params[df.loc[idx,'parameter']]['unit'])
        note += '\n%s' % '-'*self.width
        print(note)


    def bin_data(x, y, width, log=False, mode='mean'):
        """Bin data
    
        Bins 2D series of data

        Parameters
            x, y : numpy.ndarray, numpy.ndarray
                the x and y values of the data
            width : float
                bin width (typically in :math:`\\rm \\mu Hz`)
            log : bool
                creates equal bin sizes in logarithmic space when `True`

        Returns
            bin_x, bin_y, bin_yerr : numpy.ndarray, numpy.ndarray, numpy.ndarray
                binned arrays (and error is computed using the standard deviation)

        """
        if log:
            mi = np.log10(min(x))
            ma = np.log10(max(x))
            no = int(np.ceil((ma-mi)/width))
            bins = np.logspace(mi, mi+(no+1)*width, no)
        else:
            bins = np.arange(min(x), max(x)+width, width)
        digitized = np.digitize(x, bins)
        if mode == 'mean':
            bin_x = np.array([x[digitized == i].mean() for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
            bin_y = np.array([y[digitized == i].mean() for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
        elif mode == 'median':
            bin_x = np.array([np.median(x[digitized == i]) for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
            bin_y = np.array([np.median(y[digitized == i]) for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
        else:
            pass
        bin_yerr = np.array([y[digitized == i].std()/np.sqrt(len(y[digitized == i])) for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
        return bin_x, bin_y, bin_yerr

    def process_star(self,):
        """Run `pySYD`

        Process a given star with the main `pySYD` pipeline

        Methods
            - :mod:`pysyd.target.Target.estimate_parameters`
            - :mod:`pysyd.target.Target.derive_parameters`
            - :mod:`pysyd.utils.show_results`

        """
        self.params['results'], self.params['plotting'] = {}, {}
        self.estimate_parameters()
        self.derive_parameters()
        utils.show_results(self)

##########################################################################################

    def estimate_parameters(self, estimate=True, module='estimate'):
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
        self.module = 'estimate'
        if 'results' not in self.params:
            self.params['results'] = {}
        if 'plotting' not in self.params:
            self.params['plotting'] = {}
        self.params['results'][self.module], self.params['plotting'][self.module] = {}, {}
        if self.params['estimate']:
            # get initial values and fix data
            self.estimate_initial()
            # execute function
            self.estimate_values()
            # save results
            self.save_estimates()

    def estimate_initial(self, lower_ex=1.0, upper_ex=8000.0, max_trials=6,):
        """Initial guesses
    
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
        if hasattr(self, 'time'):
            self.params['plotting'][self.module].update({'time':np.copy(self.time),'flux':np.copy(self.flux)})
        if self.params['n_trials'] > max_trials:
            self.params['n_trials'] = max_trials
        if (self.params['numax'] is not None and self.params['numax'] <= 500.) or (self.nyquist is not None and self.nyquist <= 300.) or (max(self.frequency) < 300.):
            self.params['boxes'] = np.logspace(np.log10(0.5), np.log10(25.), self.params['n_trials'])
        else:
            self.params['boxes'] = np.logspace(np.log10(50.), np.log10(500.), self.params['n_trials'])

    def estimate_values(self, binning=0.005, bin_mode='mean', smooth_width=20.0, ask=False,):
        r"""Estimate :math:`\\rm \\nu_{max}`

        Automated routine to identify power excess due to solar-like oscillations and estimate
        an initial starting point for :term:`numax` (:math:`\\nu_{\\mathrm{max}}`)

        Parameters
            binning : float
                logarithmic binning width (i.e. evenly spaced in log space)
            bin_mode : str, {'mean', 'median', 'gaussian'}
                mode to use when binning
            smooth_width: float
                box filter width (in :math:`\\rm \\mu Hz`) to smooth power spectrum
            ask : bool, default=False
                If `True`, it will ask which trial to use as the estimate for numax

        Attributes
            bin_freq, bin_pow : numpy.ndarray, numpy.ndarray
                copy of the power spectrum (i.e. `freq` & `pow`) binned equally in logarithmic space
            smooth_freq, smooth_pow : numpy.ndarray, numpy.ndarray
                copy of the binned power spectrum (i.e. `bin_freq` & `bin_pow`) binned equally in linear space -- *yes, this is doubly binned intentionally*
            interp_pow : numpy.ndarray
                the smoothed power spectrum (i.e. `smooth_freq` & `smooth_pow`) interpolated back to the original frequency array (also referred to as "crude background model")
            bgcorr_pow : numpy.ndarray
                approximate :term:`background-corrected power spectrum` computed by dividing the original PS (`pow`) by the interpolated PS (`interp_pow`) 

        Methods
            - :mod:`pysyd.target.Target._collapsed_acf`

        """
        # Smooth the power in log-space
        self.bin_freq, self.bin_pow, _ = self.bin_data(self.freq, self.pow, width=self.params['binning'], log=True, mode=self.params['bin_mode'])
        # Smooth the power in linear-space
        try:
            self.smooth_freq, self.smooth_pow, _ = self.bin_data(self.bin_freq, self.bin_pow, width=self.params['smooth_width'])
            if len(self.smooth_freq) <= 20:
                self.errors = ['binned PS is too small (i.e. %d points)' % len(self.smooth_freq),
                               'please adjust the smoothing width (--sw) to a smaller value',
                               'default is currently set to => %d muHz' % int(self.params['smooth_width'])]
                errors = self.get_message(witch='error', delim='#')
                raise InputError(errors)
        except InputError as error:
            print(error.msg)
            return
        else:
            if self.params['verbose']:
                print('%s\nPS binned to %d datapoints\n\nNumax estimates\n---------------' % ('-'*self.width, len(self.smooth_freq)))
            # Mask out frequency values that are lower than the smoothing width to avoid weird looking fits
            if min(self.freq) < self.params['smooth_width']:
                mask = (self.smooth_freq >= self.params['smooth_width'])
                self.smooth_freq, self.smooth_pow = self.smooth_freq[mask], self.smooth_pow[mask]
            s = InterpolatedUnivariateSpline(self.smooth_freq, self.smooth_pow, k=1)
            # Interpolate and divide to get a crude background-corrected power spectrum
            self.interp_pow = s(self.freq)
            self.bgcorr_pow = self.pow/self.interp_pow
            self.params['plotting'][self.module].update({'bin_freq':np.copy(self.bin_freq),'bin_pow':np.copy(self.bin_pow),
                                                        'interp_pow':np.copy(self.interp_pow),'bgcorr_pow':np.copy(self.bgcorr_pow)})
            # Collapsed ACF to find numax
            self._collapsed_acf()
            self.params['best'] = self.params['compare'].index(max(self.params['compare']))+1
            # Select trial that resulted with the highest SNR detection
            if not self.params['ask']:
                if self.params['verbose']:
                    print('Selecting model %d' % self.params['best'])
            # Or ask which estimate to use
            else:
                self = plots.select_trial(self)

    def _collapsed_acf(self, n_trials=3, step=0.25, max_snr=100.0,):
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
        c = utils.Constants()
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
                best_vars, _ = curve_fit(models.gaussian, np.array(md), np.array(csum), p0=[np.median(csum), 1.0-np.median(csum), md[idx], c.width_sun*(md[idx]/c.numax_sun)], maxfev=5000, bounds=((-np.inf,-np.inf,1,-np.inf),(np.inf,np.inf,np.inf,np.inf)),)
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

    def save_estimates(self, variables=['star','numax','dnu','snr']):
        """Save results
    
        Saves the parameter estimates (i.e. results from first module)

        Parameters
            star : pysyd.target.Target
                processed pipeline target
            variables : List[str]
                list of estimated variables to save (e.g., :math:`\\rm \\nu_{max}`, :math:`\\Delta\\nu`)

        Returns
            star : pysyd.target.Target
                updated pipeline target

        """
        if 'best' in self.params:
            best = self.params['best']
            results = [self.name, self.params['results']['estimates'][best]['value'], delta_nu(self.params['results']['estimates'][best]['value']), self.params['results']['estimates'][best]['snr']]
            if self.params['save']:
                save_path = os.path.join(self.params['path'], 'estimates.csv')
                if not self.params['overwrite']:
                    save_path = self._get_next(save_path)
                ascii.write(np.array(results), save_path, names=variables, delimiter=',', overwrite=True)
            self.params['numax'], self.params['dnu'], self.params['snr'] = results[1], results[2], results[3]

    def check_numax(self, columns=['numax', 'dnu', 'snr']):
        r"""Check :math:`\\rm \\nu_{max}`
    
        Checks if there is an initial starting point or estimate for :term:`numax`

        Parameters
            columns : List[str]
                saved columns if the estimate_numax() function was run

        Raises
            :mod:`pysyd.utils.InputError`
                if an invalid value was provided as input for numax
            :mod:`pysyd.utils.ProcessingError`
                if it still cannot find any estimate for :term:`numax`


        """
        self.errors = []
        if self.params['numax'] is not None:
            if np.isnan(float(self.params['numax'])):
                self.errors.append('invalid value for numax')
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
                    self.errors.append('invalid value for numax')
            else:
                self.errors.append('no decent numax value found, try providing one instead')
        if self.errors != []:
            try:
                errors = self.get_message(witch='error', delim='#')
                raise InputError(errors)
            except InputError as error:
                print(error.msg)
                return False
            else:
                return True

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
        self.module = 'derive'
        if 'results' not in self.params:
            self.params['results'] = {}
        if 'plotting' not in self.params:
            self.params['plotting'] = {}
        self.params['results'][self.module], self.params['plotting'][self.module] = {}, {}
        # make sure there is an estimate for numax
        if self.check_numax():
            # get initial values and fix data
            self.derive_initial()
            self.first_step()
            # if the first step is ok, carry on
            if self.params['mc_iter'] > 1:
                self.get_samples()
            # Save results
            utils._save_parameters(self)

    def derive_initial(self, lower_bg=1.0, upper_bg=8000.0,):
        """Initial guesses
    
        Gets initial guesses for granulation components (i.e. timescales and amplitudes) using
        solar scaling relations. This resets the power spectrum and has its own independent
        filter (i.e. [lower,upper] mask) to use for this subroutine.

        Parameters
            lower_bg : float, default=1.0
                lower frequency limit of PS to use for the background fit
            upper_bg : float, default=8000.0
                upper frequency limit of PS to use for the background fit

        Attributes
            frequency, power : numpy.ndarray, numpy.ndarray
                copy of the entire oversampled (or critically-sampled) power spectrum (i.e. `freq_os` & `pow_os`) 
            frequency, random_pow : numpy.ndarray, numpy.ndarray
                copy of the entire oversampled (or critically-sampled) power spectrum (i.e. `freq_os` & `pow_os`) after applying the mask~[lower_bg,upper_bg]
            module : str, default='parameters'
                which part of the pipeline is currently being used
            i : int, default=0
                iteration number

        Methods
            :mod:`pysyd.target.Target.solar_scaling`
                uses multiple solar scaling relations to determnie accurate initial guesses for
                many of the derived parameters 

        .. warning::

            This is typically sufficient for most stars but may affect evolved stars and
            need to be adjusted!

        """
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
        self.params['plotting'][self.module].update({'frequency':np.copy(self.frequency),'random_pow':np.copy(self.power)})
        if hasattr(self, 'time'):
            self.params['plotting'][self.module].update({'time':np.copy(self.time),'flux':np.copy(self.flux)})
        # Get other relevant initial conditions
        self.i = 0
        for parameter in utils.get_dict(type="columns")['params']:
            self.params['results'][self.module].update({parameter:[]})
        # Use scaling relations from sun to get starting points
        self.solar_scaling()
        if self.params['verbose']:
            print('%s\nGLOBAL FIT\n%s' % ('-'*self.width, '-'*self.width))


    def solar_scaling(self, numax=None, scaling='tau_sun_single', max_laws=3, ex_width=1.0,
                      lower_ps=None, upper_ps=None,):
        """Initial values
        
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
        c = utils.Constants()
        if self.params['dnu'] is None:
            self.params['dnu'] = utils.delta_nu(self.params['numax'])
        # Use scaling relations to estimate width of oscillation region to mask out of the background fit
        width = c.width_sun*(self.params['numax']/c.numax_sun)
        maxpower = [self.params['numax']-(width*self.params['ex_width']), self.params['numax']+(width*self.params['ex_width'])]
        if self.params['lower_ps'] is not None:
            maxpower[0] = self.params['lower_ps']
        if self.params['upper_ps'] is not None:
            maxpower[1] = self.params['upper_ps']
        self.params['ps_mask'] = [maxpower[0], maxpower[1]]
        # make sure interval is not empty
        if not list(self.frequency[(self.frequency>=self.params['ps_mask'][0])&(self.frequency<=self.params['ps_mask'][1])]):
            raise utils.InputError("\nERROR: frequency region for power excess is null\nPlease specify an appropriate numax and/or frequency limits for the power excess (via --lp/--up)\n")
        # Estimate granulation time scales
        if scaling == 'tau_sun_single':
            taus = np.array(c.tau_sun_single)*(c.numax_sun/self.params['numax'])
        else:
            taus = np.array(c.tau_sun)*(c.numax_sun/self.params['numax'])
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
        self.params['plotting'][self.module].update({'exp_numax':self.params['numax'],'nlaws_orig':len(self.params['mnu']),'b_orig':np.copy(self.params['b'])})

    def first_step(self, background=True, globe=True,):
        """First step

        Processes a given target for the first step, which has extra steps for each of the two 
        main parts of this method (i.e. background model and global fit):
         #. **background model:** the automated best-fit model selection is only performed in the
            first step, the results which are saved for future purposes (including the 
            background-corrected power spectrum)
         #. **global fit:** while the :term:`ACF` is computed for every iteration, a mask is
            created in the first step to prevent the estimate for dnu to latch on to a different 
            (i.e. incorrect) peak, since this is a multi-modal parameter space

        Parameters
            background : bool, default=True
                run the automated background-fitting routine
            globe : bool, default=True
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
                print('%s\nSampling routine:' % '-'*self.width)

    def single_step(self,):
        """Single step

        Similar to the first step, this function calls the same methods but uses the selected best-fit
        background model from the first step to estimate the parameters

        Attributes
            converge : bool
                removes any saved parameters if any fits did not converge (i.e. `False`)

        Returns
            converge : bool
                returns `True` if all relevant fits converged

        Methods
            :mod:`pysyd.target.Target.estimate_background`
                estimates the amplitudes/levels of both correlated and frequency-independent noise
                properties from the input power spectrum
            :mod:`pysyd.target.Target.get_background`
                unlike the first step, which iterated through several models and performed a
                best-fit model comparison, this only fits parameters from the selected model 
                in the first step
            :mod:`pysyd.target.Target.global_fit`
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
        """Get samples

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

        .. important:: if the verbose option is enabled, the `tqdm` package is required here

        """
        # Switch to critically-sampled PS if sampling
        mask = np.ma.getmask(np.ma.masked_inside(self.freq_cs, self.params['bg_mask'][0], self.params['bg_mask'][1]))
        self.frequency, self.power = np.copy(self.freq_cs[mask]), np.copy(self.pow_cs[mask])
        self.params['resolution'] = self.frequency[1]-self.frequency[0]
        # Set seed for reproducibility
        if self.params['seed'] is None:
            self._get_seed()
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

        """
        # Bin power spectrum to model stellar background/correlated red noise components
        self.bin_freq, self.bin_pow, self.bin_err = self.bin_data(self.frequency, self.random_pow, width=self.params['ind_width'], mode=self.params['bin_mode'])
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
            box_filter : float, default=1.0
                the size of the 1D box smoothing filter
            n_rms : int, default=20
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
            n_laws : int, default=None
                specify number of Harvey-like components to use in background fit 
            fix_wn : bool, default=False
                option to fix the white noise instead of it being an additional free parameter 
            basis : str, default='tau_sigma'
                which basis to use for background fitting, e.g. {a,b} parametrization **TODO: not yet operational**

        Methods
            :mod:`pysyd.models.background`
            - :mod:`scipy.curve_fit`
            - :mod:`pysyd.models._compute_aic`
            - :mod:`pysyd.models._compute_bic`
            - :mod:`pysyd.target.Target.correct_background`

        Returns
            converge : bool
                returns `False` if background model fails to converge

        Raises
            ProcessingError
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
                    bounds[0].append(10.**-2)
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
                raise utils.ProcessingWarning('%s\nWARNING: estimating global parameters from raw PS:' % '-'*self.width)
            self.bg_corr = np.copy(self.random_pow)/self.params['noise']
            self.params['pars'] = ([self.params['noise']])
        # save final guesses for plotting purposes
        self.params['plotting'][self.module].update({'pars':self.params['pars'],})

    def correct_background(self, metric='bic'):
        """Correct background

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
            self.save_file(self.frequency, self.bg_div, os.path.join(self.params['path'], '%s_BDPS.txt'%self.name), overwrite=self.params['overwrite'])
        self.bg_sub = self.random_pow-models.background(self.frequency, self.params['pars'], noise=self.params['noise'])
        if self.params['save']:
            self.save_file(self.frequency, self.bg_sub, os.path.join(self.params['path'], '%s_BSPS.txt'%self.name), overwrite=self.params['overwrite'])
        self.params['plotting'][self.module].update({'models':self.params['models'],'model':self.params['selected'],'paras':self.params['paras'],
                                                     'aic':self.params['aic'],'bic':self.params['bic'],})
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
        r"""Global fit

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
        r"""Smooth :math:`\\nu_{\\mathrm{max}}`

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
        idx, max_freq, max_pow = utils.return_max(self.region_freq, self.region_pow)
        self.params['results'][self.module]['numax_smooth'].append(max_freq)
        self.params['results'][self.module]['A_smooth'].append(max_pow)
        self.params['numax_smoo'] = self.params['results'][self.module]['numax_smooth'][0]
        self.params['exp_dnu'] = utils.delta_nu(self.params['numax_smoo'])

    def numax_gaussian(self):
        r"""Gaussian :math:`\\nu_{\\mathrm{max}}`

        Estimate numax by fitting a Gaussian to the "zoomed-in" power spectrum (i.e. `region_freq`
        and `region_pow`) using :mod:`scipy.curve_fit`

        Returns
            converge : bool
                returns `False` if background model fails to converge

        Raises
            :mod:`pysyd.utils.ProcessingError`
                if the Gaussian fit does not converge for the first step

        """
        guesses = [0.0, np.absolute(max(self.region_pow)), self.params['numax_smoo'], (max(self.region_freq)-min(self.region_freq))/np.sqrt(8.0*np.log(2.0))]
        bb = ([-np.inf,0.0,0.01,0.01],[np.inf,np.inf,np.inf,np.inf])
        try:
            gauss, _ = curve_fit(models.gaussian, self.region_freq, self.region_pow, p0=guesses, bounds=bb, maxfev=1000)
        except RuntimeError as _:
            self.converge = False
            if self.i == 0:
                raise ProcessingError("Gaussian fit for numax failed to converge.\n\nPlease check your power spectrum and try again.", width=self.width)
        else:
            if self.i == 0:
                # Create an array with finer resolution for plotting
                new_freq = np.linspace(min(self.region_freq), max(self.region_freq), 10000)
                self.params['plotting'][self.module].update({'pssm':np.copy(self.pssm),'obs_numax':self.params['numax_smoo'],'new_freq':new_freq,
                                                             'numax_fit':models.gaussian(new_freq, *gauss),'exp_dnu':self.params['exp_dnu'],
                                                             'region_freq':np.copy(self.region_freq),'region_pow':np.copy(self.region_pow),})
            # Save values
            self.params['results'][self.module]['numax_gauss'].append(gauss[2])
            self.params['results'][self.module]['A_gauss'].append(gauss[1])
            self.params['results'][self.module]['FWHM'].append(gauss[3])

    def compute_acf(self, fft=True, smooth_ps=2.5,):
        r"""ACF

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
        r"""Estimate :math:`\\Delta\\nu`

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
            ProcessingError
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
            pl, pa, _ = utils.max_elements(self.lag, self.auto, npeaks=self.params['n_peaks'], distance=self.params['exp_dnu']/4.)
            # Get peaks from ACF by providing dnu to weight the array 
            peaks_l, peaks_a, weights = utils.max_elements(self.lag, self.auto, npeaks=self.params['n_peaks'], distance=self.params['exp_dnu']/4., exp_dnu=self.params['exp_dnu'])
            # Pick "best" peak in ACF (i.e. closest to expected dnu)
            idx , self.params['best_lag'], self.params['best_auto'] = utils.return_max(peaks_l, peaks_a, exp_dnu=self.params['exp_dnu'])
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
                raise ProcessingError("Gaussian fit for dnu failed to converge.\n\nPlease check your power spectrum and try again.", width=self.width)
        # if fit converged, save appropriate results
        else:
            self.params['results'][self.module]['dnu'].append(gauss[2]) 
            if self.i == 0:
                self.params['obs_dnu'] = gauss[2]
                idx, _, _ = utils.return_max(pl, pa, exp_dnu=self.params['obs_dnu'])
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
        r"""Echelle diagram

        Calculates everything required to plot an :term:`echelle diagram` **Note:** this does not
        currently have the `get_ridges` method attached (i.e. not optimizing the spacing or stepechelle)

        Parameters
            smooth_ech : float
                value to smooth (i.e. convolve) ED by
            nox : int
                number of grid points in x-axis of echelle diagram 
            noy : str
                number of orders (y-axis) to plot in echelle diagram
            npb : int
                option to provide the number of points per bin as opposed to an arbitrary value (calculated from spacing and frequency resolution)
            nshift : int
                number of orders to shift echelle diagram (i.e. + is up, - is down)
            hey : bool
                plugin for Dan Hey's echelle package **(not currently implemented)**
            clip_value : float
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
        
	fmin, fmax = self.params['numax_smoo']-(use_dnu*(ny/2.))+(use_dnu*(nshift+0)), self.params['numax_smoo']+(use_dnu*(ny/2.))+(use_dnu*(nshift+1))
        fstart     = fmin-(fmin%use_dnu)
        zoom_freq, zoom_pow = self.zoom_freq, self.zoom_pow
        
        x = np.linspace(0, 2*use_dnu, 2*nx)
        z = np.zeros([ny+1, 2*nx])
        
        for istack in range(ny):
	    z[-istack-1,:] = np.interp(fstart+istack * use_dnu + x, zoom_freq, zoom_pow)
		
	y = fstart + np.arange(0, ny+1, 1)*use_dnu + use_dnu/2
	
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
        r"""Get ridges

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
            if self.bg_corr[mask] != []:
                xx[k] = np.median(modx[mask])
                yy[k] = np.sum(self.bg_corr[mask])
        mask = np.ma.getmask(np.ma.masked_where(yy == 0.0, yy))
        xx, yy = xx[~mask], yy[~mask]
        self.x = np.array(xx.tolist()+list(xx+self.params['plotting'][self.module]['use_dnu']))
        self.y = np.array(list(yy)+list(yy))-min(yy)
        self.params['plotting'][self.module].update({'ed':np.copy(self.ed),'extent':np.copy(self.extent),'x':np.copy(self.x),'y':np.copy(self.y),})
        self.i += 1

    def get_epsilon(self, n_trials=3):
        r"""Get epsilon

        Optimizes the large frequency separation by determining which spacing creates the
        "best" ridges (but is currently under development) think similar to a step-echelle
        but quicker and more hands off? ** WORK IN PROGRESS **

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
        r"""Collapsed ACF

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
        best = self.params['compare2'].index(max(self.params['compare2']))+1

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


class InputError(Exception):
    """Class for pySYD user input errors (i.e., halts execution)."""
    def __init__(self, error, width=60):
        self.msg, self.width = error, width
    def __repr__(self):
        return "pysyd.utils.InputError(error=%r)" % self.msg
    def __str__(self):
        return "<InputError>"

class ProcessingError(Exception):
    """Class for pySYD processing errors (i.e., halts execution)."""
    def __init__(self, error, width=60):
        self.msg, self.width = error, width
    def __repr__(self):
        return "pysyd.utils.ProcessingError(error=%r)" % self.msg
    def __str__(self):
        return "<ProcessingError>"

class InputWarning(Warning):
    """Class for pySYD user input warnings."""
    def __init__(self, warning, width=60):
        self.msg, self.width = warning, width
    def __repr__(self):
        return "pysyd.utils.InputWarning(warning=%r)" % self.msg
    def __str__(self):
        return "<InputWarning>"

class ProcessingWarning(Warning):
    """Class for pySYD user input warnings."""
    def __init__(self, warning, width=60):
        self.msg, self.width = warning, width
    def __repr__(self):
        return "pysyd.utils.ProcessingWarning(warning=%r)" % self.msg
    def __str__(self):
        return "<ProcessingWarning>"
