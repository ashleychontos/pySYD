import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft

from pysyd import models
from pysyd import utils
from pysyd import plots



import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft

from pysyd import models
from pysyd import utils
from pysyd import plots


#####################################################################
# Each star or "target" that is processed with pySYD
# -> initialization loads in data for a single star
#    but will not execute the main module unless it
#    is in the proper pySYD mode
#

class Target:
    """
    A pySYD pipeline target. Initialization stores all the relevant information and
    checks/loads in data for the given target. pySYD no longer requires BOTH the time
    series data and the power spectrum, but requires additional information via CLI if
    the former is not provided i.e. cadence or nyquist frequency, the oversampling
    factor (if relevant), etc.
        
    Attributes:
        star : object
            the star ID
        params : Dict[str,object]
            the pipeline parameters
        excess : Dict[str,object]
            the parameters of the find excess routine
        background : Dict[str,object]
            the parameters relevant for the background-fitting procedure
        globe : Dict[str,object]
            parameters relevant for estimating global asteroseismic parameters numax and dnu
        verbose : bool
            if true, turns on the verbose output
    
    Parameters:
        args : argparse.Namespace
            the parsed and updated command line arguments
    
    Methods:
        pysyd.target.Target.collapsed_acf()
        pysyd.target.Target.collapsed_acf
        target.Target.collapsed_acf()
        target.Target.collapsed_acf
    
    
    """

    def __init__(self, star, args):
        """
        Creates a `pysyd.target.Target` object for each star that is processed.
        """
        self.name = star
        self.params, self.excess, self.background, self.globe, self.verbose = \
            args.params, args.excess, args.background, args.globe, args.verbose
        self = utils.load_data(self, args)


#####################################################################
# Run pypline target
#

    def run_syd(self):
        """
        Run the pySYD pipeline routines sequentially:
        1) the find excess module to identify the any solar-like oscillations
        2) estimates the stellar background contributions before estimating the
           global asteroseismic parameters
        Returns
        -------
        None
        """
        # Run the (optional) estimate numax module
        if self.params[self.name]['excess']:
            if not self.get_estimate():
                return
        # Run the global fitting routine
        self.derive_parameters()
        if self.params['show']:
            note=''
            if self.verbose:
                note+=' - displaying figures'
            print(note)
            plt.show(block=False)
            input(' - press RETURN to exit')
            if not self.verbose:
                print('')
        if self.params['testing']:
            print(self.test)


#####################################################################
# Method to identify power excess due to solar-like oscillations 
# (optional, if not already known)
#

    def get_estimate(self):
        """
        Automatically finds power excess due to solar-like oscillations using a
        frequency-resolved, collapsed autocorrelation function (ACF).
        Returns
        -------
        None
        """
        self = utils.get_estimates(self)
        # Make sure the binning is specified, otherwise it cannot run
        if self.excess['binning'] is not None:
            # Smooth the power in log-space
            self.bin_freq, self.bin_pow, self.bin_pow_err = utils.bin_data(self.freq, self.pow, width=self.excess['binning'], log=True, mode=self.excess['mode'])
            # Smooth the power in linear-space
            self.smooth_freq, self.smooth_pow, self.smooth_pow_err = utils.bin_data(self.bin_freq, self.bin_pow, width=self.excess['smooth_width'])
            if self.verbose:
                print('------------------------------------------------------')
                print('Estimating numax:')
                print('PS binned to %d datapoints' % len(self.smooth_freq))
            # Interpolate and divide to get a crude background-corrected power spectrum
            s = InterpolatedUnivariateSpline(self.smooth_freq, self.smooth_pow, k=1)
            self.interp_pow = s(self.freq)
            self.bgcorr_pow = self.pow/self.interp_pow
            # Calculate collapsed ACF using different box (or bin) sizes
            self.excess['results'][self.name] = {}
            self.compare = []
            for b in range(self.excess['n_trials']):
                self.collapsed_acf(b)
            # Select trial that resulted with the highest SNR detection
            if not self.excess['ask']:
                self.excess['results'][self.name]['best'] = self.compare.index(max(self.compare))+1
                if self.verbose:
                    print('Selecting model %d'%self.excess['results'][self.name]['best'])
                plots.plot_estimates(self)
                utils.save_estimates(self)
                return True
            # Or ask which estimate to use
            else:
                plots.plot_estimates(self, block=True)
                value = utils.ask_int('Which estimate would you like to use? ', self.excess['n_trials'])
                if value is None:
                    print('Please try again with a valid selection.')
                    print('------------------------------------------------------\n\n')
                    return False
                else:
                    if isinstance(value, int):
                        self.excess['results'][self.name]['best'] = value
                        print('Selecting model %d'%value)
                        utils.save_estimates(self)
                    else:
                        self.params[self.name]['numax'] = value
                        self.params[self.name]['dnu'] = utils.delta_nu(self.params[self.name]['numax'])
                        print('Using numax of %.2f muHz as intial guess'%value)
                    return True
        return False


#####################################################################
# Method to automatically find numax given a power spectrum
#

    def collapsed_acf(self, b, start=0, max_iterations=5000, max_snr=100.):
        """
        Computes a collapsed autocorrelation function (ACF).
        Parameters
        ----------
        b : int
            the trial number
        start : int
            what index of the frequency array to start with, which is `0` by default.
        max_iterations : int
            maximum number of times to run the scipy.optimization before calling it quits
        j : int
            index at which to start storing the cumulative sum and mean of ACF. Default value is `0`.
        start : int
            index at which to start masking the frequency and power spectrum. Default value is `0`.
        max_iterations : int
            maximum number of interations to try in curve fitting routine. Default value is `5000`.
        max_snr : float
            maximum SNR corresponding to power excess. Default value is `100.0`.
        Returns
        -------
        None
        """
        constants = utils.Constants()
        # Computes a collapsed ACF using different "box" (or bin) sizes
        self.excess['results'][self.name][b+1] = {}
        subset = np.ceil(self.boxes[b]/self.resolution)
        steps = np.ceil((self.boxes[b]*self.excess['step'])/self.resolution)

        cumsum, md = [], []
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
        md = np.array(md)
        cumsum = np.array(cumsum)-min(cumsum)
        csum = list(cumsum/max(cumsum))
        # Pick the maximum value as an initial guess for numax
        idx = csum.index(max(csum))
        csum = np.array(csum)
        self.excess['results'][self.name][b+1].update({'x':md,'y':csum,'maxx':md[idx],'maxy':csum[idx]})
        # Fit Gaussian to get estimate value for numax
        try:
            best_vars, _ = curve_fit(models.gaussian, md, csum, p0=[np.median(csum), 1.0-np.median(csum), md[idx], constants.width_sun*(md[idx]/constants.numax_sun)], maxfev=max_iterations, bounds=((-np.inf,-np.inf,1,-np.inf),(np.inf,np.inf,np.inf,np.inf)),)
        except Exception as _:
            self.excess['results'][self.name][b+1].update({'good_fit':False})
            snr = 0.
        else:
            self.excess['results'][self.name][b+1].update({'good_fit':True})
            fitx = np.linspace(min(md), max(md), 10000)
            fity = models.gaussian(fitx, *best_vars)
            self.excess['results'][self.name][b+1].update({'fitx':fitx,'fity':fity})
            snr = max(fity)/np.absolute(best_vars[0])
            if snr > max_snr:
                snr = max_snr
            self.excess['results'][self.name][b+1].update({'numax':best_vars[2],'dnu':utils.delta_nu(best_vars[2]),'snr':snr})
            if self.verbose:
                  print('Numax estimate %d: %.2f +/- %.2f'%(b+1, best_vars[2], np.absolute(best_vars[3])/2.0))
                  print('S/N: %.2f' % snr)
        self.compare.append(snr)


#####################################################################
# Where all the pySYD parameters are derived
#

    def derive_parameters(self):
        """
        The main pySYD pipeline routine. First it 
        Perform a fit to the granulation background and measures the frequency of maximum power (numax),
        the large frequency separation (dnu) and oscillation amplitude.
        Returns
        -------
        None
        """
        # Get initial guesses for relevant parameters
        self = utils.get_initial(self)
        while self.i < self.background['mc_iter']:
            if self.params['background']:
                # Did background fit converge
                success = self.fit_background()
            else:
                success = True
                self.get_white_noise()
                self.bg_corr = np.copy(self.random_pow)/self.noise
                self.pars = ([self.noise])
            if self.params['global']:
                self.fit_global()
            if success:
                if self.i == 0:
                    # Plot results
                    plots.plot_parameters(self)
                    if self.background['mc_iter'] > 1:
                        # Switch to critically-sampled PS if sampling
                        mask = np.ma.getmask(np.ma.masked_inside(self.freq_cs, self.params[self.name]['bg_mask'][0], self.params[self.name]['bg_mask'][1]))
                        self.frequency, self.power = np.copy(self.freq_cs[mask]), np.copy(self.pow_cs[mask])
                        self.resolution = self.frequency[1]-self.frequency[0]
                        self.random_pow = (np.random.chisquare(2, len(self.frequency))*self.power)/2.
                        if self.verbose:
                            from tqdm import tqdm 
                            print('------------------------------------------------------\nRunning sampling routine:')
                            self.pbar = tqdm(total=self.background['mc_iter'])
                            self.pbar.update(1)
                else:
                    self.random_pow = (np.random.chisquare(2, len(self.frequency))*self.power)/2.
                    if self.verbose:
                        self.pbar.update(1)
                self.i += 1
                if self.i == self.background['mc_iter'] and self.background['mc_iter'] > 1 and self.verbose:
                    self.pbar.close()
        # Save derived parameters
        utils.save_results(self)
        if self.background['mc_iter'] > 1:
            # Plot results if sampling
            plots.plot_samples(self)
        if self.verbose:
            # Print results
            utils.verbose_output(self)


#####################################################################
# Functions related to background-fitting
#

    def fit_background(self):
        """
        Fits the stellar background contribution due to granulation. 
        Returns
        -------
        None
        """
        # Bin power spectrum to model stellar background/correlated red noise components
        bin_freq, bin_pow, bin_err = utils.bin_data(self.frequency, self.random_pow, width=self.background['ind_width'], mode=self.excess['mode'])
        # Mask out region with power excess
        mask = np.ma.getmask(np.ma.masked_outside(bin_freq, self.params[self.name]['ps_mask'][0], self.params[self.name]['ps_mask'][1]))
        self.bin_freq, self.bin_pow, self.bin_err = bin_freq[mask], bin_pow[mask], bin_err[mask]
        if self.i == 0 and self.verbose:
            print('------------------------------------------------------')
            print('Determining background model:')
            print('PS binned to %d data points'%len(bin_freq))
        if self.i == 0 and self.params['testing']:
            self.test+='%d/%d points are being used for background fit\n'%(np.sum(mask),len(bin_freq))
        # Estimate white noise level
        self.get_white_noise()
        # Get initial guesses for the optimization of the background model
        self.estimate_initial_red()
        # If optimization does not converge, the rest of the code will not run
        if self.i == 0:
            if self.get_best_model():
                print('WARNING: Bad initial fit for star %d. Check this and try again.'%self.name)
                return False
        else:
            if self.get_red_noise():
                return False
        return True


    def _get_white_noise(self):
        """
        Estimate the white noise level (in muHz) by taking the mean of
        the last 10% of the power spectrum.
        Returns
        -------
        None
        """
        mask = (self.frequency > (max(self.frequency)-0.1*max(self.frequency)))&(self.frequency < max(self.frequency))
        self.noise = np.mean(self.random_pow[mask])


    def _estimate_initial_red(self, a=[]):
        """
        Estimates amplitude of red noise components by using a smoothed version of the power
        spectrum with the power excess region masked out. This will take the mean of a specified 
        number of points (via -nrms, default=20) for each Harvey-like component.
        Parameters
        ----------
        a : List[float]
            initial guesses for the amplitudes of all Harvey components
        Returns
        -------
        None
        """
        # Exclude region with power excess and smooth to estimate red noise components
        boxkernel = Box1DKernel(int(np.ceil(self.background['box_filter']/self.resolution)))
        mask = (self.frequency >= self.params[self.name]['ps_mask'][0])&(self.frequency <= self.params[self.name]['ps_mask'][1])
        self.smooth_pow = convolve(self.random_pow, boxkernel)
        # Temporary array for inputs into model optimization
        self.guesses = np.zeros((self.nlaws*2+1))
        # Estimate amplitude for each harvey component
        for n, nu in enumerate(self.mnu):
            diff = list(np.absolute(self.frequency-nu))
            idx = diff.index(min(diff))
            if idx < self.background['n_rms']:
                self.guesses[2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][:self.background['n_rms']]))/(4.*self.b[n]))
            elif (len(self.smooth_pow[~mask])-idx) < self.background['n_rms']:
                self.guesses[2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][-self.background['n_rms']:]))/(4.*self.b[n]))
            else:
                self.guesses[2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][idx-int(self.background['n_rms']/2):idx+int(self.background['n_rms']/2)]))/(4.*self.b[n]))
            self.guesses[2*n] = self.b[n]
            a.append(self.guesses[2*n+1])
        self.guesses[-1] = self.noise
        self.a_orig = np.array(a)


    def _get_best_model(self):
        """
        Determines the best-fit model for the stellar granulation background in the power spectrum
        by iterating through several models, where the initial guess for the number of Harvey-like 
        component(s) to model is estimated from a solar scaling relation.
        Parameters
        ----------
        bounds : list
            the bounds on the Harvey parameters for a given model
        bic : list
            the BIC statistic
        aic : list
            the AIC statistic
        paras : list
            the fitted parameters for each model that was explored
        Returns
        -------
        again : bool
            will return `True` if fitting failed and the iteration must be repeated, otherwise `False`.
        """
        # Get best-fit model
        self.bounds, self.bic, self.aic, self.paras = [], [], [], []
        if self.params['testing']:
            self.test+='\n    ----------------------------------------------\n    ------------- MODEL COMPARISONS --------------\n    ----------------------------------------------'
        if self.background['n_laws'] is not None:
            if not self.background['fix_wn']:
                self.models = [self.background['n_laws']*2, self.background['n_laws']*2+1]
            else:
                self.models = [self.background['n_laws']*2]
        else:
            if self.background['fix_wn']:
                self.models = np.arange(0,(self.nlaws+1)*2,2)
            else:
                self.models = np.arange((self.nlaws+1)*2)
        if self.verbose and len(self.models) > 1:
            print('Comparing %d different models:'%len(self.models))
        for n, n_free in enumerate(self.models):
            note = 'Model %d: %d Harvey-like component(s) + '%(n, n_free//2)
            if self.background['basis'] == 'a_b':
                bounds = ([0.0,0.0]*(n_free//2), [np.inf,self.tau_upper]*(n_free//2))
            else:
                bounds = ([0.0,0.0]*(n_free//2), [np.inf,np.inf]*(n_free//2))
            if not n_free%2:
                note += 'white noise fixed'
                *guesses, = self.guesses[-int(2*(n_free//2)+1):-1]
            else:
                note += 'white noise term'
                bounds[0].append(10.**-2)
                bounds[1].append(np.inf)
                *guesses, = self.guesses[-int(2*(n_free//2)+1):]
            self.bounds.append(bounds)
            if n_free == 0:
                self.paras.append([])
                model = np.ones_like(self.bin_pow)*self.noise
                b, a = models.compute_bic(self.bin_pow, model, n_parameters=n_free), models.compute_aic(self.bin_pow, model, n_parameters=n_free)
                self.bic.append(b)
                self.aic.append(a)
            else:
                try:
                    if not n_free%2:
                        # If white noise is fixed, pass "noise" estimate to lambda operator
                        pars, _ = curve_fit(self.background['functions'][n_free](self.noise), self.bin_freq, self.bin_pow, p0=guesses, sigma=self.bin_err, bounds=bounds)
                    else:
                        # If white noise is a free parameter, good to go!
                        pars, _ = curve_fit(self.background['functions'][n_free], self.bin_freq, self.bin_pow, p0=guesses, sigma=self.bin_err, bounds=bounds)
                except RuntimeError as _:
                    self.paras.append([])
                    self.bic.append(np.inf)
                    self.aic.append(np.inf)
                else:
                    self.paras.append(pars)
                    model = models.background(self.bin_freq, pars, noise=self.noise)
                    b, a = models.compute_bic(self.bin_pow, model, n_parameters=n_free), models.compute_aic(self.bin_pow, model, n_parameters=n_free)
                    self.bic.append(b)
                    self.aic.append(a)
            if self.verbose:
                if self.background['include']:
                    note += '\n BIC = %.2f | AIC = %.2f'%(b, a)
                print(note)
            if self.params['testing']:
                note += '\n    BIC = %.2f | AIC = %.2f'%(b, a)
                self.test += '\n   %s'%note
        # If the fitting converged (fix to bic? depending on performance)
        if self.background['metric'] == 'bic':
            if np.isfinite(min(self.bic)):
                self.save_best_model()
                return False
            else:
                # If fit did not converge, run again (i.e. True)
                return True
        else:
            if np.isfinite(min(self.aic)):
                self.save_best_model()
                return False
            else:
                return True


    def _save_best_model(self, use='bic', test=True):
        """
        Saves information re: the selected best-fit model (for the stellar background).
        
        Parameters
        ----------
        use : str
            which metric to use for model selection, choices ~['bic','aic']. Default is `'bic'`.
        """
        if self.background['metric'] == 'bic':
            idx = self.bic.index(min(self.bic))
            self.model = self.models[idx]
        else:
            idx = self.aic.index(min(self.aic))
            self.model = self.models[idx]
        self.bounds = self.bounds[idx]
        self.pars = self.paras[idx]
        # If model with fixed white noise is preferred, change 'fix_wn' option
        if not int(self.model%2):
            self.background['fix_wn'] = True
        # Store model results for plotting
        if self.nlaws != self.model//2:
            self.nlaws = self.model//2
            self.b = self.b[:(self.nlaws)]
            self.mnu = self.mnu[:(self.nlaws)]
        if self.verbose and len(self.models) > 1:
            print('Based on %s statistic: model %d'%(use.upper(),idx))
        # Compare different model results
        if self.params['testing'] and len(self.models) > 1:
            plots.plot_fits(self)
        self.bg_corr = self.random_pow/models.background(self.frequency, self.pars, noise=self.noise)
        # Save background-corrected power spectrum
        utils.save_file(self)
        # Create appropriate keys for star based on best-fit model
        for n in range(self.nlaws):
            self.background['results'][self.name]['tau_%d'%(n+1)] = []
            self.background['results'][self.name]['sigma_%d'%(n+1)] = []
        if not self.background['fix_wn']:
            self.background['results'][self.name]['white'] = []
        # Save the final values
        for n in range(self.nlaws):
            self.background['results'][self.name]['tau_%d'%(n+1)].append(self.pars[2*n]*10.**6)
            self.background['results'][self.name]['sigma_%d'%(n+1)].append(self.pars[2*n+1])
        if not self.background['fix_wn']:
            self.background['results'][self.name]['white'].append(self.pars[-1])


    def _get_red_noise(self):
        """
        Calculates red noise level, or stellar background contribution, from power spectrum.
        Returns
        -------
        result : bool
            will return `False` if model converges, otherwise `True`.
        
        """
        if self.model != 0:
            # Use as initial guesses for the optimized model
            try:
                if self.background['fix_wn']:
                    # If white noise is fixed, pass "noise" estimate to lambda operator
                    self.pars, _ = curve_fit(self.background['functions'][self.model](self.noise), self.bin_freq, self.bin_pow, p0=self.guesses[:-1], sigma=self.bin_err, bounds=self.bounds)
                else:
                    # If white noise is a free parameter, good to go!
                    self.pars, _ = curve_fit(self.background['functions'][self.model], self.bin_freq, self.bin_pow, p0=self.guesses, sigma=self.bin_err, bounds=self.bounds)
            except RuntimeError as _:
                return True
            else:
                self.bg_corr = self.random_pow/models.background(self.frequency, self.pars, noise=self.noise)
                # save final values for Harvey components
                for n in range(self.nlaws):
                    self.background['results'][self.name]['tau_%d'%(n+1)].append(self.pars[2*n]*10.**6)
                    self.background['results'][self.name]['sigma_%d'%(n+1)].append(self.pars[2*n+1])
                if not self.background['fix_wn']:
                    self.background['results'][self.name]['white'].append(self.pars[-1])
            return False
        else:
            self.bg_corr = self.random_pow/self.noise
            return False


#####################################################################
# Main function for determining global seismic parameters
#

    def fit_global(self):
        """
        The main pySYD pipeline routine. First it 
        Perform a fit to the granulation background and measures the frequency of maximum power (numax),
        the large frequency separation (dnu) and oscillation amplitude.
        Returns
        -------
        None
        """
        # get numax
        self.get_numax_smooth()
        self.get_numax_gaussian()
        # get dnu
        self.compute_acf()
        if self.i == 0:
            if not self.params['background']:
                boxkernel = Box1DKernel(int(np.ceil(self.background['box_filter']/self.resolution)))
                self.smooth_pow = convolve(self.random_pow, boxkernel)
                if self.verbose:
                    print('------------------------------------------------------\nEstimating global parameters from raw PS:')
            self.initial_dnu()
            self.get_acf_cutout()
            self.get_ridges()
        else:
            # define the peak in the ACF
            zoom_lag = self.lag[(self.lag>=self.params[self.name]['acf_mask'][0])&(self.lag<=self.params[self.name]['acf_mask'][1])]
            zoom_auto = self.auto[(self.lag>=self.params[self.name]['acf_mask'][0])&(self.lag<=self.params[self.name]['acf_mask'][1])]
            # fit a Gaussian function to the selected peak in the ACF
            gauss, _ = curve_fit(models.gaussian, zoom_lag, zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb)
            # the center of that Gaussian is our estimate for Dnu
            self.globe['results'][self.name]['dnu'].append(gauss[2]) 


#####################################################################
# Functions related to estimating numax
#

    def get_numax_smooth(self, divide=True):
        """
        Estimate numax by smoothing the power spectrum and taking the peak. 
        Returns
        -------
        None
        """
        constants = utils.Constants()
        # Smoothing width for determining numax
        if self.globe['sm_par'] is not None:
            self.sm_par = self.globe['sm_par']
        else:
            self.sm_par = 4.*(self.params[self.name]['numax']/constants.numax_sun)**0.2
            if self.sm_par < 1.:
                self.sm_par = 1.
        sig = (self.sm_par*(self.params[self.name]['dnu']/self.resolution))/np.sqrt(8.0*np.log(2.0))
        self.pssm = convolve_fft(np.copy(self.random_pow), Gaussian1DKernel(int(sig)))
        self.pssm_bgcorr = self.pssm-models.background(self.frequency, self.pars, noise=self.noise)
        mask = np.ma.getmask(np.ma.masked_inside(self.frequency, self.params[self.name]['ps_mask'][0], self.params[self.name]['ps_mask'][1]))
        if not np.sum(mask):
            self.region_freq, self.region_pow = self.frequency[~mask], self.pssm_bgcorr[~mask]
        else:
            self.region_freq, self.region_pow = self.frequency[mask], self.pssm_bgcorr[mask]
        if self.params['testing'] and self.i == 0:
            self.test+='\n------------------------------------------------------ \nThe power spectrum mask includes %d data points \n i.e. the power excess region ~[%.2f,%.2f]'%(np.sum(mask), min(self.region_freq), max(self.region_freq))
        idx = utils.return_max(self.region_freq, self.region_pow, index=True)
        self.globe['results'][self.name]['numax_smooth'].append(self.region_freq[idx])
        self.globe['results'][self.name]['A_smooth'].append(self.region_pow[idx])
        # Initial guesses for the parameters of the Gaussian fit to the power envelope
        self.obs_numax = self.region_freq[idx]
        self.exp_dnu = 0.22*(self.obs_numax**0.797)


    def get_numax_gaussian(self, maxfev=5000):
        """
        Estimate numax by fitting a Gaussian to the power envelope of the smoothed power spectrum.
    
        Parameters
        ----------
        maxfev : int, optional
            maximum number of attempts for the scipy.curve_fit optimization step
        Returns
        -------
        None
        """
        guesses = [0.0,max(self.region_pow),self.obs_numax,(max(self.region_freq)-min(self.region_freq))/np.sqrt(8.0*np.log(2.0))]
        bb = ([-np.inf,0.0,min(self.region_freq),0.01],[np.inf,np.inf,max(self.region_freq),max(self.region_freq)-min(self.region_freq)])
        gauss, _ = curve_fit(models.gaussian, self.region_freq, self.region_pow, p0=guesses, bounds=bb, maxfev=maxfev)
        # Create an array with finer resolution for plotting
        self.new_freq = np.linspace(min(self.region_freq), max(self.region_freq), 10000)
        self.numax_fit = models.gaussian(self.new_freq, *gauss)
        # Save values
        self.globe['results'][self.name]['numax_gauss'].append(gauss[2])
        self.globe['results'][self.name]['A_gauss'].append(gauss[1])
        self.globe['results'][self.name]['FWHM'].append(gauss[3])


#####################################################################
# Functions related to the characteristic frequency spacing (dnu)
#

    def compute_acf(self, fft=True):
        """
        Compute the ACF of the smooth background corrected power spectrum.
        Parameters
        ----------
        fft : bool
            if true will use FFT to compute the ACF. Default value is `True`.
        Returns
        -------
        None
        """
        # Optional smoothing of PS to remove fine structure before computing ACF
        if int(self.globe['smooth_ps']) != 0:
            boxkernel = Box1DKernel(int(np.ceil(self.globe['smooth_ps']/self.resolution)))
            self.bg_corr_smooth = convolve(self.bg_corr, boxkernel)
        else:
            self.bg_corr_smooth = np.copy(self.bg_corr)
        # Use only power near the expected numax to reduce additional noise in ACF
        power = self.bg_corr_smooth[(self.frequency >= self.params[self.name]['ps_mask'][0])&(self.frequency <= self.params[self.name]['ps_mask'][1])]
        lag = np.arange(0.0, len(power))*self.resolution
        if fft:
            auto = np.real(np.fft.fft(np.fft.ifft(power)*np.conj(np.fft.ifft(power))))
        else:
            auto = np.correlate(power-np.mean(power), power-np.mean(power), "full")
            auto = auto[int(auto.size/2):]
        mask = np.ma.getmask(np.ma.masked_inside(lag, self.params[self.name]['dnu']/4., 2.*self.params[self.name]['dnu']+self.params[self.name]['dnu']/4.))
        lag = lag[mask]
        auto = auto[mask]
        auto -= min(auto)
        auto /= max(auto)
        self.lag = np.copy(lag)
        self.auto = np.copy(auto)


    def initial_dnu(self, method='D'):
        """
        More modular functions to estimate dnu on the first iteration given
        different methods. By default, we have been using a Gaussian weighting
        centered on the expected value for dnu (determine from the pipeline).
        One can also "force" or provide a value for dnu.
        Parameters
        ----------
        method : str
            which method to use, where: 
            - 'M' == Maryum == scipy's find_peaks module
            - 'A' == Ashley == Ashley's module from the functions script
            - 'D' == Dennis == weighting technique
        guess : Optional[float]
            option to "force" a dnu value, but is `None` by default.
        """
        if self.params[self.name]['force']:
            guess = self.params[self.name]['guess']
        else:
            guess = self.exp_dnu
        if self.globe['method'] == 'M':
            # Get peaks from ACF using scipy package
            peak_idx, _ = find_peaks(self.auto)
            peaks_l0, peaks_a0 = self.lag[peak_idx], self.auto[peak_idx]
            # Pick n highest peaks
            self.peaks_l, self.peaks_a = utils.max_elements(peaks_l0, peaks_a0, npeaks=self.globe['n_peaks'])
        elif self.globe['method'] == 'A':
            # Get peaks from ACF using Ashley's module 
            self.peaks_l, self.peaks_a = utils.max_elements(self.lag, self.auto, npeaks=self.globe['n_peaks'])
        elif self.globe['method'] == 'D':
            # Get peaks from ACF by providing dnu to weight the array (aka Dennis' routine)
            self.peaks_l, self.peaks_a = utils.max_elements(self.lag, self.auto, npeaks=self.globe['n_peaks'], exp_dnu=guess)
        else:
            pass
        # Pick "best" peak in ACF (i.e. closest to expected dnu)
        idx = utils.return_max(self.peaks_l, self.peaks_a, index=True, exp_dnu=guess)
        self.best_lag, self.best_auto = self.peaks_l[idx], self.peaks_a[idx]
        if self.params['testing'] and self.i == 0:
            self.test+='\n %s method to estimate dnu ~= %.2f'%(self.globe['method'], self.best_lag)
        # If a dnu is provided, overwrite the value determined by pysyd
        if self.params[self.name]['force']:
            if self.params['testing'] and self.i == 0:
                self.test+='\n ... BUT value provided, dnu~=%.2f'%(self.params[self.name]['guess'])
            if self.params[self.name]['guess'] != self.best_lag:
                yc = list(np.absolute(self.lag-self.params[self.name]['guess']))
                idx = yc.index(min(yc))
                self.best_lag = self.lag[idx]
                self.best_auto = self.auto[idx]


    def get_acf_cutout(self, threshold=1.0):
        """
        Gets the region in the ACF centered on the correct peak.
        Parameters
        ----------
        threshold : float
            the threshold is multiplied by the full-width half-maximum value, centered on the peak 
            in the ACF to determine the width of the cutout region.
        
        Returns
        -------
        None
        """
        # Calculate FWHM
        if list(self.lag[(self.lag<self.best_lag)&(self.auto<=self.best_auto/2.)]) != []:
            left_lag = self.lag[(self.lag<self.best_lag)&(self.auto<=self.best_auto/2.)][-1]
            left_auto = self.auto[(self.lag<self.best_lag)&(self.auto<=self.best_auto/2.)][-1]
        else:
            left_lag = self.lag[0]
            left_auto = self.auto[0]
        if list(self.lag[(self.lag>self.best_lag)&(self.auto<=self.best_auto/2.)]) != []:
            right_lag = self.lag[(self.lag>self.best_lag)&(self.auto<=self.best_auto/2.)][0]
            right_auto = self.auto[(self.lag>self.best_lag)&(self.auto<=self.best_auto/2.)][0]
        else:
            right_lag = self.lag[-1]
            right_auto = self.auto[-1]
        # Lag limits to use for ACF mask or "cutout"
        self.params[self.name]['acf_mask']=[self.best_lag-(self.best_lag-left_lag)*self.globe['threshold'],self.best_lag+(right_lag-self.best_lag)*self.globe['threshold']]
        self.zoom_lag = self.lag[(self.lag>=self.params[self.name]['acf_mask'][0])&(self.lag<=self.params[self.name]['acf_mask'][1])]
        self.zoom_auto = self.auto[(self.lag>=self.params[self.name]['acf_mask'][0])&(self.lag<=self.params[self.name]['acf_mask'][1])]
        # If dnu is already provided
        if self.params[self.name]['force']:
            guesses = [np.mean(self.zoom_auto), self.best_auto, self.best_lag*0.01*2.]
            bb = ([-np.inf,0.,10**-2.],[np.inf,np.inf,2.*(max(self.zoom_lag)-min(self.zoom_lag))]) 
            # Fix center of Gaussian on desired spacing
            p_gauss, _ = curve_fit(lambda frequency, offset, amplitude, width: models.gaussian(frequency, offset, amplitude, self.best_lag, width), self.zoom_lag, self.zoom_auto, p0=guesses, bounds=bb) 
            if self.params['testing'] and self.i == 0:
                self.test+='\n values from fixed fit: \n %s'%list(p_gauss)
            gauss = [p_gauss[0], p_gauss[1], self.best_lag, p_gauss[2]]
            if self.params['testing'] and self.i == 0:
                self.test+='\n\n -- \n then adjusted for rest of run: \n %s'%list(gauss)
            # Readjust boundaries for 4-parameter Gaussian fit
            self.acf_guesses = [np.mean(self.zoom_auto), self.best_auto, self.best_lag, self.best_lag*0.01*2.]
            self.acf_bb = ([-np.inf,0.,min(self.zoom_lag),10**-2.],[np.inf,np.inf,max(self.zoom_lag),2.*(max(self.zoom_lag)-min(self.zoom_lag))]) 
        else:
           	# Boundary conditions and initial guesses stay the same for all iterations
            self.acf_guesses = [np.mean(self.zoom_auto), self.best_auto, self.best_lag, self.best_lag*0.01*2.]
            self.acf_bb = ([-np.inf,0.,min(self.zoom_lag),10**-2.],[np.inf,np.inf,max(self.zoom_lag),2.*(max(self.zoom_lag)-min(self.zoom_lag))]) 
            # Fit a Gaussian function to the selected peak in the ACF to get dnu
            gauss, _ = curve_fit(models.gaussian, self.zoom_lag, self.zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb)
        self.obs_dnu = gauss[2]
        self.globe['results'][self.name]['dnu'].append(self.obs_dnu)
        # Save for plotting
        self.new_lag = np.linspace(min(self.zoom_lag),max(self.zoom_lag),2000)
        self.dnu_fit = models.gaussian(self.new_lag,*gauss)
        self.obs_acf = max(self.dnu_fit)


#####################################################################
# Functions related to making the echelle diagram
#

    def _get_ridges(self):
        """
        Create echelle diagram.
        Parameters
        ----------
        clip_value : float
            lower limit of distance modulus. Default value is `0.0`.
        Returns
        -------
        None
        """
        self.echelle()
        copy = self.z.flatten()
        n = int(np.ceil(self.obs_dnu/self.resolution))
        xax = np.linspace(0.0, self.obs_dnu, n)
        yax = np.zeros_like(xax)
        modx = self.frequency%self.obs_dnu
        for k in range(n-1):
            mask = (modx >= xax[k])&(modx < xax[k+1])
            if self.bg_corr[(modx >= xax[k])&(modx < xax[k+1])] != []:
                xax[k] = np.median(modx[(modx >= xax[k])&(modx < xax[k+1])])
                yax[k] = np.sum(self.bg_corr[(modx >= xax[k])&(modx < xax[k+1])])
        mask = np.ma.getmask(np.ma.masked_where(yax == 0.0, yax))
        xax, yax = xax[~mask], yax[~mask]
        self.xax = np.array(xax.tolist()+list(xax+self.obs_dnu))
        self.yax = np.array(list(yax)+list(yax))-min(yax)
        # Clip the lower bound (`clip_value`)
        if int(self.globe['clip_value']) != 0:
            cut = np.nanmedian(copy)+(self.globe['clip_value']*np.nanmedian(copy))
            if self.params['testing']:
                self.test+='\n\n Median of ED: %.2f \n Clip value of ED: %.2f'%(np.nanmedian(copy),cut)
            copy[copy >= cut] = cut
        self.zz = copy
        self.z = copy.reshape((self.z.shape[0], self.z.shape[1]))


    def echelle(self):
        """
        Creates an echelle diagram.
        Parameters
        ----------
        nox : int
            number of grid points in x-axis of echelle diagram. Default value is `50`.
        noy : int
            number of orders (y-axis) to show in echelle diagram. Default value is `5`.
        startx : float
            lower limit of distance modulus. Default value is `0.0`.
        Returns
        -------
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
        if self.globe['smooth_ech'] is not None:
            boxkernel = Box1DKernel(int(np.ceil(self.globe['smooth_ech']/self.resolution)))
            smooth_y = convolve(self.bg_corr, boxkernel)
        else:
            smooth_y = np.copy(self.bg_corr)
        # If the number of desired orders is not provided
        if self.globe['noy'] == 0:
            self.globe['noy'] = int(self.obs_numax/self.obs_dnu//2)
        # Make sure n_across isn't finer than the actual resolution grid
        if self.globe['nox'] >= int(np.ceil(self.obs_dnu/self.resolution)):
            self.globe['nox'] = int(np.ceil(self.obs_dnu/self.resolution/3.))
        nx, ny = (self.globe['nox'], self.globe['noy'])
        self.x = np.linspace(0.0, 2*self.obs_dnu, 2*nx+1)
        yy = np.arange(self.obs_numax%(self.obs_dnu/2.),max(self.frequency),self.obs_dnu)
        lower, upper = self.obs_numax-3*self.obs_dnu/2.-(self.obs_dnu*(ny//2)), self.obs_numax+self.obs_dnu/2.+(self.obs_dnu*(ny//2))
        self.y = yy[(yy >= lower)&(yy <= upper)]
        z = np.zeros((ny+1,2*nx))
        for i in range(1,ny+1):
            y_mask = ((self.frequency >= self.y[i-1]) & (self.frequency < self.y[i]))
            for j in range(nx):
                x_mask = ((self.frequency%(self.obs_dnu) >= self.x[j]) & (self.frequency%(self.obs_dnu) < self.x[j+1]))
                if smooth_y[x_mask & y_mask] != []:
                    z[i][j] = np.sum(smooth_y[x_mask & y_mask])
                else:
                    z[i][j] = np.nan
        z[0][:nx], z[-1][nx:] = np.nan, np.nan
        for k in range(ny):
            z[k][nx:] = z[k+1][:nx]
        self.z = np.copy(z)
        self.extent = [min(self.x),max(self.x),min(self.y),max(self.y)]
