import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft

from pysyd import functions
from pysyd import models
from pysyd import utils
from pysyd import plots



class Target:
    """
    A pySYD pipeline target. Initialization stores all the relevant information and
    checks/loads in data for the given target. pySYD no longer requires BOTH the time
    series data and the power spectrum, but requires additional information via CLI if
    the former is not provided i.e. cadence or nyquist frequency, the oversampling
    factor (if relevant), etc.

    Attributes
    ----------
    star : int
        the star ID
    params : Dict[str,object]
        the pipeline parameters
    findex : Dict[str,object]
        the parameters of the find excess routine
    fitbg : Dict[str,object]
        the parameters of the fit background routine
    verbose : bool
        if true, turns on the verbose output

    Parameters
    ----------
    args : argparse.Namespace
        the parsed and updated command line arguments

    Methods
    -------
    TODO: Add methods

    """

    def __init__(self, star, args):
        self.name = star
        self.params = args.params
        self.findex = args.findex
        self.fitbg = args.fitbg
        self.globe = args.globe
        self.verbose = args.verbose
        self = utils.load_data(self, args)


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
        # Run the find excess routine
        if self.params[self.name]['excess']:
            self = utils.get_findex(self)
            self.find_excess()
        # Run the global fitting routine
        if self.params[self.name]['background']:
            if utils.check_fitbg(self):
                self = utils.get_fitbg(self)
                self.fit_global()


    def find_excess(self):
        """
        Automatically finds power excess due to solar-like oscillations using a
        frequency-resolved, collapsed autocorrelation function (ACF).

        Returns
        -------
        None

        """
        # Make sure the binning is specified, otherwise it cannot run
        if self.findex['binning'] is not None:
            self.bin_freq, self.bin_pow, _ = functions.bin_data(self.freq, self.pow, width=self.findex['binning'], log=True, mode=self.findex['mode'])
            if self.verbose:
                print('----------------------------------------------------')
                print('Running find_excess module:')
                print('PS binned to %d datapoints' % len(self.bin_freq))
                print('Binned freq res: %.2f muHz'%(self.bin_freq[1]-self.bin_freq[0]))
            # Smooth the binned power spectrum for a rough estimate of background
            boxsize=int(np.ceil(self.findex['smooth_width']/(self.bin_freq[1]-self.bin_freq[0])))
            if boxsize >= len(self.bin_freq):
                boxsize=int(np.ceil(len(self.bin_freq)/10.))
            sp = convolve(self.bin_pow, Box1DKernel(boxsize))
            smooth_freq = self.bin_freq[int(boxsize/2):-int(boxsize/2)]
            smooth_pow = sp[int(boxsize/2):-int(boxsize/2)]

            # Interpolate and divide to get a crude background-corrected power spectrum
            s = InterpolatedUnivariateSpline(smooth_freq, smooth_pow, k=1)
            self.interp_pow = s(self.freq)
            self.bgcorr_pow = self.pow/self.interp_pow

            # Calculate collapsed ACF using different box (or bin) sizes
            self.findex['results'][self.name] = {}
            self.compare = []
            for b in range(self.findex['n_trials']):
                self.collapsed_acf(b)

            # Select trial that resulted with the highest SNR detection
            self.findex['results'][self.name]['best'] = self.compare.index(max(self.compare))+1
            if self.verbose:
                print('selecting model %d'%self.findex['results'][self.name]['best'])
            utils.save_findex(self)
            plots.plot_excess(self)


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
        self.findex['results'][self.name][b+1] = {}
        subset = np.ceil(self.boxes[b]/self.resolution)
        steps = np.ceil((self.boxes[b]*self.findex['step'])/self.resolution)

        cumsum = []
        md = []
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
        self.findex['results'][self.name][b+1].update({'x':md,'y':csum,'maxx':md[idx],'maxy':csum[idx]})

        # Fit Gaussian to get estimate value for numax
        try:
            best_vars, _ = curve_fit(models.gaussian, md, csum, p0=[np.median(csum), 1.0-np.median(csum), md[idx], constants.width_sun*(md[idx]/constants.numax_sun)], maxfev=max_iterations, bounds=((-np.inf,-np.inf,1,-np.inf),(np.inf,np.inf,np.inf,np.inf)),)
        except Exception as _:
            self.findex['results'][self.name][b+1].update({'good_fit':False})
            snr = 0.
        else:
            self.findex['results'][self.name][b+1].update({'good_fit':True})
            fitx = np.linspace(min(md), max(md), 10000)
            fity = models.gaussian(fitx, *best_vars)
            self.findex['results'][self.name][b+1].update({'fitx':fitx,'fity':fity})
            snr = max(fity)/np.absolute(best_vars[0])
            if snr > max_snr:
                snr = max_snr
            self.findex['results'][self.name][b+1].update({'numax':best_vars[2],'dnu':functions.delta_nu(best_vars[2]),'snr':snr})
            if self.verbose:
                  print('power excess trial %d: numax = %.2f +/- %.2f'%(b+1, best_vars[2], np.absolute(best_vars[3])/2.0))
                  print('S/N: %.2f' % snr)
        self.compare.append(snr)


    def fit_global(self):
        """
        The second main pySYD pipeline routine. First it 
        Perform a fit to the granulation background and measures the frequency of maximum power (numax),
        the large frequency separation (dnu) and oscillation amplitude.

        Returns
        -------
        None

        """
        while self.i < self.fitbg['mc_iter']:
            # Requires convergence of background fit before going to the next step 
            if self.fit_background():
                self.get_numax()
                self.get_dnu()
                # First step?
                if self.i == 0:
                    # Plot results
                    plots.plot_background(self)
                    if self.fitbg['mc_iter'] > 1:
                        # Switch to critically-sampled PS if sampling
                        mask = np.ma.getmask(np.ma.masked_inside(self.freq_cs, self.params[self.name]['bg_mask'][0], self.params[self.name]['bg_mask'][1]))
                        self.frequency, self.power = np.copy(self.freq_cs[mask]), np.copy(self.pow_cs[mask])
                        self.resolution = self.frequency[1]-self.frequency[0]
                        if self.verbose:
                            print('----------------------------------------------------\nRunning sampling routine:')
                            self.pbar = tqdm(total=self.fitbg['mc_iter'])
                            self.pbar.update(1)
                else:
                    if self.verbose:
                        self.pbar.update(1)
                self.i += 1
                if self.i == self.fitbg['mc_iter'] and self.fitbg['mc_iter'] > 1:
                    self.pbar.close()
        # Save results of second module
        utils.save_fitbg(self)
        if self.fitbg['mc_iter'] > 1:
            # Plot results if sampling
            plots.plot_samples(self)
        if self.verbose:
            # Print results
            utils.verbose_output(self)


    def fit_background(self):
        """
        Fits the stellar background contribution due to granulation. 

        Returns
        -------
        None

        """
        # Bin power spectrum to model stellar background/correlated red noise components
        if self.i != 0:
            self.random_pow = (np.random.chisquare(2, len(self.frequency))*self.power)/2.
        bin_freq, bin_pow, bin_err = functions.bin_data(self.frequency, self.random_pow, width=self.fitbg['ind_width'], mode=self.findex['mode'])
        # Mask out region with power excess
        mask = np.ma.getmask(np.ma.masked_outside(bin_freq, self.params[self.name]['ps_mask'][0], self.params[self.name]['ps_mask'][1]))
        self.bin_freq = bin_freq[mask]
        self.bin_pow = bin_pow[mask]
        self.bin_err = bin_err[mask]

        if self.i == 0 and self.verbose:
            print('----------------------------------------------------')
            print('Running fit_background module:')
            print('PS binned to %d data points' % len(bin_freq))
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


    def get_white_noise(self):
        """
        Estimate the white noise level (in muHz) by taking a mean over a region 
        in the power spectrum near the nyquist frequency.

        Returns
        -------
        None

        """
        if hasattr(self, 'nyquist') and max(self.frequency) > self.nyquist:
            if self.nyquist < 400.:
                mask = (self.frequency > 200.)&(self.frequency < 270.)
                self.noise = np.mean(self.random_pow[mask])
            elif self.nyquist > 400. and self.nyquist < 5000.:
                mask = (self.frequency > 4000.)&(self.frequency < 4167.)
                self.noise = np.mean(self.random_pow[mask])
            elif self.nyquist > 5000. and self.nyquist < 9000.:
                mask = (self.frequency > 8000.)&(self.frequency < 8200.)
                self.noise = np.mean(self.random_pow[mask])
            else:
                pass
        else:
            mask = (self.frequency > (max(self.frequency)-0.1*max(self.frequency)))&(self.frequency < max(self.frequency))
            self.noise = np.mean(self.random_pow[mask])


    def estimate_initial_red(self, a=[]):
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
        boxkernel = Box1DKernel(int(np.ceil(self.fitbg['box_filter']/self.resolution)))
        mask = (self.frequency >= self.params[self.name]['ps_mask'][0])&(self.frequency <= self.params[self.name]['ps_mask'][1])
        self.smooth_pow = convolve(self.random_pow, boxkernel)
        # Temporary array for inputs into model optimization
        pars = np.zeros((self.nlaws*2+1))
        # Estimate amplitude for each harvey component
        for n, nu in enumerate(self.mnu):
            diff = list(np.absolute(self.frequency-nu))
            idx = diff.index(min(diff))
            if idx < self.fitbg['n_rms']:
                pars[2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][:self.fitbg['n_rms']]))/(4.*self.b[n]))
            elif (len(self.smooth_pow[~mask])-idx) < self.fitbg['n_rms']:
                pars[2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][-self.fitbg['n_rms']:]))/(4.*self.b[n]))
            else:
                pars[2*n+1] = np.sqrt((np.mean(self.smooth_pow[~mask][idx-int(self.fitbg['n_rms']/2):idx+int(self.fitbg['n_rms']/2)]))/(4.*self.b[n]))
            pars[2*n] = self.b[n]
            a.append(pars[2*n+1])
        pars[-1] = self.noise
        self.pars = pars
        self.a_orig = np.array(a)


    def get_best_model(self):
        """
        Determines the best-fit model for the stellar granulation background in the power spectrum
        by iterating through several models, where the initial guess for the number of Harvey-like 
        component(s) to model is estimated from a solar scaling relation.

        Parameters
        ----------
        names : List[str]
            the number of Harvey components to use in the background model
        bounds : list
            the bounds on the Harvey parameters for a given model
        bic : list
            the BIC statistic
        aic : list
            the AIC statistic
        paras : list
            the Harvey model parameters

        Returns
        -------
        again : bool
            will return `True` if fitting failed and the iteration must be repeated, otherwise `False`.

        """
        # Get best-fit model
        if self.fitbg['n_laws'] is None:
            self.bounds = []
            self.bic = []
            self.aic = []
            self.paras = []
            print('Comparing %d different models:'%(self.nlaws+1))
            for nlaws in range(self.nlaws+1):
                note=''
                bb = np.zeros((2,2*nlaws+1)).tolist()
                if nlaws != 0:
                    for z in range(nlaws):
                        bb[0][int(2*z)] = 0.
                        bb[1][int(2*z)] = self.baseline/10**6.
                        bb[0][int(2*z+1)] = 0.
                        bb[1][int(2*z+1)] = np.inf
                bb[0][-1] = 0.
                bb[1][-1] = np.inf
                self.bounds.append(tuple(bb))
                if self.verbose:
                    if nlaws == 0:
                        note += 'Model %d: Flat white-noise term only'%nlaws
                    else:
                        note += 'Model %d: White noise term + %d Harvey-like component(s)'%(nlaws,nlaws)
                try:
                    pp, _ = curve_fit(self.fitbg['functions'][nlaws], self.bin_freq, self.bin_pow, p0=self.pars[-int(2*nlaws+1):], sigma=self.bin_err, bounds=tuple(bb))
                except RuntimeError as _:
                    self.paras.append([])
                    self.bic.append(np.inf)
                    self.aic.append(np.inf)
                else:
                    self.paras.append(pp)
                    mask = np.ma.getmask(np.ma.masked_outside(self.frequency, self.params[self.name]['ps_mask'][0], self.params[self.name]['ps_mask'][1]))
                    observations=self.random_pow[mask]
                    model = models.harvey(self.frequency[mask], pp, total=True)
                    b = functions.compute_bic(observations, model, n_parameters=len(pp))
                    self.bic.append(b)
                    a = functions.compute_aic(observations, model, n_parameters=len(pp))
                    self.aic.append(a)
#                note += '\n BIC = %.2f | AIC = %.2f'%(b, a)
                    if self.verbose:
                        print(note)
            # If the fitting converged (fix to bic? depending on performance)
            if self.fitbg['metric'] == 'bic':
                if np.isfinite(min(self.bic)):
                    self.save_best_model()
                    return False
                else:
                    return True
            else:
                if np.isfinite(min(self.aic)):
                    self.save_best_model()
                    return False
                else:
                    return True
        else:
            note=''
            bb = np.zeros((2,2*self.fitbg['n_laws']+1)).tolist()
            if self.fitbg['n_laws'] != 0:
                for z in range(self.fitbg['n_laws']):
                    bb[0][int(2*z)] = 0.
                    bb[1][int(2*z)] = self.baseline/10**6.
                    bb[0][int(2*z+1)] = 0.
                    bb[1][int(2*z+1)] = np.inf
            bb[0][-1] = 0.
            bb[1][-1] = np.inf
            self.bounds = tuple(bb)
            if self.verbose:
                if nlaws == 0:
                    note += 'Using flat white-noise term only'%self.fitbg['n_laws']
                else:
                    note += 'Using %d Harvey-like component(s) + flat white-noise term'%self.fitbg['n_laws']
            try:
                pp, _ = curve_fit(self.fitbg['functions'][self.fitbg['n_laws']], self.bin_freq, self.bin_pow, p0=self.pars[-int(2*self.fitbg['n_laws']+1):], sigma=self.bin_err, bounds=tuple(bb))
            except RuntimeError as _:
                self.pars = []
            else:
                self.pars = pp
            if self.verbose:
                print(note)
            # If the fitting converged (fix to bic? depending on performance)
            if self.pars != []:
                self.save_best_model()
                return False
            else:
                return True


    def save_best_model(self, use='bic'):
        """
        Saves information re: the selected best-fit model (for the stellar background).
        
        Parameters
        ----------
        use : str
            which metric to use for model selection, choices ~['bic','aic']. Default is `'bic'`.

        """
        if self.fitbg['n_laws'] is None:
            if self.fitbg['metric'] == 'bic':
                model = self.bic.index(min(self.bic))
            else:
                model = self.aic.index(min(self.aic))
            # Store model results for plotting
            if self.nlaws != model:
                self.nlaws = model
                self.b = self.b[:(self.nlaws)]
            if self.verbose:
                print('Based on %s statistic: model %d'%(use.upper(),model))
            self.bounds = self.bounds[model]
            self.pars = self.paras[model]
        else:
            if self.nlaws != self.fitbg['n_laws']:
                self.nlaws = self.fitbg['n_laws']
                self.b = self.b[:(self.nlaws)]
        self.bg_corr = self.random_pow/models.harvey(self.frequency, self.pars, total=True)
        # Save background-corrected power spectrum
        utils.save_file(self)
        # Create appropriate keys for star based on nlaws
        for n in range(self.nlaws):
            self.fitbg['results'][self.name]['tau_%d'%(n+1)] = []
            self.fitbg['results'][self.name]['sigma_%d'%(n+1)] = []
        # Save the final values
        for n in range(self.nlaws):
            self.fitbg['results'][self.name]['tau_%d'%(n+1)].append(self.pars[2*n]*10**6.)
            self.fitbg['results'][self.name]['sigma_%d'%(n+1)].append(self.pars[2*n+1])
        self.fitbg['results'][self.name]['white'].append(self.pars[2*self.nlaws])


    def get_numax(self):
        """
        Simple function to call both numax methods.

        Returns
        -------
        None

        """
        self.get_numax_smooth()
        self.get_numax_gaussian()


    def get_numax_smooth(self, divide=True):
        """
        Estimate numax by smoothing the power spectrum and taking the peak. Also
        computes the background-corrected power spectrum and saves to a text file.

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
        self.pssm_bgcorr = self.pssm/models.harvey(self.frequency, self.pars, total=True)
        mask = np.ma.getmask(np.ma.masked_inside(self.frequency, self.params[self.name]['ps_mask'][0], self.params[self.name]['ps_mask'][1]))
        self.region_freq = self.frequency[mask]
        self.region_pow = self.pssm_bgcorr[mask]
        idx = functions.return_max(self.region_freq, self.region_pow, index=True)
        self.fitbg['results'][self.name]['numax_smooth'].append(self.region_freq[idx])
        self.fitbg['results'][self.name]['A_smooth'].append(self.region_pow[idx])
        # Initial guesses for the parameters of the Gaussian fit to the power envelope
        self.guesses = [0.0, max(self.region_pow), self.region_freq[idx], (max(self.region_freq)-min(self.region_freq))/np.sqrt(8.0*np.log(2.0))]


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
        constants = utils.Constants()
        bb = functions.gaussian_bounds(self.region_freq, self.region_pow, self.guesses)
        p_gauss1, _ = curve_fit(models.gaussian, self.region_freq, self.region_pow, p0=self.guesses, bounds=bb[0], maxfev=maxfev)
        # create array with finer resolution for purposes of quantifying uncertainty
        self.new_freq = np.linspace(min(self.region_freq), max(self.region_freq), 10000)
        self.numax_fit = list(models.gaussian(self.new_freq, *p_gauss1))
        d = self.numax_fit.index(max(self.numax_fit))
        self.fitbg['results'][self.name]['numax_gauss'].append(self.new_freq[d])
        self.fitbg['results'][self.name]['A_gauss'].append(p_gauss1[1])
        self.fitbg['results'][self.name]['FWHM'].append(p_gauss1[3])


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
        if self.globe['smooth_ps'] is not None:
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


    def get_acf_cutout(self, threshold=1.0):
        """
        Estimate the large frequency spacing or dnu.
        NOTE: this is only used during the first iteration!

        Parameters
        ----------
        threshold : float
            the threshold is multiplied by the full-width half-maximum value, centered on the peak 
            in the ACF to determine the width of the cutout region.
        
        Returns
        -------
        None

        """
        # Get peaks from ACF
        peak_idx,_ = find_peaks(self.auto)
        peaks_l,peaks_a = self.lag[peak_idx],self.auto[peak_idx]
        
        # Pick n highest peaks
        peaks_l = peaks_l[peaks_a.argsort()[::-1]][:self.globe['n_peaks']]
        peaks_a = peaks_a[peaks_a.argsort()[::-1]][:self.globe['n_peaks']]
        # Pick best peak in ACF by using Gaussian weight according to expected dnu
        idx = functions.return_max(peaks_l, peaks_a, index=True, exp_dnu=self.params[self.name]['dnu'])
        self.best_lag = peaks_l[idx]
        self.best_auto = peaks_a[idx]
        # Change fitted value with nan to highlight differently in plot
        peaks_l[idx] = np.nan
        peaks_a[idx] = np.nan
        self.peaks_l = peaks_l
        self.peaks_a = peaks_a
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
        self.params[self.name]['acf_mask']=[self.best_lag-(self.globe['threshold']*((right_lag-left_lag)/2.)),self.best_lag+(self.globe['threshold']*((right_lag-left_lag)/2.))]
        self.zoom_lag = self.lag[(self.lag>=self.params[self.name]['acf_mask'][0])&(self.lag<=self.params[self.name]['acf_mask'][1])]
        self.zoom_auto = self.auto[(self.lag>=self.params[self.name]['acf_mask'][0])&(self.lag<=self.params[self.name]['acf_mask'][1])]
        # Boundary conditions and initial guesses stay the same for all iterations
        self.acf_guesses = [np.mean(self.zoom_auto), self.best_auto, self.best_lag, self.best_lag*0.01*2.]
        self.acf_bb = functions.gaussian_bounds(self.zoom_lag, self.zoom_auto, self.acf_guesses, best_x=self.best_lag, sigma=10**-2)
        # Fit a Gaussian function to the selected peak in the ACF to get dnu
        p_gauss3, _ = curve_fit(models.gaussian, self.zoom_lag, self.zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb[0])
       	# If dnu is provided, use that instead
        if self.params[self.name]['force']:
            p_gauss3[2] = self.params[self.name]['guess']
        self.fitbg['results'][self.name]['dnu'].append(p_gauss3[2])
        # Save for plotting
        self.new_lag = np.linspace(min(self.zoom_lag),max(self.zoom_lag),2000)
        self.dnu_fit = models.gaussian(self.new_lag,*p_gauss3)
        self.obs_acf = max(self.dnu_fit)


    def get_dnu(self):
        """
        Estimate a value for dnu.

        Returns
        -------
        None

        """
        self.compute_acf()
        if self.i == 0:
            self.get_acf_cutout()
            self.get_ridges()
        else:
            # define the peak in the ACF
            zoom_lag = self.lag[(self.lag>=self.params[self.name]['acf_mask'][0])&(self.lag<=self.params[self.name]['acf_mask'][1])]
            zoom_auto = self.auto[(self.lag>=self.params[self.name]['acf_mask'][0])&(self.lag<=self.params[self.name]['acf_mask'][1])]
            # fit a Gaussian function to the selected peak in the ACF
            p_gauss3, _ = curve_fit(models.gaussian, zoom_lag, zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb[0])
            # the center of that Gaussian is our estimate for Dnu
            self.fitbg['results'][self.name]['dnu'].append(p_gauss3[2]) 


    def get_ridges(self, start=0.0):
        """
        Create echelle diagram.

        Parameters
        ----------
        start : float
            lower limit of distance modulus. Default value is `0.0`.

        Returns
        -------
        None

        """
        ech, gridx, gridy, extent = self.echelle()
        N, M = ech.shape[0], ech.shape[1]
        ech_copy = np.array(list(ech.reshape(-1)))

        n = int(np.ceil(self.fitbg['results'][self.name]['dnu'][0]/self.resolution))
        xax = np.zeros(n)
        yax = np.zeros(n)
        modx = self.frequency%self.fitbg['results'][self.name]['dnu'][0]
        for k in range(n):
            use = np.where((modx >= start)&(modx < start+self.resolution))[0]
            if len(use) == 0:
                continue
            xax[k] = np.median(modx[use])
            yax[k] = np.sum(self.bg_corr[use])
            start += self.resolution
        xax = np.array(list(xax)+list(xax+self.fitbg['results'][self.name]['dnu'][0]))
        yax = np.array(list(yax)+list(yax))-min(yax)
        mask = np.ma.getmask(np.ma.masked_where(yax == 0.0, yax))
        # Clip the lower bound (`clip_value`)
        if self.globe['clip_ech']:
            if self.globe['clip_value'] is not None:
                cut = self.globe['clip_value']
            else:
                cut = np.nanmedian(ech_copy)+(3.0*np.nanmedian(ech_copy))
            ech_copy[ech_copy > cut] = cut
        self.ech_copy = ech_copy
        self.ech = ech_copy.reshape((N, M))
        self.extent = extent
        self.xax = xax[~mask]
        self.yax = yax[~mask]


    def echelle(self, n_across=50, startx=0.0):
        """
        Creates an echelle diagram.

        Parameters
        ----------
        n_across : int
            number of grid points in x-axis of echelle diagram. Default value is `50`.
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
        nox = self.globe['n_across']
        # fix ndown (no obvious way to do this yet)
        noy = int(np.ceil((max(self.frequency)-min(self.frequency))/self.fitbg['results'][self.name]['dnu'][0]))
        if nox > 2 and noy > 5:
            xax = np.arange(0.0, self.fitbg['results'][self.name]['dnu'][0]+(self.fitbg['results'][self.name]['dnu'][0]/nox)/2.0, self.fitbg['results'][self.name]['dnu'][0]/nox)
            yax = np.arange(min(self.frequency), max(self.frequency), self.fitbg['results'][self.name]['dnu'][0])
            arr = np.zeros((len(xax), len(yax)))
            gridx = np.zeros(len(xax))
            gridy = np.zeros(len(yax))
            modx = self.frequency%(self.fitbg['results'][self.name]['dnu'][0])
            starty = min(self.frequency)
            for ii in range(len(gridx)):
                for jj in range(len(gridy)):
                    if self.bg_corr[((modx >= startx)&(modx < startx+(self.fitbg['results'][self.name]['dnu'][0]/nox)))&((self.frequency >= starty)&(self.frequency < starty+self.fitbg['results'][self.name]['dnu'][0]))] != []:
                        arr[ii, jj] = np.sum(self.bg_corr[((modx >= startx)&(modx < startx+(self.fitbg['results'][self.name]['dnu'][0]/n_across)))&((self.frequency >= starty)&(self.frequency < starty+self.fitbg['results'][self.name]['dnu'][0]))])
                    else:
                        arr[ii, jj] = np.nan
                    gridy[jj] = starty + (self.fitbg['results'][self.name]['dnu'][0]/2.0)
                    starty += self.fitbg['results'][self.name]['dnu'][0]
                gridx[ii] = startx + (self.fitbg['results'][self.name]['dnu'][0]/self.globe['n_across']/2.0)
                starty = min(self.frequency)
                startx += self.fitbg['results'][self.name]['dnu'][0]/self.globe['n_across']
            smoothed = arr
            dim = smoothed.shape

            smoothed_2 = np.zeros((2*dim[0], dim[1]))
            smoothed_2[0:dim[0],:] = smoothed
            smoothed_2[dim[0]:(2*dim[0]),:] = smoothed
            smoothed = np.swapaxes(smoothed_2, 0, 1)
            extent = [min(gridx)-(self.fitbg['results'][self.name]['dnu'][0]/n_across/2.0), 2*max(gridx)+(self.fitbg['results'][self.name]['dnu'][0]/n_across/2.0), min(gridy)-(self.fitbg['results'][self.name]['dnu'][0]/2.0), max(gridy)+(self.fitbg['results'][self.name]['dnu'][0]/2.0)]
            gridx = np.array(list(gridx)+list(gridx+self.fitbg['results'][self.name]['dnu'][0]))
            return smoothed, gridx, gridy, extent


    def get_red_noise(self):
        """
        Calculates red noise level, or stellar background contribution, from power spectrum.

        Returns
        -------
        result : bool
            will return `False` if model converges, otherwise `True`.
        
        """
        constants = utils.Constants()
        # Use as initial guesses for the optimized model
        try:
            pars, _ = curve_fit(self.fitbg['functions'][self.nlaws], self.bin_freq, self.bin_pow, p0=self.pars, sigma=self.bin_err, bounds=self.bounds)
        except RuntimeError as _:
            return True
        else:
            self.pars = pars
            self.bg_corr = self.random_pow/models.harvey(self.frequency, self.pars, total=True)
            # save final values for Harvey components
            for n in range(self.nlaws):
                self.fitbg['results'][self.name]['tau_%d'%(n+1)].append(self.pars[2*n]*10**6.)
                self.fitbg['results'][self.name]['sigma_%d'%(n+1)].append(self.pars[2*n+1])
            self.fitbg['results'][self.name]['white'].append(self.pars[2*self.nlaws])
        return False


    def estimate_dnu(self):
        """
        Estimate a value for $\delta \nu$.

        Returns
        -------
        None
        
        """
        # define the peak in the ACF
        zoom_lag = self.lag[(self.lag>=self.fitbg['acf_mask'][self.name][0])&(self.lag<=self.fitbg['acf_mask'][self.name][1])]
        zoom_auto = self.auto[(self.lag>=self.fitbg['acf_mask'][self.name][0])&(self.lag<=self.fitbg['acf_mask'][self.name][1])]

		      # fit a Gaussian function to the selected peak in the ACF
        p_gauss3, _ = curve_fit(models.gaussian, zoom_lag, zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb[0])
        # the center of that Gaussian is our estimate for Dnu
        dnu = p_gauss3[2]
        self.fitbg['results'][self.name]['dnu'].append(dnu)