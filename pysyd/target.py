import numpy as np
from tqdm import tqdm
from scipy.stats import chisquare
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
    A pipeline target. Initialization will cause the pipeline to process the target.

    Attributes
    ----------
    star : int
        the star ID
    params : dict
        the pipeline parameters
    findex : dict
        the parameters of the find excess routine
    fitbg : dict
        the parameters of the fit background routine
    verbose : bool
        if true, verbosity will increase
    show_plots : bool
        if true, plots will be shown
    keplercorr : bool
        if true will correct Kepler artefacts in the power spectrum
    filter : float
        the box filter width [muHz] for the power spectrum

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
        self.verbose = args.verbose
        self.info = args.info
        self.check_data()


    def check_data(self):
        """
        Load star data.

        """
        # Make sure data load was successsful
        lc_data, ps_data, self = utils.load_data(self)
        if lc_data and ps_data:
            self.run = 1
        else:
            self.run = 0
            if self.verbose:
                print('ERROR: data not found for star %d' % self.name)


    def run_syd(self):
        """
        Run the pipeline routines.

        """
        # Run the find excess routine
        if self.params[self.name]['excess']:
            self.find_excess()
        # Run the fit background routine
        if self.params[self.name]['background']:
            self.fit_background()


    def find_excess(self):
        """
        Automatically finds power excess due to solar-like oscillations using a
        frequency resolved collapsed autocorrelation function.

        """

        # Make sure the binning is specified, otherwise it cannot run
        if self.findex['binning'] is not None:
            self.bin_freq, self.bin_pow = functions.bin_data(self.freq, self.pow, self.findex)
            if self.verbose:
                print('-------------------------------------------------')
                print('Running find_excess module:')
                print('PS binned to %d datapoints' % len(self.bin_freq))
            # Smooth the binned power spectrum for a rough estimate of background
            boxsize = int(np.ceil(float(self.findex['smooth_width'])/(self.bin_freq[1]-self.bin_freq[0])))
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
                print('selecting model %d' % self.findex['results'][self.name]['best'])
            utils.save_findex(self)
            plots.plot_excess(self)


    def fit_background(self):
        """
        Perform a fit to the granulation background and measures the frequency of maximum power (numax),
        the large frequency separation (dnu) and oscillation amplitude.

        """

        # Will only run routine if there is a prior numax estimate
        if utils.check_fitbg(self):
            # Check for guesses or find_excess results
            self = utils.get_initial_guesses(self)
            self.fitbg['results'][self.name] = {
                'numax_smooth': [],
                'amp_smooth': [],
                'numax_gaussian': [],
                'amp_gaussian': [],
                'fwhm_gaussian': [],
                'dnu': [],
                'wn': []
            }
            if self.verbose:
                print('-------------------------------------------------')
                print('Running fit_background module:')
                print('PS binned to %d data points' % len(self.bin_freq))

            # Run first iteration (which has different steps than any other n>1 runs)
            good = self.single_step()
            if not good:
                pass
            else:
                # If sampling is enabled (i.e., args.mciter > 1), a progress bar is created w/ verbose output
                if self.fitbg['mc_iter'] > 1:
                    if self.verbose:
                        print('-------------------------------------------------')
                        print('Running sampling routine:')
                        self.pbar = tqdm(total=self.fitbg['mc_iter'])
                        self.pbar.update(1)
                    self.i = 1
                    # Continue to sample while the number of successful steps is less than args.mciter
                    while self.i < self.fitbg['mc_iter']:
                        self.sampling_step()
                    utils.save_fitbg(self)
                    plots.plot_samples(self)
                    if self.verbose:
                        # Print results with uncertainties
                        utils.verbose_output(self, sampling=True)
                # Single iteration
                else:
                    utils.save_fitbg(self)
                    if self.verbose:
                        # Print results without uncertainties
                        utils.verbose_output(self)


    def collapsed_acf(self, b, j=0, start=0, max_iterations=5000, max_snr=100.):
        """
        TODO

        """
        # Computes a collapsed ACF using different "box" (or bin) sizes
        self.findex['results'][self.name][b+1] = {}
        subset = np.ceil(self.boxes[b]/self.resolution)
        steps = np.ceil((self.boxes[b]*self.findex['step'])/self.resolution)

        cumsum = np.zeros_like(self.freq)
        md = np.zeros_like(self.freq)
        # Iterates through entire power spectrum using box width
        while True:
            if (start+subset) > len(self.freq):
                break
            f = self.freq[int(start):int(start+subset)]
            p = self.bgcorr_pow[int(start):int(start+subset)]

            lag = np.arange(0.0, len(p))*self.resolution
            auto = np.real(np.fft.fft(np.fft.ifft(p)*np.conj(np.fft.ifft(p))))
            corr = np.absolute(auto-np.mean(auto))

            cumsum[j] = np.sum(corr)
            md[j] = np.mean(f)

            start += steps
            j += 1

        # Only keep non-zero elements in CDF and then normalize
        md = md[~np.ma.getmask(np.ma.masked_values(cumsum, 0.0))]
        cumsum = cumsum[~np.ma.getmask(np.ma.masked_values(cumsum, 0.0))] - min(cumsum[~np.ma.getmask(np.ma.masked_values(cumsum, 0.0))])
        cumsum = list(cumsum/max(cumsum))
        # Pick the maximum value from the CDF as an initial guess for numax
        idx = cumsum.index(max(cumsum))
        self.findex['results'][self.name][b+1].update({'x':md,'y':np.array(cumsum),'maxx':md[idx],'maxy':cumsum[idx]})

        # Fit Gaussian to get estimate value for numax
        try:
            best_vars, _ = curve_fit(models.gaussian, md, cumsum, 
                 p0=[np.mean(cumsum), 1.0-np.mean(cumsum), md[idx], self.params['width_sun']*(md[idx]/self.params['numax_sun'])],
                 maxfev=max_iterations,
                 bounds=((-np.inf,-np.inf,1,-np.inf),(np.inf,np.inf,np.inf,np.inf)),
                 )
        except Exception as _:
            self.findex['results'][self.name][b+1].update({'good_fit':False})
            snr = 0.
        else:
            self.findex['results'][self.name][b+1].update({'good_fit':True})
            fitx = np.linspace(min(md), max(md), 10000)
            fity = models.gaussian(fitx, *best_vars)
            self.findex['results'][self.name][b+1].update({'fitx':fitx,'fity':fity})
            snr = max(fity)/best_vars[0]
            if snr > max_snr:
                snr = max_snr
            self.findex['results'][self.name][b+1].update({'numax':best_vars[2],'dnu':functions.delta_nu(best_vars[2]),'snr':snr})
            if self.verbose:
                  print('power excess trial %d: numax = %.2f +/- %.2f' % (b+1, best_vars[2], np.absolute(best_vars[3])/2.0))
                  print('S/N: %.2f' % snr)
        self.compare.append(snr)


    def single_step(self):
        """
        The first step in the background fitting, which determines the best-fit stellar 
        contribution model (i.e. number of Harvey-like components) and corrects for this

        TODO: implement more robust criterion (i.e. BIC or AIC) and also include the simplest model.
        before estimating numax and dnu.

        single_step: The main iteration of the background fitting, which operates in the following steps:
                     1) determines the best-fit model (i.e. number of Harvey components) using a reduced chi-sq analysis
                     2) corrects for stellar background contributions by dividing the power spectrum by the best-fit model
                     3) estimates two values for numax by fitting a Gaussian and by using a heavy smoothing filter
                     4) takes the autocorrelation (using ffts) of the masked power spectrum that contains the power excess
                     5) selects the peak (via -npeaks, default=10) closest to the expected spacing based on the calculated numax
                     6) fits Gaussian to the "cutout" peak of the ACF, where center is dnu

        """
        # Save a copy of original power spectrum
        self.random_pow = np.copy(self.power)
        # Estimate white noise level
        self.get_white_noise()
        # Get initial guesses for the optimization of the background model
        self.estimate_initial_red()
        # If optimization does not converge, the rest of the code will not run
        if self.get_best_model():
            print('WARNING: Bad initial fit for star %d. Check this and try again.'%self.name)
            return False
        # Estimate numax using two different methods
        self.get_numax_smooth()
        if list(self.region_freq) != []:
            self.exp_numax, self.exp_dnu, self.width, self.new_freq, self.numax_fit = self.get_numax_gaussian(output=True)
        # Estimate the large frequency spacing w/ special function
        self.get_acf_cutout()
        # Use the fitted dnu to create an echelle diagram and plot
        self.get_ridges()
        plots.plot_background(self)
        return True


    def sampling_step(self):
        """
        Used in the background fitting routine to quantify the estimated parameter 
        uncertainties. This is executed through a procedure analogous to the bootstrapping
        method, which will randomize the power spectrum based on a chi-squared distribution
        and attempt to recover the derived properties from the first step. This is invoked 
        when the args.mciter > 1, where args.mciter = 200 is typically sufficient for 
        estimating an uncertainty.

        """
        # Randomize power spectrum to get uncertainty on measured values
        self.random_pow = (np.random.chisquare(2, len(self.frequency))*self.power)/2.
        # Bin randomized power spectra
        bin_freq, bin_pow, bin_err = functions.mean_smooth_ind(self.frequency, self.random_pow, self.fitbg['ind_width'])
        self.bin_freq = bin_freq[~((bin_freq > self.maxpower[0]) & (bin_freq < self.maxpower[1]))]
        self.bin_pow = bin_pow[~((bin_freq > self.maxpower[0]) & (bin_freq < self.maxpower[1]))]
        self.bin_err = bin_err[~((bin_freq > self.maxpower[0]) & (bin_freq < self.maxpower[1]))]

        # Estimate simulated red noise amplitudes
        self.estimate_initial_red()
        # Estimate stellar background contribution
        if self.get_red_noise():
            return
        # If the model converges, continue estimating global parameters 
        # Get numaxes
        self.get_numax_smooth()
        if list(self.region_freq) != []:
            self.get_numax_gaussian()
        # Estimate dnu
        self.get_frequency_spacing()
        self.i += 1
        if self.verbose:
            self.pbar.update(1)
            if self.i == self.fitbg['mc_iter']:
                self.pbar.close()


    def get_white_noise(self):
        """
        Estimate the white noise level (in muHz) by taking a mean over a region 
        in the power spectrum near the nyquist frequency.

        """
        if self.nyquist < 400.0:
            mask = (self.frequency > 200.0) & (self.frequency < 270.0)
            self.noise = np.mean(self.random_pow[mask])
        elif self.nyquist > 400.0 and self.nyquist < 5000.0:
            mask = (self.frequency > 4000.0) & (self.frequency < 4167.)
            self.noise = np.mean(self.random_pow[mask])
        elif self.nyquist > 5000.0 and self.nyquist < 9000.0:
            mask = (self.frequency > 8000.0) & (self.frequency < 8200.0)
            self.noise = np.mean(self.random_pow[mask])
        else:
            mask = (self.frequency > (max(self.frequency) - 0.1*max(self.frequency))) & (self.frequency < max(self.frequency))
            self.noise = np.mean(self.random_pow[mask])


    def estimate_initial_red(self):
        """
        Estimates amplitude of red noise components by using a smoothed version of the power
        spectrum with the power excess region masked out. This will take the mean of a specified 
        number of points (via -nrms, default=20) for each Harvey-like component.

        """
        # Exclude region with power excess and smooth to estimate red noise components
        boxkernel = Box1DKernel(int(np.ceil(self.fitbg['box_filter']/self.resolution)))
        self.params[self.name]['ps_mask'] = (self.frequency >= self.maxpower[0]) & (self.frequency <= self.maxpower[1])
        self.smooth_pow = convolve(self.random_pow[~self.params[self.name]['ps_mask']], boxkernel)
        # Temporary array for inputs into model optimization
        pars = np.zeros((self.nlaws*2 + 1))
        # Estimate amplitude for each harvey component
        for n, nu in enumerate(self.mnu):
            diff = list(np.absolute(self.frequency - nu))
            idx = diff.index(min(diff))
            if idx < self.fitbg['n_rms']:
                pars[2*n] = np.mean(self.smooth_pow[:self.fitbg['n_rms']])
            elif (len(self.smooth_pow)-idx) < self.fitbg['n_rms']:
                pars[2*n] = np.mean(self.smooth_pow[-self.fitbg['n_rms']:])
            else:
                pars[2*n] = np.mean(self.smooth_pow[idx-int(self.fitbg['n_rms']/2):idx+int(self.fitbg['n_rms']/2)])
            pars[2*n+1] = self.b[n]
        pars[-1] = self.noise
        self.pars = pars


    def get_best_model(self):
        """
        Determines the best-fit model for the stellar granulation background in the power spectrum
        by iterating through several models, where the initial guess for the number of Harvey-like 
        component(s) to model is estimated from a solar scaling relation.

        Parameters
        ----------
        names : list
            the number of Harvey components to use in the background model
        bounds : list
            the bounds on the Harvey parameters for a given model
        reduced_chi2 : list
            the reduced chi-squared statistic
        paras : list
            the Harvey model parameters
        a : list
            the amplitude of the individual Harvey components

        Returns
        -------
        again : bool
            will return `True` if fitting failed and the iteration must be repeated otherwise `False`.
        """

        # Get best fit model
        names=['one', 'one', 'two', 'two', 'three', 'three', 'four', 'four', 'five', 'five', 'six', 'six']
        reduced_chi2 = []
        bounds = []
        a = []
        paras = []
        for n in range(self.nlaws):
            a.append(self.pars[2*n])
        self.a_orig = np.array(a)
        if self.verbose:
            print('Comparing %d different models:' % (self.nlaws*2))
        for law in range(self.nlaws):
            bb = np.zeros((2, 2*(law+1)+1)).tolist()
            for z in range(2*(law+1)):
                bb[0][z] = -np.inf
                bb[1][z] = np.inf
            bb[0][-1] = 0.
            bb[1][-1] = np.inf
            bounds.append(tuple(bb))
        dict1 = dict(zip(np.arange(2*self.nlaws), names[:2*self.nlaws]))
        for t in range(2*self.nlaws):
            if t % 2 == 0:
                if self.verbose:
                    print('%d: %s harvey model w/ white noise free parameter' % (t+1, dict1[t]))
                delta = 2*(self.nlaws-(t//2+1))
                pams = list(self.pars[:(-delta-1)])
                pams.append(self.pars[-1])
                try:
                    pp, _ = curve_fit(
                        self.fitbg['functions'][t//2+1],
                        self.bin_freq,
                        self.bin_pow,
                        p0=pams,
                        sigma=self.bin_err
                    )
                except RuntimeError as _:
                    paras.append([])
                    reduced_chi2.append(np.inf)
                else:
                    paras.append(pp)
                    chi, _ = chisquare(
                        f_obs=self.random_pow[~self.params[self.name]['ps_mask']],
                        f_exp=models.harvey(
                            self.frequency[~self.params[self.name]['ps_mask']],
                            pp,
                            total=True
                        )
                    )
                    reduced_chi2.append(chi/(len(self.frequency[~self.params[self.name]['ps_mask']])-len(pams)))
            else:
                if self.verbose:
                    print('%d: %s harvey model w/ white noise fixed' % (t+1, dict1[t]))
                delta = 2*(self.nlaws-(t//2+1))
                pams = list(self.pars[:(-delta-1)])
                pams.append(self.pars[-1])
                try:
                    pp, _ = curve_fit(
                        self.fitbg['functions'][t//2+1],
                        self.bin_freq,
                        self.bin_pow,
                        p0=pams,
                        sigma=self.bin_err,
                        bounds=bounds[t//2]
                    )
                except RuntimeError as _:
                    paras.append([])
                    reduced_chi2.append(np.inf)
                else:
                    paras.append(pp)
                    chi, p = chisquare(
                        f_obs=self.random_pow[~self.params[self.name]['ps_mask']],
                        f_exp=models.harvey(
                            self.frequency[~self.params[self.name]['ps_mask']],
                            pp,
                            total=True
                            )
                        )
                    reduced_chi2.append(chi/(len(self.frequency[~self.params[self.name]['ps_mask']])-len(pams)+1))

        # If the fitting converged
        if np.isfinite(min(reduced_chi2)):
            model = reduced_chi2.index(min(reduced_chi2)) + 1
            if self.nlaws != (((model-1)//2)+1):
                self.nlaws = ((model-1)//2)+1
                self.mnu = self.mnu[:(self.nlaws)]
                self.b = self.b[:(self.nlaws)]
            if self.verbose:
                print('Based on reduced chi-squared statistic: model %d' % model)
            self.bounds = bounds[self.nlaws-1]
            self.pars = paras[model-1]
            self.bg_corr = self.random_pow/models.harvey(self.frequency, self.pars, total=True)
            self.exp_numax = self.params[self.name]['numax']
            self.exp_dnu = self.params[self.name]['dnu']
            self.sm_par = 4.*(self.exp_numax/self.params['numax_sun'])**0.2
            if self.sm_par < 1.:
                self.sm_par = 1.
            for n in range(self.nlaws):
                self.fitbg['results'][self.name]['a_%d' % (n+1)] = []
                self.fitbg['results'][self.name]['b_%d' % (n+1)] = []
            # save final values for Harvey laws from model fit
            for n in range(self.nlaws):
                self.fitbg['results'][self.name]['a_%d' % (n+1)].append(self.pars[2*n])
                self.fitbg['results'][self.name]['b_%d' % (n+1)].append(self.pars[2*n+1])
            self.fitbg['results'][self.name]['wn'].append(self.pars[2*self.nlaws])
            return False
        else:
            return True


    def get_numax_smooth(self):
        """
        Estimate numax by smoothing the power spectrum and taking the peak.

        """

        sig = (self.sm_par*(self.exp_dnu/self.resolution))/np.sqrt(8.0*np.log(2.0))
        pssm = convolve_fft(np.copy(self.random_pow), Gaussian1DKernel(int(sig)))
        model = models.harvey(self.frequency, self.pars, total=True)
        inner_freq = list(self.frequency[self.params[self.name]['ps_mask']])
        inner_obs = list(pssm[self.params[self.name]['ps_mask']])
        outer_freq = list(self.frequency[~self.params[self.name]['ps_mask']])
        outer_mod = list(model[~self.params[self.name]['ps_mask']])
        if self.fitbg['slope']:
            # Correct for edge effects and residual slope in Gaussian fit
            inner_mod = model[self.params[self.name]['mask']]
            delta_y = inner_obs[-1]-inner_obs[0]
            delta_x = inner_freq[-1]-inner_freq[0]
            slope = delta_y/delta_x
            b = slope*(-1.0*inner_freq[0]) + inner_obs[0]
            corrected = np.array([inner_freq[z]*slope + b for z in range(len(inner_freq))])
            corr_pssm = [inner_obs[z] - corrected[z] + inner_mod[z] for z in range(len(inner_obs))]
            final_y = np.array(corr_pssm + outer_mod)
        else:
            outer_freq = list(self.frequency[~self.params[self.name]['ps_mask']])
            outer_mod = list(model[~self.params[self.name]['ps_mask']])
            final_y = np.array(inner_obs + outer_mod)
        final_x = np.array(inner_freq + outer_freq)
        ss = np.argsort(final_x)
        final_x = final_x[ss]
        final_y = final_y[ss]
        self.pssm = np.copy(final_y)
        self.pssm_bgcorr = self.pssm-models.harvey(final_x, self.pars, total=True)
        self.region_freq = self.frequency[self.params[self.name]['ps_mask']]
        self.region_pow = self.pssm_bgcorr[self.params[self.name]['ps_mask']]
        idx = functions.return_max(self.region_freq, self.region_pow, index=True)
        self.fitbg['results'][self.name]['numax_smooth'].append(self.region_freq[idx])
        self.fitbg['results'][self.name]['amp_smooth'].append(self.region_pow[idx])
        # Initial guesses for the parameters of the Gaussian fit to the power envelope
        self.guesses = [
            0.0,
            max(self.region_pow),
            self.region_freq[idx],
            (max(self.region_freq) - min(self.region_freq))/np.sqrt(8.0*np.log(2.0))
        ]


    def get_numax_gaussian(self, output=False):
        """
        Estimate numax by fitting a Gaussian to the power envelope of the smoothed power spectrum.

        """

        bb = functions.gaussian_bounds(self.region_freq, self.region_pow, self.guesses)
        p_gauss1, _ = curve_fit(models.gaussian, self.region_freq, self.region_pow, p0=self.guesses, bounds=bb[0], maxfev=5000)
        # create array with finer resolution for purposes of quantifying uncertainty
        new_freq = np.linspace(min(self.region_freq), max(self.region_freq), 10000)
        numax_fit = list(models.gaussian(new_freq, *p_gauss1))
        d = numax_fit.index(max(numax_fit))
        self.fitbg['results'][self.name]['numax_gaussian'].append(new_freq[d])
        self.fitbg['results'][self.name]['amp_gaussian'].append(p_gauss1[1])
        self.fitbg['results'][self.name]['fwhm_gaussian'].append(p_gauss1[3])
        if output:
            return new_freq[d], 0.22*(self.exp_numax**0.797), self.params['width_sun']*(new_freq[d]/self.params['numax_sun'])/2., np.copy(new_freq), np.array(numax_fit)


    def compute_acf(self, fft=True):
        """Compute the ACF of the smooth background corrected power spectrum.

        Parameters
        ----------
        fft : bool
            if true will use FFT to compute the ACF
        """
        # Optional smoothing of PS to remove fine structure before computing ACF
        if self.fitbg['smooth_ps'] is not None:
            boxkernel = Box1DKernel(int(np.ceil(self.fitbg['smooth_ps']/self.resolution)))
            self.bg_corr_smooth = convolve(self.bg_corr, boxkernel)
        else:
            self.bg_corr_smooth = np.copy(self.bg_corr)

        # Use only power near the expected numax to reduce additional noise in ACF
        power = self.bg_corr_smooth[(self.frequency >= self.exp_numax-self.width) & (self.frequency <= self.exp_numax+self.width)]
        lag = np.arange(0.0, len(power))*self.resolution
        if fft:
            auto = np.real(np.fft.fft(np.fft.ifft(power)*np.conj(np.fft.ifft(power))))
        else:
            auto = np.correlate(power-np.mean(power), power-np.mean(power), "full")
            auto = auto[int(auto.size/2):]
        mask = np.ma.getmask(np.ma.masked_inside(lag, self.exp_dnu/4., 2.*self.exp_dnu+self.exp_dnu/4.))
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
        dnu : float
            the estimated value of dnu
        threshold : float
            the threshold is multiplied by the full-width half-maximum value, centered on the peak 
            in the ACF to determine the width of the cutout region
        """
        self.compute_acf()
        # Get peaks from ACF
        peak_idx,_ = find_peaks(self.auto) #indices of peaks, threshold=half max(ACF)
        peaks_l,peaks_a = self.lag[peak_idx],self.auto[peak_idx]
        
        # Pick n highest peaks
        peaks_l = peaks_l[peaks_a.argsort()[::-1]][:self.fitbg['n_peaks']]
        peaks_a = peaks_a[peaks_a.argsort()[::-1]][:self.fitbg['n_peaks']]
        
        # Pick best peak in ACF by using Gaussian weight according to expected dnu
        idx = functions.return_max(peaks_l, peaks_a, index=True, exp_dnu=self.exp_dnu)
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
        self.fitbg['acf_mask'][self.name]=[self.best_lag-(threshold*((right_lag-left_lag)/2.)),self.best_lag+(threshold*((right_lag-left_lag)/2.))]
        self.zoom_lag = self.lag[(self.lag>=self.fitbg['acf_mask'][self.name][0])&(self.lag<=self.fitbg['acf_mask'][self.name][1])]
        self.zoom_auto = self.auto[(self.lag>=self.fitbg['acf_mask'][self.name][0])&(self.lag<=self.fitbg['acf_mask'][self.name][1])]

        # Boundary conditions and initial guesses stay the same for all iterations
        self.acf_guesses = [np.mean(self.zoom_auto), self.best_auto, self.best_lag, self.best_lag*0.01*2.]
        self.acf_bb = functions.gaussian_bounds(self.zoom_lag, self.zoom_auto, self.acf_guesses, best_x=self.best_lag, sigma=10**-2)
        # Fit a Gaussian function to the selected peak in the ACF to get dnu
        p_gauss3, _ = curve_fit(models.gaussian, self.zoom_lag, self.zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb[0])
       	# If dnu is provided, use that instead
        if self.params[self.name]['force']:
            p_gauss3[2] = self.params[self.name]['guess']
        self.fitbg['results'][self.name]['dnu'].append(p_gauss3[2])
        self.obs_dnu = p_gauss3[2]
        # Save for plotting
        self.new_lag = np.linspace(min(self.zoom_lag),max(self.zoom_lag),2000)
        self.dnu_fit = models.gaussian(self.new_lag,*p_gauss3)
        self.obs_acf = max(self.dnu_fit)


    def get_frequency_spacing(self):
        """
        Estimate a value for dnu.

        """
        self.compute_acf()
        # define the peak in the ACF
        zoom_lag = self.lag[(self.lag>=self.fitbg['acf_mask'][self.name][0])&(self.lag<=self.fitbg['acf_mask'][self.name][1])]
        zoom_auto = self.auto[(self.lag>=self.fitbg['acf_mask'][self.name][0])&(self.lag<=self.fitbg['acf_mask'][self.name][1])]

        # fit a Gaussian function to the selected peak in the ACF
        p_gauss3, _ = curve_fit(models.gaussian, zoom_lag, zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb[0])
        # the center of that Gaussian is our estimate for Dnu
        dnu = p_gauss3[2]
        self.fitbg['results'][self.name]['dnu'].append(dnu) 


    def get_ridges(self, start=0.0):
        """
        Create echelle diagram.

        Parameters
        ----------
        start : float
            TODO: Write description. Default value is `0.0`.

        """

        ech, gridx, gridy, extent = self.echelle()
        N, M = ech.shape[0], ech.shape[1]
        ech_copy = np.array(list(ech.reshape(-1)))

        n = int(np.ceil(self.obs_dnu/self.resolution))
        xax = np.zeros(n)
        yax = np.zeros(n)
        modx = self.frequency % self.obs_dnu
        for k in range(n):
            use = np.where((modx >= start) & (modx < start+self.resolution))[0]
            if len(use) == 0:
                continue
            xax[k] = np.median(modx[use])
            yax[k] = np.sum(self.bg_corr[use])
            start += self.resolution
        xax = np.array(list(xax)+list(xax+self.obs_dnu))
        yax = np.array(list(yax)+list(yax))-min(yax)
        mask = np.ma.getmask(np.ma.masked_where(yax == 0.0, yax))
        # Clip the lower bound (`clip_value`)
        if self.fitbg['clip_ech']:
            if self.fitbg['clip_value'] is not None:
                cut = self.fitbg['clip_value']
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
            TODO: Write description. Default value is `50`.
        startx : float
            TODO: Write description. Default value is `0.0`.

        Returns
        -------
        TODO: Write return arguments.

        """

        if self.fitbg['smooth_ech'] is not None:
            boxkernel = Box1DKernel(int(np.ceil(self.fitbg['smooth_ech']/self.resolution)))
            smooth_y = convolve(self.bg_corr, boxkernel)
        nox = n_across
        noy = int(np.ceil((max(self.frequency)-min(self.frequency))/self.obs_dnu))
        if nox > 2 and noy > 5:
            xax = np.arange(0.0, self.obs_dnu+(self.obs_dnu/n_across)/2.0, self.obs_dnu/n_across)
            yax = np.arange(min(self.frequency), max(self.frequency), self.obs_dnu)
            arr = np.zeros((len(xax), len(yax)))
            gridx = np.zeros(len(xax))
            gridy = np.zeros(len(yax))

            modx = self.frequency % self.obs_dnu
            starty = min(self.frequency)
            for ii in range(len(gridx)):
                for jj in range(len(gridy)):
                    use = np.where((modx >= startx) & (modx < startx+self.obs_dnu/n_across) & (self.frequency >= starty) & (self.frequency < starty+self.obs_dnu))[0]
                    if len(use) == 0:
                        arr[ii, jj] = np.nan
                    else:
                        arr[ii, jj] = np.sum(self.bg_corr[use])
                    gridy[jj] = starty + self.obs_dnu/2.0
                    starty += self.obs_dnu
                gridx[ii] = startx + self.obs_dnu/n_across/2.0
                starty = min(self.frequency)
                startx += self.obs_dnu/n_across
            smoothed = arr
            dim = smoothed.shape

            smoothed_2 = np.zeros((2*dim[0], dim[1]))
            smoothed_2[0:dim[0], :] = smoothed
            smoothed_2[dim[0]:(2*dim[0]), :] = smoothed
            smoothed = np.swapaxes(smoothed_2, 0, 1)
            extent = [
                min(gridx) - self.obs_dnu/n_across/2.0,
                2*max(gridx) + self.obs_dnu/n_across/2.0,
                min(gridy) - self.obs_dnu/2.0,
                max(gridy) + self.obs_dnu/2.0
            ]
            return smoothed, np.array(list(gridx) + list(gridx + self.obs_dnu)), gridy, extent


    def get_red_noise(self):
        """
        TODO

        """
        # Use as initial guesses for the optimized model
        try:
            pars, _ = curve_fit(
                self.fitbg['functions'][self.nlaws],
                self.bin_freq,
                self.bin_pow,
                p0=self.pars,
                sigma=self.bin_err,
                bounds=self.bounds
            )
        except RuntimeError as _:
            return True
        else:
            self.pars = pars
            self.bg_corr = self.random_pow/models.harvey(self.frequency, self.pars, total=True)
            self.sm_par = 4.0*(self.exp_numax/self.params['numax_sun'])**0.2
            if self.sm_par < 1.0:
                self.sm_par = 1.0
            # save final values for Harvey components
            for n in range(self.nlaws):
                self.fitbg['results'][self.name]['a_%d' % (n+1)].append(self.pars[2*n])
                self.fitbg['results'][self.name]['b_%d' % (n+1)].append(self.pars[2*n+1])
            self.fitbg['results'][self.name]['wn'].append(self.pars[2*self.nlaws])
        return False


    def estimate_dnu(self):
        """
        Estimate a value for dnu.

        """
	
        # define the peak in the ACF
        zoom_lag = self.lag[(self.lag>=self.fitbg['acf_mask'][self.name][0])&(self.lag<=self.fitbg['acf_mask'][self.name][1])]
        zoom_auto = self.auto[(self.lag>=self.fitbg['acf_mask'][self.name][0])&(self.lag<=self.fitbg['acf_mask'][self.name][1])]

		      # fit a Gaussian function to the selected peak in the ACF
        p_gauss3, _ = curve_fit(models.gaussian, zoom_lag, zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb[0])
        # the center of that Gaussian is our estimate for Dnu
        dnu = p_gauss3[2]
        self.fitbg['results'][self.name]['dnu'].append(dnu)
