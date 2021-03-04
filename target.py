import os
import pdb
import glob
import subprocess

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft
from astropy.io import ascii
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.signal import find_peaks

from functions import *
from models import *
from utils import *
from plots import *


class Target:
    """A pipeline target. Initialisation will cause the pipeline to process the target.

    Attributes
    ----------
    target : int
        the target ID
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

    def __init__(self, args):
        self.target = args.target
        self.params = args.params
        self.findex = args.findex
        self.fitbg = args.fitbg
        self.verbose = args.verbose
        self.show_plots = args.show
        self.keplercorr = args.keplercorr
        self.filter = args.filter
        # Run the pipeline
        self.run_syd()


    def run_syd(self):
        """Load target data and run the pipeline routines."""
        # Make sure data load was successsful
        data, self = load_data(self)
        if data is not None:
            # Run the find excess routine
            if self.findex['do']:
                self.find_excess()
            # Run the fit background routine
            if self.fitbg['do']:
                self.fit_background()


    def find_excess(self):
        """
        Automatically finds power excess due to solar-like oscillations using a
        frequency resolved collapsed autocorrelation function.
        """

        N = int(self.findex['n_trials'] + 3)
        if N % 3 == 0:
            self.nrows = (N-1)//3
        else:
            self.nrows = N//3

        if self.findex['binning'] is not None:
            bin_freq, bin_pow = bin_data(self.freq, self.pow, self.findex)
            self.bin_freq = bin_freq
            self.bin_pow = bin_pow
            if self.verbose:
                print('Running FINDEX:')
                print('binned to %d datapoints' % len(self.bin_freq))

            boxsize = int(np.ceil(float(self.findex['smooth_width'])/(bin_freq[1]-bin_freq[0])))
            sp = convolve(bin_pow, Box1DKernel(boxsize))
            smooth_freq = bin_freq[int(boxsize/2):-int(boxsize/2)]
            smooth_pow = sp[int(boxsize/2):-int(boxsize/2)]

            s = InterpolatedUnivariateSpline(smooth_freq, smooth_pow, k=1)
            self.interp_pow = s(self.freq)
            self.bgcorr_pow = self.pow/self.interp_pow

            if self.params[self.target]['numax'] <= 500.:
                boxes = np.logspace(np.log10(0.5), np.log10(25.), self.findex['n_trials'])*1.
            else:
                boxes = np.logspace(np.log10(50.), np.log10(500.), self.findex['n_trials'])*1.

            results = []
            self.md = []
            self.cumsum = []
            self.fit_numax = []
            self.fit_gauss = []
            self.fit_snr = []
            self.fx = []
            self.fy = []

            for i, box in enumerate(boxes):

                subset = np.ceil(box/self.resolution)
                steps = np.ceil((box*self.findex['step'])/self.resolution)

                cumsum = np.zeros_like(self.freq)
                md = np.zeros_like(self.freq)
                j = 0
                start = 0

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

                self.md.append(md[~np.ma.getmask(np.ma.masked_values(cumsum, 0.0))])
                cumsum = cumsum[~np.ma.getmask(np.ma.masked_values(cumsum, 0.0))] - min(cumsum[~np.ma.getmask(np.ma.masked_values(cumsum, 0.0))])
                cumsum = list(cumsum/max(cumsum))
                idx = cumsum.index(max(cumsum))
                self.cumsum.append(np.array(cumsum))
                self.fit_numax.append(self.md[i][idx])

                try:
                    best_vars, _ = curve_fit(gaussian, self.md[i], self.cumsum[i], p0=[np.mean(self.cumsum[i]), 1.0-np.mean(self.cumsum[i]), self.md[i][idx], self.params['width_sun']*(self.md[i][idx]/self.params['numax_sun'])])
                except Exception as _:
                    results.append([self.target, np.nan, np.nan, -np.inf])
                else:
                    self.fx.append(np.linspace(min(md), max(md), 10000))
                    # self.fy.append(gaussian(self.fx[i], best_vars[0], best_vars[1], best_vars[2], best_vars[3]))
                    self.fy.append(gaussian(self.fx[i], *best_vars))
                    snr = max(self.fy[i])/best_vars[0]
                    if snr > 100.:
                        snr = 100.
                    self.fit_snr.append(snr)
                    self.fit_gauss.append(best_vars[2])
                    results.append([self.target, best_vars[2], delta_nu(best_vars[2]), snr])
                    if self.verbose:
                        print('power excess trial %d: numax = %.2f +/- %.2f' % (i+1, best_vars[2], np.absolute(best_vars[3])/2.0))
                        print('S/N: %.2f' % snr)

            compare = [each[-1] for each in results]
            best = compare.index(max(compare))
            if self.verbose:
                print('picking model %d' % (best+1))
            save_findex(self, results[best])
            plot_findex(self)


    def fit_background(self, result=''):
        """Perform a fit to the granulation background and measures the frequency of maximum power (numax),
        the large frequency separation (dnu) and oscillation amplitude.

        Parameters
        ----------
        result : str
            TODO: Currently unused
        """

        # Will only run routine if there is a prior numax estimate
        if check_fitbg(self):
            self = get_initial_guesses(self)
            self.final_pars = {
                'numax_smooth': [],
                'amp_smooth': [],
                'numax_gaussian': [],
                'amp_gaussian': [],
                'fwhm_gaussian': [],
                'dnu': [],
                'wn': []
            }
            # Sampling process
            i = 0
            while i < self.fitbg['num_mc_iter']:
                self.i = i
                if self.i == 0:
                    # Record original PS information for plotting
                    self.random_pow = np.copy(self.power)
                    bin_freq, bin_pow, bin_err = mean_smooth_ind(self.frequency, self.random_pow, self.fitbg['ind_width'])
                    if self.verbose:
                        print('-------------------------------------------------')
                        print('Running FITBG:')
                        print('binned to %d data points' % len(bin_freq))
                else:
                    # Randomize power spectrum to get uncertainty on measured values
                    self.random_pow = (np.random.chisquare(2, len(self.frequency))*self.power)/2.
                    bin_freq, bin_pow, bin_err = mean_smooth_ind(self.frequency, self.random_pow, self.fitbg['ind_width'])
                self.bin_freq = bin_freq[~((bin_freq > self.maxpower[0]) & (bin_freq < self.maxpower[1]))]
                self.bin_pow = bin_pow[~((bin_freq > self.maxpower[0]) & (bin_freq < self.maxpower[1]))]
                self.bin_err = bin_err[~((bin_freq > self.maxpower[0]) & (bin_freq < self.maxpower[1]))]
                # Estimate white noise level
                self.get_white_noise()

                # Exclude region with power excess and smooth to estimate red/white noise components
                boxkernel = Box1DKernel(int(np.ceil(self.fitbg['box_filter']/self.resolution)))
                self.params[self.target]['mask'] = (self.frequency >= self.maxpower[0]) & (self.frequency <= self.maxpower[1])
                self.smooth_pow = convolve(self.random_pow[~self.params[self.target]['mask']], boxkernel)

                # Temporary array for inputs into model optimization (changes with each iteration)
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

                # Smooth power spectrum - ONLY for plotting purposes, not used in subsequent analyses (TODO: this should not be done for all iterations!)
                self.smooth_power = convolve(self.power, Box1DKernel(int(np.ceil(self.fitbg['box_filter']/self.resolution))))
                # If optimization does not converge, the rest of the code will not run
                if self.get_red_noise():
                    continue
                # save final values for Harvey laws from model fit
                for n in range(self.nlaws):
                    self.final_pars['a_%d' % (n+1)].append(self.pars[2*n])
                    self.final_pars['b_%d' % (n+1)].append(self.pars[2*n+1])
                self.final_pars['wn'].append(self.pars[2*self.nlaws])

                # Estimate numax by 1) smoothing power and 2) fitting Gaussian
                self.get_numax_smooth()
                if list(self.region_freq) != []:
                    self.get_numax_gaussian()
                self.bg_corr = self.random_pow/harvey(self.frequency, self.pars, total=True)

                # Optional smoothing of PS to remove fine structure before computing ACF
                if self.fitbg['smooth_ps'] is not None:
                    boxkernel = Box1DKernel(int(np.ceil(self.fitbg['smooth_ps']/self.resolution)))
                    self.bg_corr_smooth = convolve(self.bg_corr, boxkernel)
                else:
                    self.bg_corr_smooth = np.copy(self.bg_corr)

                # Calculate ACF using ffts (default) and estimate large frequency separation
                dnu = self.get_frequency_spacing()
                self.final_pars['dnu'].append(dnu)
                if self.i == 0:
                    self.get_ridges()
                    plot_fitbg(self)
                i += 1

            # Save results
            save_fitbg(self)
            # Multiple iterations
            if self.fitbg['num_mc_iter'] > 1:
                # Plot results of Monte-Carlo sampling
                plot_mc(self)
                if self.verbose:
                    print('numax (smoothed): %.2f +/- %.2f muHz' % (self.final_pars['numax_smooth'][0], mad_std(self.final_pars['numax_smooth'])))
                    print('maxamp (smoothed): %.2f +/- %.2f ppm^2/muHz' % (self.final_pars['amp_smooth'][0], mad_std(self.final_pars['amp_smooth'])))
                    print('numax (gaussian): %.2f +/- %.2f muHz' % (self.final_pars['numax_gaussian'][0], mad_std(self.final_pars['numax_gaussian'])))
                    print('maxamp (gaussian): %.2f +/- %.2f ppm^2/muHz' % (self.final_pars['amp_gaussian'][0], mad_std(self.final_pars['amp_gaussian'])))
                    print('fwhm (gaussian): %.2f +/- %.2f muHz' % (self.final_pars['fwhm_gaussian'][0], mad_std(self.final_pars['fwhm_gaussian'])))
                    print('dnu: %.2f +/- %.2f muHz' % (self.final_pars['dnu'][0], mad_std(self.final_pars['dnu'])))
                    print('-------------------------------------------------')
                    print()
            # Single iteration
            else:
                if self.verbose:
                    print('numax (smoothed): %.2f muHz' % (self.final_pars['numax_smooth'][0]))
                    print('maxamp (smoothed): %.2f ppm^2/muHz' % (self.final_pars['amp_smooth'][0]))
                    print('numax (gaussian): %.2f muHz' % (self.final_pars['numax_gaussian'][0]))
                    print('maxamp (gaussian): %.2f ppm^2/muHz' % (self.final_pars['amp_gaussian'][0]))
                    print('fwhm (gaussian): %.2f muHz' % (self.final_pars['fwhm_gaussian'][0]))
                    print('dnu: %.2f' % (self.final_pars['dnu'][0]))
                    print('-------------------------------------------------')
                    print()

    def get_white_noise(self):
        """Estimate white level by taking a mean over a section of the power spectrum."""

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

    def get_red_noise(
            self,
            names=['one', 'one', 'two', 'two', 'three', 'three', 'four', 'four', 'five', 'five', 'six', 'six'],
            bounds=[],
            reduced_chi2=[],
            paras=[],
            a=[]
    ):
        """Fits a Harvey model for the stellar granulation background for the power spectrum.

        Parameters
        ----------
        names : list
            the Harvey components to use in the background model
        bounds : list
            the bounds on the Harvey parameters
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
        if self.i == 0:
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
                            f_obs=self.random_pow[~self.params[self.target]['mask']],
                            f_exp=harvey(
                                self.frequency[~self.params[self.target]['mask']],
                                pp,
                                total=True
                            )
                        )
                        reduced_chi2.append(chi/(len(self.frequency[~self.params[self.target]['mask']])-len(pams)))
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
                            f_obs=self.random_pow[~self.params[self.target]['mask']],
                            f_exp=harvey(
                                self.frequency[~self.params[self.target]['mask']],
                                pp,
                                total=True
                                )
                            )
                        reduced_chi2.append(chi/(len(self.frequency[~self.params[self.target]['mask']])-len(pams)+1))

            # Fitting succeeded
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
                self.exp_numax = self.params[self.target]['numax']
                self.exp_dnu = self.params[self.target]['dnu']
                self.sm_par = 4.*(self.exp_numax/self.params['numax_sun'])**0.2
                if self.sm_par < 1.:
                    self.sm_par = 1.
                for n in range(self.nlaws):
                    self.final_pars['a_%d' % (n+1)] = []
                    self.final_pars['b_%d' % (n+1)] = []
                again = False
            else:
                again = True
        else:
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
                again = True
            else:
                self.pars = pars
                self.sm_par = 4.0*(self.exp_numax/self.params['numax_sun'])**0.2
                if self.sm_par < 1.0:
                    self.sm_par = 1.0
                again = False
        return again


    def get_numax_smooth(self):
        """Estimate numax by smoothing the power spectrum and taking the peak."""

        sig = (self.sm_par*(self.exp_dnu/self.resolution))/np.sqrt(8.0*np.log(2.0))
        pssm = convolve_fft(np.copy(self.random_pow), Gaussian1DKernel(int(sig)))
        model = harvey(self.frequency, self.pars, total=True)
        inner_freq = list(self.frequency[self.params[self.target]['mask']])
        inner_obs = list(pssm[self.params[self.target]['mask']])
        outer_freq = list(self.frequency[~self.params[self.target]['mask']])
        outer_mod = list(model[~self.params[self.target]['mask']])
        if self.fitbg['slope']:
            # Correct for edge effects and residual slope in Gaussian fit
            inner_mod = model[self.params[self.target]['mask']]
            delta_y = inner_obs[-1]-inner_obs[0]
            delta_x = inner_freq[-1]-inner_freq[0]
            slope = delta_y/delta_x
            b = slope*(-1.0*inner_freq[0]) + inner_obs[0]
            corrected = np.array([inner_freq[z]*slope + b for z in range(len(inner_freq))])
            corr_pssm = [inner_obs[z] - corrected[z] + inner_mod[z] for z in range(len(inner_obs))]
            final_y = np.array(corr_pssm + outer_mod)
        else:
            outer_freq = list(self.frequency[~self.params[self.target]['mask']])
            outer_mod = list(model[~self.params[self.target]['mask']])
            final_y = np.array(inner_obs + outer_mod)
        final_x = np.array(inner_freq + outer_freq)
        ss = np.argsort(final_x)
        final_x = final_x[ss]
        final_y = final_y[ss]
        self.pssm = np.copy(final_y)
        self.pssm_bgcorr = self.pssm-harvey(final_x, self.pars, total=True)
        self.region_freq = self.frequency[self.params[self.target]['mask']]
        self.region_pow = self.pssm_bgcorr[self.params[self.target]['mask']]
        idx = return_max(self.region_pow, index=True)
        self.final_pars['numax_smooth'].append(self.region_freq[idx])
        self.final_pars['amp_smooth'].append(self.region_pow[idx])
        # Initial guesses for the parameters of the Gaussian fit to the power envelope
        self.guesses = [
            0.0,
            max(self.region_pow),
            self.region_freq[idx],
            (max(self.region_freq) - min(self.region_freq))/np.sqrt(8.0*np.log(2.0))
        ]


    def get_numax_gaussian(self):
        """Estimate numax by fitting a Gaussian to the power envelope of the smoothed power spectrum."""
        bb = gaussian_bounds(self.region_freq, self.region_pow)
        p_gauss1, _ = curve_fit(gaussian, self.region_freq, self.region_pow, p0=self.guesses, bounds=bb[0])
        # create array with finer resolution for purposes of quantifying uncertainty
        new_freq = np.linspace(min(self.region_freq), max(self.region_freq), 10000)
        # numax_fit = list(gaussian(new_freq, p_gauss1[0], p_gauss1[1], p_gauss1[2], p_gauss1[3]))
        numax_fit = list(gaussian(new_freq, *p_gauss1))
        d = numax_fit.index(max(numax_fit))
        self.final_pars['numax_gaussian'].append(new_freq[d])
        self.final_pars['amp_gaussian'].append(p_gauss1[1])
        self.final_pars['fwhm_gaussian'].append(p_gauss1[3])
        if self.i == 0:
            self.exp_numax = new_freq[d]
            self.exp_dnu = 0.22*(self.exp_numax**0.797)
            self.width = self.params['width_sun']*(self.exp_numax/self.params['numax_sun'])/2.
            self.new_freq = np.copy(new_freq)
            self.numax_fit = np.array(numax_fit)


    def get_frequency_spacing(self):
        """Estimate the large frequency spacing or dnu.

        Parameters
        ----------
        dnu : float
            the estimated value of dnu
        """

        # Compute the ACF
        self.compute_acf()
        dnu = self.estimate_dnu()
        return dnu


    def compute_acf(self, fft=True):
        """Compute the ACF of the smooth background corrected power spectrum.

        Parameters
        ----------
        fft : bool
            if true will use FFT to compute the ACF
        """

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
        if self.i == 0:
            self.freq = self.frequency[self.params[self.target]['mask']]
            self.psd = self.bg_corr_smooth[self.params[self.target]['mask']]
            self.peaks_f, self.peaks_p = max_elements(self.freq, self.psd, self.fitbg['n_peaks'])

    def get_acf_cutout(self):
        """Center on the closest peak to expected dnu and cut out a region around the peak for dnu uncertainties."""
        lag,auto=self.lag,self.auto
        lag_of_peak,acf_of_peak=self.lag_of_peak,self.acf_of_peak
        hmax=acf_of_peak/2.  #half of ACF peak
        lag_idx=np.where(lag==lag_of_peak)[0][0]
    
        # Find Full Width at Half Maximum (FWHM):
        right_indices=np.where(auto[lag_idx:]<hmax)[0]
        right_idx=right_indices[0]
        
        left_indices=np.where(auto[:lag_idx]<hmax)[0]
        left_idx=left_indices[-1]

        left_fwhm_idx=np.where(lag==(lag[:lag_idx][left_idx]))[0]    #index in lag&auto of left FWHM val
        right_fwhm_idx=np.where(lag==(lag[lag_idx:][right_idx]))[0]  #index in lag&auto of right FWHM val
        left_fwhm,right_fwhm=lag[left_fwhm_idx],lag[right_fwhm_idx]  #vals of FWHM on both sides of peak

        # MASK limits using FWHM method:
        threshold=1.0
        fw= right_fwhm-left_fwhm   #full width
        frac_fwhm = threshold*fw   #fraction of FWHM for ACF MASK
        left_lim_FWHM,right_lim_FWHM=lag_of_peak-frac_fwhm,lag_of_peak+frac_fwhm
        left_lim,right_lim=left_lim_FWHM,right_lim_FWHM

        idx=np.where((left_lim <= lag) & (lag <= right_lim))[0]  #get indices of ACF mask
        
        return lag[idx],auto[idx]


    def estimate_dnu(self):
        """Estimate a value for dnu."""

		      # for the first iteration (real data) only, estimate the peak on the ACF to fit
        if self.i == 0:
            # Get peaks from ACF
            peak_idx,_ = find_peaks(self.auto) #indices of peaks, threshold=half max(ACF)
            peaks_l,peaks_a = self.lag[peak_idx],self.auto[peak_idx]

            # pick the peak closest to the exp_dnu
            idx = return_max(peaks_l, index=True, dnu=True, exp_dnu=self.exp_dnu)

            best_lag=(self.lag[peak_idx])[idx]           #best estimate of dnu
            best_auto=(self.auto[peak_idx])[idx]         #best estimate of dnu
            self.lag_of_peak=(self.lag[peak_idx])[idx]   #lag val corresponding to peak
            self.acf_of_peak=(self.auto[peak_idx])[idx]  #acf val corresponding to peak

            og_zoom_lag,og_zoom_auto=self.get_acf_cutout()
            self.fitbg['acf_mask']=[min(og_zoom_lag),max(og_zoom_lag)]  # lag limits to use for ACF mask
	
        # define the peak in the ACF
        zoom_lag = self.lag[(self.lag>=self.fitbg['acf_mask'][0])&(self.lag<=self.fitbg['acf_mask'][1])]
        zoom_auto = self.auto[(self.lag>=self.fitbg['acf_mask'][0])&(self.lag<=self.fitbg['acf_mask'][1])]

		      # boundary conditions and initial guesses stay the same as for the first iteration
        if self.i == 0:
            self.acf_bb = gaussian_bounds(zoom_lag, zoom_auto, best_x=best_lag, sigma=10**-2)
            self.acf_guesses = [np.mean(zoom_auto), best_auto, best_lag, best_lag*0.01*2.]

		      # fit a Gaussian function to the selected peak in the ACF
        p_gauss3, p_cov3 = curve_fit(gaussian, zoom_lag, zoom_auto, p0=self.acf_guesses, bounds=self.acf_bb[0])
        # the center of that Gaussian is our estimate for Dnu
        dnu = p_gauss3[2]
        
   
        if self.i == 0:
			         # variables for plotting. this only needs to be done during the first iteration
            new_lag = np.linspace(min(zoom_lag),max(zoom_lag),2000)
            dnu     = new_lag[np.argmax(gaussian(new_lag,*p_gauss3))] #value of Gaussian peak
            dnu_fit = gaussian(new_lag,*p_gauss3)                     #Gaussian fit to zoom_ACF
            peaks_l[idx] = np.nan
            peaks_a[idx] = np.nan
            self.peaks_l = peaks_l
            self.peaks_a = peaks_a
            self.best_lag = best_lag
            self.best_auto = best_auto
            self.zoom_lag = og_zoom_lag
            self.zoom_auto = og_zoom_auto
            self.obs_dnu = dnu
            self.new_lag = new_lag
            self.dnu_fit = dnu_fit

        return dnu

    def get_ridges(self, start=0.0):
        """Create echelle diagram.

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
        if self.fitbg['clip']:
            if self.fitbg['clip_value'] != 0.:
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
        """Creates an echelle diagram.

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

        if self.fitbg['ech_smooth']:
            boxkernel = Box1DKernel(int(np.ceil(self.fitbg['ech_filter']/self.resolution)))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Python version of asteroseismic 'SYD'
                            pipeline ( Huber+2009). This script will initialize
                            the SYD-PY pipeline, which is broken up into two main modules:
                            1) find excess (findex)
                            2) fit background (fitbg).
                            By default, both modules will run unless otherwise specified.
                            See -excess and -fitbg for more details.
                            SYD-PY is actively being developed at
                            https://github.com/ashleychontos/SYD-PYpline
                            by: Ashley Chontos (achontos@hawaii.edu)
                                Maryum Sayeed
                                Daniel Huber (huberd@hawaii.edu)
                            Please contact Ashley for more details or new ideas for
                            implementations within the package.
                            [The ReadTheDocs page is currently under construction.]"""
    )
    parser.add_argument(
        '-ex', '--ex', '-findex', '--findex', '-excess', '--excess',
        help="""Turn off the find excess module. This is only recommended when a list
                    of numaxes or a list of stellar parameters (to estimate the numaxes)
                    are provided. Otherwise the second module, which fits the background
                    will not be able to run properly.""",
        default=True, dest='excess', action='store_false'
    )
    parser.add_argument(
        '-bg', '--bg', '-fitbg', '--fitbg', '-background', '--background',
        help="""Turn off the background fitting process (although this is not recommended).
                    Asteroseismic estimates are typically unreliable without properly removing
                    stellar contributions from granulation processes. Since this is the money
                    maker, fitbg is set to 'True' by default.""",
        default=True, dest='background', action='store_false'
    )
    parser.add_argument(
        '-filter', '--filter', '-smooth', '--smooth',
        help='Box filter width [muHz] for the power spectrum (Default = 2.5 muHz)',
        default=2.5, dest='filter'
    )
    parser.add_argument(
        '-kc', '--kc', '-keplercorr', '--keplercorr',
        help="""Turn on Kepler short-cadence artefact corrections""",
        default=False, dest='keplercorr', action='store_true'
    )
    parser.add_argument(
        '-v', '--v', '-verbose', '--verbose',
        help="""Turn on the verbose output. 
                    Please note: the defaults is 'False'.""",
        default=False, dest='verbose', action='store_true'
    )
    parser.add_argument(
        '-show', '--show', '-plot', '--plot', '-plots', '--plots', '-p',
        help="""Shows the appropriate output figures in real time. If the findex module is
                    run, this will show one figure at the end of findex. If the fitbg module is
                    run, a figure will appear at the end of the first iteration. If the monte
                    carlo sampling is turned on, this will provide another figure at the end of
                    the MC iterations. Regardless of this option, the figures will be saved to
                    the output directory. If running more than one target, this is not
                    recommended. """,
        default=False, dest='show', action='store_true'
    )
  
    parser.add_argument(
        '-mc', '--mc', '-mciter', '--mciter', dest='mciter', default=1, type=int,
        help='Number of MC iterations (Default = 1)'
    )

    main(parser.parse_args())
