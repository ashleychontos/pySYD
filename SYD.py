import os
import glob
import argparse
import subprocess
import matplotlib
import numpy as np
import pandas as pd
from functions import *
from astropy.io import ascii
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from astropy.stats import mad_std
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel, Box1DKernel
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter, ScalarFormatter




def main(findex=True, fitbg=True, verbose=True, show_plots=True, ignore=False):

    PS = PowerSpectrum(findex, fitbg, verbose, show_plots, ignore)

    for target in PS.params['todo']:
        PS.target = target
        if PS.load_data():
            if PS.findex['do']:
                PS.find_excess()
            if PS.fitbg['do']:
                PS.fit_background()

    if PS.verbose:
        print('Combining results into single csv file.')
        print()
        subprocess.call(['python scrape_output.py'], shell=True)

    return


##########################################################################################
#                                                                                        #
#                                      DICTIONARIES                                      #
#                                                                                        #
##########################################################################################


class PowerSpectrum:
    
    def __init__(self, findex, fitbg, verbose, show_plots, ignore, correct=False):
        self.findex = {}
        self.findex['do'] = findex
        self.fitbg = {}
        self.fitbg['do'] = fitbg
        self.verbose = verbose
        self.show_plots = show_plots
        self.ignore = ignore
        self.get_info()
        self.set_plot_params()

    def get_info(self, star_info = 'Files/star_info.csv'):

        self.params = {}
        with open('Files/todo.txt', "r") as f:
            todo = np.array([int(float(line.strip().split()[0])) for line in f.readlines()])
        if len(todo) > 1 and not self.ignore:
            self.verbose = False
            self.show_plots = False
        self.params['path'] = 'Files/data/'
        self.params.update({'numax_sun':3090., 'dnu_sun':135.1, 'width_sun':1300., 'todo':todo, 'G':6.67428e-8, 
                            'tau_sun':[5.2e6, 1.8e5, 1.7e4, 2.5e3, 280., 80.], 'teff_sun':5777., 'mass_sun':1.9891e33,
                            'tau_sun_single':[3.8e6, 2.5e5, 1.5e5, 1.0e5, 230., 70.], 'radius_sun':6.95508e10})
        for target in todo:
            self.params[target] = {}
            self.params[target]['path'] = '/'.join(self.params['path'].split('/')[:-2])+'/results/%d/'%target

        if self.findex['do']:
            self.read_excess_params('Files/params_findex.txt')
        if self.fitbg['do']:
            self.read_bg_params('Files/params_fitbg.txt')

        self.get_star_info(star_info)

    def set_plot_params(self):

        plt.style.use('dark_background')
        plt.rcParams.update({
                             'agg.path.chunksize': 10000,
                             'mathtext.fontset': 'stix',
                             'figure.autolayout': True,
                             'lines.linewidth': 1,
                             'axes.titlesize': 18.,
                             'axes.labelsize': 16.,
                             'axes.linewidth': 1.25,
                             'axes.formatter.useoffset': False,
                             'xtick.major.size':10.,
                             'xtick.minor.size':5.,
                             'xtick.major.width':1.25,
                             'xtick.minor.width':1.25,
                             'xtick.direction': 'inout',
                             'ytick.major.size':10.,
                             'ytick.minor.size':5.,
                             'ytick.major.width':1.25,
                             'ytick.minor.width':1.25,
                             'ytick.direction': 'inout',
    })

##########################################################################################
#                                                                                        #
#                               READING/WRITING TO/FROM FILES                            #
#                                                                                        #
##########################################################################################

    def read_excess_params(self, findex_file):

        pars = ['box', 'step', 'lower_lag', 'upper_lag', 'binning', 'mode', 'smooth_width', 'check', 'lower_limit', 'upper_limit', 'plot', 'long_cadence', 'check_step', 'n_trials', 'save']
        dtype = [False, False, False, False, False, True, False, True, False, False, True, True, True, True, True]
        vals = []

        i = 0
        with open(findex_file, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    val = line.strip().split()[0]
                    if val == 'None':
                        vals.append(None)
                    else:
                        if dtype[i]:
                            vals.append(int(float(line.strip().split()[0])))
                        else:
                            vals.append(float(line.strip().split()[0]))
                    i += 1

        if self.verbose:
            print()
            print('# FIND EXCESS PARAMS: %d valid lines read'%i)

        self.findex.update(dict(zip(pars, vals)))
        if self.findex['save']:
            for target in self.params['todo']:
                if not os.path.exists(self.params[target]['path']):
                    os.makedirs(self.params[target]['path'])

    def read_bg_params(self, fitbg_file):

        pars = ['lower_limit', 'upper_limit', 'num_mc_iter', 'n_laws', 'lower_noise', 'upper_noise', 'fix_wn', 'box_filter', 'ind_width', 'n_rms', 'lower_numax', 'upper_numax', 'lower_lag', 'upper_lag', 'n_peaks', 'smooth_ps', 'plot', 'force', 'guess', 'save', 'clip', 'clip_value', 'ech_smooth', 'ech_filter']
        dtype = [False, False, True, True, False, False, True, False, True, True, False, False, False, False, True, False, True, True, False, True, True, False, True, False]
        vals = []

        i = 0
        with open(fitbg_file, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    val = line.strip().split()[0]
                    if val == 'None':
                        vals.append(None)
                    else:
                        if dtype[i]:
                            vals.append(int(float(line.strip().split()[0])))
                        else:
                            vals.append(float(line.strip().split()[0]))
                    i += 1

        if self.verbose:
            print('# FIT BACKGROUND PARAMS: %d valid lines read'%i)
            print()

        self.fitbg.update(dict(zip(pars, vals)))
        self.fitbg['functions'] = {1:harvey_one, 2:harvey_two, 3:harvey_three}
        if self.fitbg['save']:
            for target in self.params['todo']:
                if not os.path.exists(self.params[target]['path']):
                    os.makedirs(self.params[target]['path'])

    def get_star_info(self, star_info, cols = ['rad', 'logg', 'teff']):
        if os.path.exists(star_info):
            df = pd.read_csv(star_info)
            targets = df.targets.values.tolist()
            for todo in self.params['todo']:
                if todo in targets:
                    idx = targets.index(todo)
                    for col in cols:
                        self.params[todo][col] = df.loc[idx,col]
                    if 'numax' in df.columns.values.tolist():
                        self.params[todo]['numax'] = df.loc[idx,'numax']
                        self.params[todo]['dnu'] =  0.22*(df.loc[idx,'numax']**0.797)
                    else:
                        self.params[todo]['mass'] = (((self.params[todo]['rad']*self.params['radius_sun'])**(2.))*10**(self.params[todo]['logg'])/self.params['G'])/self.params['mass_sun']
                        self.params[todo]['numax'] = self.params['numax_sun']*self.params[todo]['mass']*(self.params[todo]['rad']**(-2.))*((self.params[todo]['teff']/self.params['teff_sun'])**(-0.5))
                        self.params[todo]['dnu'] = self.params['dnu_sun']*(self.params[todo]['mass']**(0.5))*(self.params[todo]['rad']**(-1.5))  

    def get_file(self, path):

        f = open(path, "r")
        lines = f.readlines()
        f.close()

        self.x = np.array([float(line.strip().split()[0]) for line in lines])
        self.y = np.array([float(line.strip().split()[1]) for line in lines])

    def write_excess(self, results):

        variables = ['target', 'numax', 'dnu', 'snr']
        ascii.write(np.array(results),self.params[self.target]['path']+'%d_findex.csv'%self.target,names=variables,delimiter=',',overwrite=True)

    def write_bgfit(self, results):

        variables = ['target', 'numax(smooth)', 'numax(smooth)_err', 'maxamp(smooth)', 'maxamp(smooth)_err', 'numax(gauss)', 'numax(gauss)_err', 'maxamp(gauss)', 'maxamp(gauss)_err', 'fwhm', 'fwhm_err', 'dnu', 'dnu_err']
        ascii.write(np.array(results),self.params[self.target]['path']+'%d_globalpars.csv'%self.target,names=variables,delimiter=',',overwrite=True)

    def load_data(self):

        # now done at beginning to make sure it only does this one per target
        if glob.glob(self.params['path']+'%d_*'%self.target) != []:
            # load light curve
            if not os.path.exists(self.params['path']+'%d_LC.txt'%self.target):
                if self.verbose:
                    print('Error: %s%d_LC.txt not found'%(self.params['path'], self.target))
                return False
            else:
                self.get_file(self.params['path']+'%d_LC.txt'%self.target)
                self.time = np.copy(self.x)
                self.flux = np.copy(self.y)
                self.cadence = int(np.nanmedian(np.diff(self.time)*24.*60.*60.))
                self.nyquist = 10**6/(2.*self.cadence)
                if self.verbose:
                    print('# LIGHT CURVE: %d lines of data read'%len(self.time))

            # load power spectrum
            if not os.path.exists(self.params['path']+'%d_PS.txt'%self.target):
                if self.verbose:
                    print('Error: %s%d_PS.txt not found'%(self.params['path'], self.target))
                return False
            else:
                self.get_file(self.params['path']+'%d_PS.txt'%self.target)
                self.frequency = np.copy(self.x)
                self.power = np.copy(self.y)
                if self.verbose:
                    print('# POWER SPECTRUM: %d lines of data read'%len(self.frequency))
            self.oversample = int(round((1./((max(self.time)-min(self.time))*0.0864))/(self.frequency[1]-self.frequency[0])))
            self.resolution = (self.frequency[1]-self.frequency[0])*self.oversample
            if self.verbose:
                print('-------------------------------------------------')
                print('Target: %d'%self.target)
                if self.oversample == 1:
                    print('critically sampled')
                else:
                    print('oversampled by a factor of %d'%self.oversample)
                print('time series cadence: %d seconds'%self.cadence)
                print('power spectrum resolution: %.6f muHz'%self.resolution)
                print('-------------------------------------------------')

            return True
        else:
            print('Error: data not found for target %d'%self.target)
            return False

##########################################################################################
#                                                                                        #
#                            [CRUDE] FIND POWER EXCESS ROUTINE                           #
#                                                                                        #
##########################################################################################
# TODOS
# 1) add in process to check the binning/crude bg fit and retry if desired
# 2) allow user to pick model instead of it picking the highest SNR
# 3) check if the gaussian is fully resolved
# 4) maybe change the part that guesses the offset (mean of entire frequency range - not just the beginning)
# ADDED
# 1) Ability to add more trials for numax determination

    def find_excess(self):

        if self.findex['lower_limit'] is not None:
            if self.findex['upper_limit'] is not None:
                mask = np.ma.getmask(np.ma.masked_inside(self.frequency, self.findex['lower_limit'], self.findex['upper_limit']))
            else:
                mask = np.ma.getmask(np.ma.masked_greater_equal(self.frequency, self.findex['lower_limit']))
        else:
            if self.findex['upper_limit'] is not None:
                mask = np.ma.getmask(np.ma.masked_less_equal(self.frequency, self.findex['upper_limit']))
            else:
                mask = np.ones_like(self.frequency)
        self.mask = mask
        frequency = self.frequency[mask]
        power = self.power[mask]

        N = int(self.findex['n_trials']+3)
        if N%3 == 0:
            self.nrows = (N-1)//3
        else:
            self.nrows = N//3

        if self.findex['binning'] is not None:
            bf, bp = bin_data(frequency, power, self.findex['binning'])
            self.bin_freq = np.copy(bf)
            self.bin_pow = np.copy(bp)
            if self.verbose:
                print('binned to %d datapoints'%len(bf))

            boxsize = np.ceil(float(self.findex['smooth_width'])/self.resolution)
            smooth_pow = convolve(bp, Box1DKernel(boxsize))
            sf = bf[int(boxsize/2):-int(boxsize/2)]
            sp = smooth_pow[int(boxsize/2):-int(boxsize/2)]

            s = InterpolatedUnivariateSpline(sf, sp, k = 1)
            self.interp_pow = s(frequency)
            self.bgcorr_pow = power/self.interp_pow

            if self.findex['long_cadence']:
                boxes = np.logspace(np.log10(0.5), np.log10(25.), self.findex['n_trials'])*self.findex['box']
            else:
                boxes = np.logspace(np.log10(50.), np.log10(1000.), self.findex['n_trials'])*self.findex['box']

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

                cumsum = np.zeros_like(frequency)
                md = np.zeros_like(frequency)
                j = 0
                start = 0

                while True:
                    if (start+subset) > len(frequency):
                        break
                    f = frequency[int(start):int(start+subset)]
                    p = self.bgcorr_pow[int(start):int(start+subset)]

                    lag = np.arange(0., len(p))*self.resolution
                    auto = np.real(np.fft.fft(np.fft.ifft(p)*np.conj(np.fft.ifft(p))))
                    auto = auto[np.ma.getmask(np.ma.masked_inside(lag, self.findex['lower_lag'], self.findex['upper_lag']))]
                    lag = lag[np.ma.getmask(np.ma.masked_inside(lag, self.findex['lower_lag'], self.findex['upper_lag']))]
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
                    best_vars, covar = curve_fit(gaussian, self.md[i], self.cumsum[i], p0 = [np.mean(self.cumsum[i]), 1.0-np.mean(self.cumsum[i]), self.md[i][idx], self.params['width_sun']*(self.md[i][idx]/self.params['numax_sun'])])
                except:
                    pass
                finally:
                    self.fx.append(np.linspace(min(md), max(md), 10000))
                    self.fy.append(gaussian(self.fx[i], best_vars[0], best_vars[1], best_vars[2], best_vars[3]))
                    snr = max(self.fy[i])/best_vars[0]
                    if snr > 100.:
                        snr = 100.
                    self.fit_snr.append(snr)
                    self.fit_gauss.append(best_vars[2])
                    results.append([self.target, best_vars[2], delta_nu(best_vars[2]), snr])
                    if self.verbose:
                        print('power excess trial %d: numax = %.2f +/- %.2f'%(i+1, best_vars[2], np.absolute(best_vars[3])/2.))
                        print('S/N: %.2f'%snr)

            compare = [each[-1] for each in results]
            best = compare.index(max(compare))
            if self.verbose:
                print('picking model %d'%(best+1))
            self.write_excess(results[best])

            # if findex module has been run, let that override star info guesses
            df = pd.read_csv(self.params[self.target]['path']+'%d_findex.csv'%self.target)
            for col in ['numax', 'dnu', 'snr']:
                self.params[self.target][col] = df.loc[0,col]
            self.width = self.params['width_sun']*(self.params[self.target]['numax']/self.params['numax_sun'])
            self.times = self.width/self.params[self.target]['dnu']

            if self.findex['plot']:
                self.plot_findex()


##########################################################################################
#                                                                                        #
#                                  FIT BACKGROUND ROUTINE                                #
#                                                                                        #
##########################################################################################
# TODOS
# 1) Change the way the n_laws is modified within the code (i.e. drop Faculae term, long period trends)
# 2) Making sure the correct number of harvey components make it to the final fit
# ADDED
# 1) Ability to change number of points used to calculate RMS of harvey component (default = 10)

    def check(self):

        if not self.findex['do']:
            # SYD needs some prior knowledge about numax to work well 
            # (either from findex module or from star info csv)
            if 'numax' not in self.params[self.target].keys():
                print("WARNING: Suggested use of this pipeline requires either \n stellar properties to estimate a nuMax or run the entire \n pipeline from scratch (i.e. find_excess) first to \n statistically determine a starting point for nuMax.")
                return False
            else:
                return True
        else:
            return True

    def fit_background(self):

        if self.check():

            results = []
            # mask out any unwanted frequencies
            if self.fitbg['lower_limit'] is not None:
                if self.fitbg['upper_limit'] is not None:
                    self.mask = np.ma.getmask(np.ma.masked_inside(self.frequency, self.fitbg['lower_limit'], self.fitbg['upper_limit']))
                else:
                    self.mask = np.ma.getmask(np.ma.masked_greater_equal(self.frequency, self.fitbg['lower_limit']))
            else:
                if self.fitbg['upper_limit'] is not None:
                    self.mask = np.ma.getmask(np.ma.masked_less_equal(self.frequency, self.fitbg['upper_limit']))
                else:
                    self.mask = np.ones_like(self.frequency)
            self.frequency = list(self.frequency[self.mask])
            self.power = list(self.power[self.mask])

            # create independent frequency points (need to if oversampled)
            if self.oversample != 1:
                self.original_freq = np.array(self.frequency[self.oversample-1::self.oversample])
                self.original_pow = np.array(self.power[self.oversample-1::self.oversample])
            else:
                self.original_freq = np.copy(self.frequency)
                self.original_pow = np.copy(self.power)

            # arbitrary snr cut for leaving region out of background fit, ***statistically validate later?
            if self.fitbg['lower_numax'] is not None:
                self.maxpower = [self.fitbg['lower_numax'], self.fitbg['upper_numax']]
            else:
                if self.params[self.target]['snr'] < 2.:
                    self.maxpower = [self.params[self.target]['numax']-self.width/2., self.params[self.target]['numax']+self.width/2.]
                else:
                    self.maxpower = [self.params[self.target]['numax']-self.times*self.params[self.target]['dnu'],self.params[self.target]['numax']+self.times*self.params[self.target]['dnu']]

            i = 0
            self.nlaws = self.fitbg['n_laws']
            # sampling process
            while i < self.fitbg['num_mc_iter']:
                self.i = i
                if self.i == 0:
                    # record original PS information for plotting
                    random_pow = np.copy(self.original_pow)
                    bin_freq, bin_pow, bin_err = mean_smooth_ind(self.original_freq, random_pow, self.fitbg['ind_width'])
                    if self.verbose:
                        print('-------------------------------------------------')
                        print('binned to %d data points'%len(bin_freq))
                else:
                    # randomize power spectrum to get uncertainty on measured values
                    random_pow = (np.random.chisquare(2, len(self.original_freq))*self.original_pow)/2.
                    bin_freq, bin_pow, bin_err = mean_smooth_ind(self.original_freq, random_pow, self.fitbg['ind_width'])

                # estimate white noise level
                if self.fitbg['upper_noise'] is not None:
                    self.noise = np.mean(random_pow[(self.original_freq>self.fitbg['lower_noise'])&(self.original_freq<self.fitbg['upper_noise'])])
                else:
                    self.noise = np.mean(random_pow[(self.original_freq>(max(self.original_freq)-0.1*max(self.original_freq)))])
                pars = np.zeros((self.nlaws*2+1))
                pars[2*self.nlaws] = self.noise

                # exclude region with power excess and smooth to estimate red/white noise components
                boxkernel = Box1DKernel(int(np.ceil(self.fitbg['box_filter']/self.resolution)))
                self.params[self.target]['mask'] = (self.original_freq>=self.maxpower[0])&(self.original_freq<=self.maxpower[1])
                outer_x = self.original_freq[~self.params[self.target]['mask']]
                outer_y = random_pow[~self.params[self.target]['mask']]
                smooth_pow = convolve(outer_y, boxkernel)

                # use scaling relation from sun to get starting points
                scale = self.params['numax_sun']/((self.maxpower[1]+self.maxpower[0])/2.)
                if min(self.original_freq) >= 0.01:
                    taus = np.array(self.params['tau_sun'][int(np.log10(min(self.original_freq)*100.)):])*scale
                else:
                    taus = np.array(self.params['tau_sun'])*scale

                b = 2.*np.pi*(taus*1e-6)
                mnu = (1./taus)*1e5
                a = np.zeros_like(mnu)
                msk = (bin_freq>self.maxpower[0])&(bin_freq<self.maxpower[1])

                if max(mnu) < min(bin_freq[~msk]):
                    scup = min(bin_freq[~msk])/max(mnu)
                    mnu[-1] = scup*mnu[-1]
                    taus[-1] = 1e5/mnu[-1]
                    b[-1] = 2.*np.pi*(taus[-1]*1e-6)

                # estimate amplitude for each harvey component
                for j, nu in enumerate(mnu):
                    idx = 0
                    while self.original_freq[idx] < nu:
                        idx += 1
                    if idx < self.fitbg['n_rms']:
                        a[j] = np.mean(smooth_pow[:self.fitbg['n_rms']])
                    elif (len(smooth_pow)-idx) < self.fitbg['n_rms']:
                        a[j] = np.mean(smooth_pow[-self.fitbg['n_rms']:])
                    else:
                        a[j] = np.mean(smooth_pow[idx-int(self.fitbg['n_rms']/2):idx+int(self.fitbg['n_rms']/2)])

                for n in range(self.nlaws):
                    pars[2*n] = a[n]
                    pars[2*n+1] = b[n]

                if self.i == 0:
                    self.bin_freq = bin_freq[~msk]
                    self.bin_pow = bin_pow[~msk]
                    self.bin_err = bin_err[~msk]
                    self.frequency = np.copy(self.original_freq)
                    self.power = np.copy(self.original_pow)
                    self.mnu = mnu
                    self.a = a
                    smooth = convolve(self.original_pow, Box1DKernel(int(np.ceil(self.fitbg['box_filter']/self.resolution))))
                    self.smooth_power = np.copy(smooth)
                    if self.verbose:
                        print('Comparing %d different models:'%(self.nlaws*2))
                    # get best fit model 
                    bounds = []
                    for law in range(self.nlaws):
                        b = np.zeros((2,2*(law+1)+1)).tolist()
                        for z in range(2*(law+1)):
                            b[0][z] = -np.inf
                            b[1][z] = np.inf
                        b[0][-1] = pars[-1]-0.1
                        b[1][-1] = pars[-1]+0.1
                        bounds.append(tuple(b))
                    reduced_chi2 = []
                    paras = []
                    names = ['one', 'one', 'two', 'two', 'three', 'three']
                    dict1 = dict(zip(np.arange(2*self.nlaws),names[:2*self.nlaws]))
                    for t in range(2*self.nlaws):
                        if t%2 == 0:
                            if self.verbose:
                                print('%d: %s harvey model w/ white noise free parameter'%(t+1, dict1[t]))
                            delta = 2*(self.nlaws-(t//2+1))
                            pams = list(pars[:(-delta-1)])
                            pams.append(pars[-1])
                            pp, cv = curve_fit(self.fitbg['functions'][t//2+1], bin_freq, bin_pow, p0 = pams, sigma = bin_err)
                            paras.append(pp)
                            chi, p = chisquare(f_obs = outer_y, f_exp = harvey(outer_x, pp, total=True))
                            reduced_chi2.append(chi/(len(outer_x)-len(pams)))
                        else:
                            if self.verbose:
                                print('%d: %s harvey model w/ white noise fixed'%(t+1, dict1[t]))
                            delta = 2*(self.nlaws-(t//2+1))
                            pams = list(pars[:(-delta-1)])
                            pams.append(pars[-1])
                            pp, cv = curve_fit(self.fitbg['functions'][t//2+1], bin_freq, bin_pow, p0 = pams, sigma = bin_err, bounds = bounds[t//2])
                            paras.append(pp)
                            chi, p = chisquare(f_obs = outer_y, f_exp = harvey(outer_x, pp, total=True))
                            reduced_chi2.append(chi/(len(outer_x)-len(pams)+1))

                    self.model = reduced_chi2.index(min(reduced_chi2))+1
                    self.nlaws = ((self.model-1)//2)+1
                    if self.verbose:
                        print('Based on reduced chi-squared statistic: model %d'%self.model)
                    pars = paras[self.model-1]
                    self.best_model = self.model-1
                    self.pars = pars
                    final_pars = np.zeros((self.fitbg['num_mc_iter'],self.nlaws*2+12))

                    self.sm_par = 4.*(self.params[self.target]['numax']/self.params['numax_sun'])**0.2
                    if self.sm_par < 1.:
                        self.sm_par = 1.
                    again = False
                else:
                    try:
                        pars, cv = curve_fit(self.fitbg['functions'][self.nlaws], bin_freq, bin_pow, p0 = pams, sigma = bin_err, bounds = bounds[self.nlaws-1])
                    except RuntimeError:
                        again = True
                    else:
                        again = False

                if again:
                    continue

                final_pars[self.i,0:2*self.nlaws+1] = pars

                fwhm = self.sm_par*self.params[self.target]['dnu']/self.resolution
                sig = fwhm/np.sqrt(8*np.log(2))
                gauss_kernel = Gaussian1DKernel(int(sig))
                pssm = convolve_fft(random_pow[:], gauss_kernel)
                model = harvey(self.original_freq, pars, total=True)

                # correct for edge effects and residual slope in Gaussian fit
                x0 = list(self.original_freq[self.params[self.target]['mask']])
                t0 = pssm[self.params[self.target]['mask']]
                t1 = model[self.params[self.target]['mask']]
                delta_y = t0[-1] - t0[0]
                delta_x = x0[-1] - x0[0]
                slope = delta_y/delta_x
                b = slope*(-1.*x0[0]) + t0[0]
                corrected = np.array([x0[z]*slope + b for z in range(len(x0))])
                corr_pssm = [t0[z]-corrected[z] + t1[z] for z in range(len(t0))]
                x2 = list(self.original_freq[~self.params[self.target]['mask']])
                t2 = list(model[~self.params[self.target]['mask']])
                final_x = np.array(x0+x2)
                final_y = np.array(corr_pssm+t2)
                ss = np.argsort(final_x)
                final_x = final_x[ss]
                final_y = final_y[ss]

                pssm = np.copy(final_y)
                pssm_bgcorr = pssm-harvey(final_x, pars, total=True)

                region_freq = np.copy(self.original_freq[self.params[self.target]['mask']])
                region_pow = pssm_bgcorr[self.params[self.target]['mask']]
                idx = self.return_max(region_pow, index=True)
                final_pars[self.i,2*self.nlaws+1] = region_freq[idx]
                final_pars[self.i,2*self.nlaws+2] = region_pow[idx]

                if list(region_freq) != []:
                    bb = self.gaussian_bounds(region_freq, region_pow)
                    guesses = [0., max(region_pow), region_freq[idx], (max(region_freq)-min(region_freq))/8./np.sqrt(8.*np.log(2.))]
                    p_gauss1, p_cov = curve_fit(gaussian, region_freq, region_pow, p0 = guesses, bounds = bb[0])
                    final_pars[self.i,2*self.nlaws+3] = p_gauss1[2]
                    final_pars[self.i,2*self.nlaws+4] = p_gauss1[1]
                    final_pars[self.i,2*self.nlaws+5] = p_gauss1[3]

                self.bg_corr = random_pow/harvey(self.original_freq, pars, total=True)

                # optional smoothing of PS to remove fine structure
                if self.fitbg['smooth_ps'] is not None:
                    boxkernel = Box1DKernel(int(np.ceil(self.fitbg['smooth_ps']/self.resolution)))
                    self.bg_corr_smooth = convolve(self.bg_corr, boxkernel)
                else:
                    self.bg_corr_smooth = np.array(self.bg_corr[:])

                self.width = self.params['width_sun']*(p_gauss1[2]/self.params['numax_sun'])
                self.numax = p_gauss1[2]
                self.sm_par = 4.*(self.numax/self.params['numax_sun'])**0.2
                self.dnu = 0.22*(self.numax**0.797)
                self.times = self.width/self.dnu
                lim_factor = self.times*self.dnu
                if self.fitbg['lower_numax'] is not None:
                    msk = (self.original_freq >= self.fitbg['lower_numax'])&(self.original_freq <= self.fitbg['upper_numax'])
                else:
                    msk = (self.original_freq >= self.numax-lim_factor)&(self.original_freq <= self.numax+lim_factor)

                freq = self.original_freq[msk]
                psd = self.bg_corr_smooth[msk]
                lag, auto = self.corr(freq, psd)
                peaks_l, peaks_a = self.max_elements(list(lag), list(auto), limit = [True, 20.*self.resolution])

                # pick the peak closest to the modeled numax
                idx = self.return_max(peaks_l, index=True, dnu=True)
                best_lag = peaks_l[idx]
                best_auto = peaks_a[idx]
                peaks_l[idx] = np.nan
                peaks_a[idx] = np.nan

                bb = self.gaussian_bounds(lag, auto, best_x=best_lag, sigma=10**-2)
                guesses = [np.mean(auto), best_auto, best_lag, best_lag*0.01*2.]
                p_gauss2, p_cov2 = curve_fit(gaussian, lag, auto, p0 = guesses, bounds = bb[0])

                mask = (lag >= best_lag-3.*p_gauss2[3])&(lag <= best_lag+3.*p_gauss2[3])
                zoom_lag = lag[mask]
                zoom_auto = auto[mask]
                fit = gaussian(zoom_lag, p_gauss2[0], p_gauss2[1], p_gauss2[2], p_gauss2[3])
                idx = self.return_max(fit, index=True)

                if self.fitbg['force']:
                    self.dnu = self.fitbg['guess']
                else:
                    self.dnu = zoom_lag[idx]
                self.get_ridges()
                final_pars[self.i,2*self.nlaws+6] = self.dnu

                if self.i == 0:
                    self.pssm = pssm
                    self.region_freq = region_freq
                    self.region_pow = region_pow
                    self.gauss_1 = p_gauss1
                    self.freq = freq
                    self.psd = psd
                    self.lag = lag
                    self.auto = auto
                    self.best_lag = best_lag
                    self.best_auto = best_auto
                    self.peaks_l = peaks_l
                    self.peaks_a = peaks_a
                    self.zoom_lag = zoom_lag
                    self.zoom_auto = zoom_auto
                    self.gauss_2 = p_gauss2
                    self.plot_fitbg()

                i += 1

            if self.fitbg['num_mc_iter'] > 1:
                if self.verbose:
                    print('numax (smoothed): %.2f +/- %.2f muHz'%(final_pars[0,2*self.nlaws+1],mad_std(final_pars[:,2*self.nlaws+1])))
                    print('maxamp (smoothed): %.2f +/- %.2f ppm^2/muHz'%(final_pars[0,2*self.nlaws+2],mad_std(final_pars[:,2*self.nlaws+2])))
                    print('numax (gaussian): %.2f +/- %.2f muHz'%(final_pars[0,2*self.nlaws+3],mad_std(final_pars[:,2*self.nlaws+3])))
                    print('maxamp (gaussian): %.2f +/- %.2f ppm^2/muHz'%(final_pars[0,2*self.nlaws+4],mad_std(final_pars[:,2*self.nlaws+4])))
                    print('fwhm (gaussian): %.2f +/- %.2f muHz'%(final_pars[0,2*self.nlaws+5],mad_std(final_pars[:,2*self.nlaws+5])))
                    print('dnu: %.2f +/- %.2f muHz'%(final_pars[0,2*self.nlaws+6],mad_std(final_pars[:,2*self.nlaws+6])))
                    print('-------------------------------------------------')
                    print()
                results.append([self.target,final_pars[0,2*self.nlaws+1],mad_std(final_pars[:,2*self.nlaws+1]),final_pars[0,2*self.nlaws+2],mad_std(final_pars[:,2*self.nlaws+2]),final_pars[0,2*self.nlaws+3],mad_std(final_pars[:,2*self.nlaws+3]),final_pars[0,2*self.nlaws+4],mad_std(final_pars[:,2*self.nlaws+4]),final_pars[0,2*self.nlaws+5],mad_std(final_pars[:,2*self.nlaws+5]),final_pars[0,2*self.nlaws+6],mad_std(final_pars[:,2*self.nlaws+6])])
                self.plot_mc(final_pars)

            else:
                if self.verbose:
                    print('numax (smoothed): %.2f +/- %.2f muHz'%(final_pars[0,2*self.nlaws+1]))
                    print('maxamp (smoothed): %.2f +/- %.2f ppm^2/muHz'%(final_pars[0,2*self.nlaws+2]))
                    print('numax (gaussian): %.2f +/- %.2f muHz'%(final_pars[0,2*self.nlaws+3]))
                    print('maxamp (gaussian): %.2f +/- %.2f ppm^2/muHz'%(final_pars[0,2*self.nlaws+4]))
                    print('fwhm (gaussian): %.2f +/- %.2f muHz'%(final_pars[0,2*self.nlaws+5]))
                    print('dnu: %.2f +/- %.2f muHz'%(final_pars[0,2*self.nlaws+6]))
                    print('-------------------------------------------------')
                    print()
                results.append([self.target,final_pars[0,2*self.nlaws+1],0.,final_pars[0,2*self.nlaws+2],0.,final_pars[0,2*self.nlaws+3],0.,final_pars[0,2*self.nlaws+4],0.,final_pars[0,2*self.nlaws+5],0.,final_pars[0,2*self.nlaws+6],0.])

            self.write_bgfit(results[0])

##########################################################################################
#                                                                                        #
#                                    PLOTTING ROUTINES                                   #
#                                                                                        #
##########################################################################################

    def plot_findex(self):

        plt.figure(figsize = (12,8))
       
        # time series data 
        ax1 = plt.subplot(1+self.nrows,3,1)
        ax1.plot(self.time, self.flux, 'w-')
        ax1.set_xlim([min(self.time), max(self.time)])
        ax1.set_title(r'$\rm Time \,\, series$')
        ax1.set_xlabel(r'$\rm Time \,\, [days]$')
        ax1.set_ylabel(r'$\rm Flux$')

        # log-log power spectrum with crude background fit
        ax2 = plt.subplot(1+self.nrows,3,2)
        ax2.loglog(self.frequency[self.mask], self.power[self.mask], 'w-')
        ax2.set_xlim([min(self.frequency[self.mask]), max(self.frequency[self.mask])])
        ax2.set_ylim([min(self.power[self.mask]), max(self.power[self.mask])*1.25])
        ax2.set_title(r'$\rm Crude \,\, background \,\, fit$')
        ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
        if self.findex['binning'] is not None:
            ax2.loglog(self.bin_freq, self.bin_pow, 'r-')
        ax2.loglog(self.frequency[self.mask], self.interp_pow, color = 'lime', linestyle = '-', lw = 2.)

        # crude background-corrected power spectrum
        ax3 = plt.subplot(1+self.nrows,3,3)
        ax3.plot(self.frequency[self.mask], self.bgcorr_pow, 'w-')
        ax3.set_xlim([min(self.frequency[self.mask]), max(self.frequency[self.mask])])
        ax3.set_ylim([0., max(self.bgcorr_pow)*1.25])
        ax3.set_title(r'$\rm Background \,\, corrected \,\, PS$')
        ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')

        # ACF trials to determine numax
        for i in range(self.findex['n_trials']):
            xran = max(self.fx[i])-min(self.fx[i])
            ymax = max(self.cumsum[i])
            if max(self.fy[i]) > ymax:
                ymax = max(self.fy[i])
            yran = np.absolute(ymax)
            ax = plt.subplot(1+self.nrows,3,4+i)
            ax.plot(self.md[i], self.cumsum[i], 'w-')
            ax.axvline(self.fit_numax[i], linestyle = 'dotted', color = 'r', linewidth = 0.75)
            ax.set_title(r'$\rm Collapsed \,\, ACF \,\, [trial \,\, %d]$'%(i+1))
            ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
            ax.set_ylabel(r'$\rm Arbitrary \,\, units$')
            ax.plot(self.fx[i], self.fy[i], color = 'lime', linestyle = '-', linewidth = 1.5)
            ax.axvline(self.fit_gauss[i], color = 'lime', linestyle = '--', linewidth = 0.75)
            ax.set_xlim([min(self.fx[i]), max(self.fx[i])])
            ax.set_ylim([-0.05, ymax+0.15*yran])
            ax.annotate(r'$\rm SNR = %3.2f$'%self.fit_snr[i], xy = (min(self.fx[i])+0.05*xran, ymax+0.025*yran), fontsize = 18)

        plt.tight_layout()
        if self.findex['save']:
            plt.savefig(self.params[self.target]['path']+'%d_findex.png'%self.target, dpi = 300)
        if self.show_plots:
            plt.show()
        plt.close()

    def plot_fitbg(self):

        fig = plt.figure(figsize = (12,12))

        # time series data
        ax1 = fig.add_subplot(3,3,1)
        ax1.plot(self.time, self.flux, 'w-')
        ax1.set_xlim([min(self.time), max(self.time)])
        ax1.set_title(r'$\rm Time \,\, series$')
        ax1.set_xlabel(r'$\rm Time \,\, [days]$')
        ax1.set_ylabel(r'$\rm Flux$')

        # initial background guesses
        ax2 = fig.add_subplot(3,3,2)
        ax2.plot(self.original_freq[self.original_freq<self.maxpower[0]], self.original_pow[self.original_freq<self.maxpower[0]], 'w-', zorder = 0)
        ax2.plot(self.original_freq[self.original_freq>self.maxpower[1]], self.original_pow[self.original_freq>self.maxpower[1]], 'w-', zorder = 0)
        ax2.plot(self.original_freq[self.original_freq<self.maxpower[0]], self.smooth_power[self.original_freq<self.maxpower[0]], 'r-', linewidth = 0.75, zorder = 1)
        ax2.plot(self.original_freq[self.original_freq>self.maxpower[1]], self.smooth_power[self.original_freq>self.maxpower[1]], 'r-', linewidth = 0.75, zorder = 1)
        for r in range(self.nlaws):
            ax2.plot(self.original_freq, harvey(self.original_freq, [self.pars[2*r], self.pars[2*r+1], self.pars[-1]]), color = 'blue', linestyle = ':', linewidth = 1.5, zorder = 3)
        ax2.plot(self.original_freq, harvey(self.original_freq, self.pars, total=True), color = 'blue', linewidth = 2., zorder = 4)
        ax2.errorbar(self.bin_freq, self.bin_pow, yerr = self.bin_err, color = 'lime', markersize = 0., fillstyle = 'none', ls = 'None', marker = 'D', capsize = 3, ecolor = 'lime', elinewidth = 1, capthick = 2, zorder = 2)
        for m, n in zip(self.mnu, self.a):
            ax2.plot(m, n, color = 'blue', fillstyle = 'none', mew = 3., marker = 's', markersize = 5.)
        ax2.axvline(self.maxpower[0], color = 'darkorange', linestyle = 'dashed', linewidth = 2., zorder = 1, dashes = (5,5))
        ax2.axvline(self.maxpower[1], color = 'darkorange', linestyle = 'dashed', linewidth = 2., zorder = 1, dashes = (5,5))
        ax2.axhline(self.noise, color = 'blue', linestyle = 'dashed', linewidth = 1.5, zorder = 3, dashes = (5,5))
        ax2.set_xlim([min(self.original_freq), max(self.original_freq)])
        ax2.set_ylim([min(self.original_pow), max(self.original_pow)*1.25])
        ax2.set_title(r'$\rm Initial \,\, guesses$')
        ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        # fitted background
        ax3 = fig.add_subplot(3,3,3)
        ax3.plot(self.original_freq[self.original_freq<self.maxpower[0]], self.original_pow[self.original_freq<self.maxpower[0]], 'w-', linewidth = 0.75, zorder = 0)
        ax3.plot(self.original_freq[self.original_freq>self.maxpower[1]], self.original_pow[self.original_freq>self.maxpower[1]], 'w-', linewidth = 0.75, zorder = 0)
        ax3.plot(self.original_freq[self.original_freq<self.maxpower[0]], self.smooth_power[self.original_freq<self.maxpower[0]], 'r-', linewidth = 0.75, zorder = 1)
        ax3.plot(self.original_freq[self.original_freq>self.maxpower[1]], self.smooth_power[self.original_freq>self.maxpower[1]], 'r-', linewidth = 0.75, zorder = 1)
        for r in range(self.nlaws):
            ax3.plot(self.original_freq, harvey(self.original_freq, [self.pars[2*r], self.pars[2*r+1], self.pars[-1]]), color = 'blue', linestyle = ':', linewidth = 1.5, zorder = 3)
        ax3.plot(self.original_freq, harvey(self.original_freq, self.pars, total=True), color = 'blue', linewidth = 2., zorder = 4)
        ax3.errorbar(self.bin_freq, self.bin_pow, yerr = self.bin_err, color = 'lime', markersize = 0., fillstyle = 'none', ls = 'None', marker = 'D', capsize = 3, ecolor = 'lime', elinewidth = 1, capthick = 2, zorder = 2)
        ax3.axvline(self.maxpower[0], color = 'darkorange', linestyle = 'dashed', linewidth = 2., zorder = 1, dashes = (5,5))
        ax3.axvline(self.maxpower[1], color = 'darkorange', linestyle = 'dashed', linewidth = 2., zorder = 1, dashes = (5,5))
        ax3.axhline(self.noise, color = 'blue', linestyle = 'dashed', linewidth = 1.5, zorder = 3, dashes = (5,5))
        ax3.plot(self.original_freq, self.pssm, color = 'yellow', linewidth = 2., linestyle = 'dashed', zorder = 5)
        ax3.set_xlim([min(self.original_freq), max(self.original_freq)])
        ax3.set_ylim([min(self.original_pow), max(self.original_pow)*1.25])
        ax3.set_title(r'$\rm Fitted \,\, model$')
        ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        # smoothed power excess w/ gaussian
        ax4 = fig.add_subplot(3,3,4)
        ax4.plot(self.region_freq, self.region_pow, 'w-', zorder = 0)
        idx = self.return_max(self.region_pow, index=True)
        ax4.plot([self.region_freq[idx]], [self.region_pow[idx]], color = 'red', marker = 's', markersize = 7.5, zorder = 0)
        ax4.axvline([self.region_freq[idx]], color = 'white', linestyle = '--', linewidth = 1.5, zorder = 0)
        gaus = gaussian(self.region_freq, self.gauss_1[0], self.gauss_1[1], self.gauss_1[2], self.gauss_1[3])
        plot_min = 0.
        if min(self.region_pow) < plot_min:
            plot_min = min(self.region_pow)
        if min(gaus) < plot_min:
            plot_min = min(gaus)
        plot_max = 0.
        if max(self.region_pow) > plot_max:
            plot_max = max(self.region_pow)
        if max(gaus) > plot_max:
            plot_max = max(gaus)
        plot_range = plot_max-plot_min
        ax4.plot(self.region_freq, gaus, 'b-', zorder = 3)
        ax4.axvline([self.gauss_1[2]], color = 'blue', linestyle = ':', linewidth = 1.5, zorder = 2)
        ax4.plot([self.gauss_1[2]], [self.gauss_1[1]], color = 'b', marker = 'D', markersize = 7.5, zorder = 1)
        ax4.set_title(r'$\rm Smoothed \,\, bg$-$\rm corrected \,\, PS$')
        ax4.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax4.set_xlim([min(self.region_freq), max(self.region_freq)])
        ax4.set_ylim([plot_min-0.1*plot_range, plot_max+0.1*plot_range])

        # background-corrected ps with n highest peaks
        peaks_f, peaks_p = self.max_elements(list(self.freq), list(self.psd), limit = [True, 2.])
        ax5 = fig.add_subplot(3,3,5)
        ax5.plot(self.freq, self.psd, 'w-', zorder = 0, linewidth = 1.0)
        ax5.scatter(peaks_f, peaks_p, s = 25., edgecolor = 'r', marker = 's', facecolor = 'none', linewidths = 1.)
        ax5.set_title(r'$\rm Bg$-$\rm corrected \,\, PS$')
        ax5.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax5.set_ylabel(r'$\rm Power$')
        ax5.set_xlim([self.numax-self.width/2., self.numax+self.width/2.])
        ax5.set_ylim([min(self.psd)-0.025*(max(self.psd)-min(self.psd)), max(self.psd)+0.1*(max(self.psd)-min(self.psd))])

        # ACF for determining dnu
        ax6 = fig.add_subplot(3,3,6)
        ax6.plot(self.lag, self.auto, 'w-', zorder = 0, linewidth = 1.)
        ax6.scatter(self.peaks_l, self.peaks_a, s = 30., edgecolor = 'r', marker = '^', facecolor = 'none', linewidths = 1.)
        ax6.axvline(self.best_lag, color = 'white', linestyle = '--', linewidth = 1.5, zorder = 2)
#        ax6.scatter()
#        ax6.axvline(peaks_l[idx0], color = 'red', linestyle = '--', linewidth = 1.5, zorder = 2)
        ax6.scatter(self.best_lag, self.best_auto, s = 45., edgecolor = 'lime', marker = 's', facecolor = 'none', linewidths = 1.)
        ax6.plot(self.zoom_lag, self.zoom_auto, 'r-', zorder = 5, linewidth = 1.)
        ax6.set_title(r'$\rm ACF \,\, for \,\, determining \,\, \Delta\nu$')
        ax6.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
        ax6.set_xlim([min(self.lag), max(self.lag)])
        ax6.set_ylim([min(self.auto)-0.05*(max(self.auto)-min(self.auto)), max(self.auto)+0.1*(max(self.auto)-min(self.auto))])

        # dnu fit
        fit = gaussian(self.zoom_lag, self.gauss_2[0], self.gauss_2[1], self.gauss_2[2], self.gauss_2[3])
        idx = self.return_max(fit, index=True)
        plot_lower = min(self.zoom_auto)
        if min(fit) < plot_lower:
            plot_lower = min(fit)
        plot_upper = max(self.zoom_auto)
        if max(fit) > plot_upper:
            plot_upper = max(fit)
        ax7 = fig.add_subplot(3,3,7)
        ax7.plot(self.zoom_lag, self.zoom_auto, 'w-', zorder = 0, linewidth = 1.0)
        ax7.axvline(self.best_lag, color = 'red', linestyle = '--', linewidth = 1.5, zorder = 2)
        ax7.plot(self.zoom_lag, fit, color = 'lime', linewidth = 1.5)
        ax7.axvline([self.zoom_lag[idx]], color = 'lime', linestyle = '--', linewidth = 1.5)
        ax7.set_title(r'$\rm \Delta\nu \,\, fit$')
        ax7.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
        ax7.annotate(r'$\Delta\nu = %.2f$'%self.zoom_lag[idx], xy = (0.025, 0.85), xycoords = "axes fraction", fontsize = 18, color = 'lime')
        ax7.set_xlim([min(self.zoom_lag), max(self.zoom_lag)])
        ax7.set_ylim([plot_lower-0.05*(plot_upper-plot_lower), plot_upper+0.1*(plot_upper-plot_lower)])

        # echelle diagram
        ax8 = fig.add_subplot(3,3,8)
        ax8.imshow(self.ech, extent=self.extent, interpolation = 'bilinear', aspect = 'auto', origin = 'lower', cmap = 'jet', norm = LogNorm(vmin = np.nanmedian(self.ech_copy), vmax = np.nanmax(self.ech_copy)))
#        ax8.imshow(self.ech, extent = self.extent, interpolation = 'kaiser', aspect = 'auto', origin = 'lower', cmap = 'jet')
        ax8.axvline([self.dnu], color = 'white', linestyle = '--', linewidth = 1., dashes=(5,5))
        ax8.set_title(r'$\rm \grave{E}chelle \,\, diagram$')
        ax8.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$'%self.dnu)
        ax8.set_ylabel(r'$\rm \nu \,\, [\mu Hz]$')
        ax8.set_xlim([0., 2.*self.dnu])
        ax8.set_ylim([self.maxpower[0], self.maxpower[1]])

        ax9 = fig.add_subplot(3,3,9)
        ax9.plot(self.xax, self.yax, color = 'white', linestyle = '-', linewidth = 0.75)
        ax9.set_title(r'$\rm Collapsed \,\, \grave{e}chelle \,\, diagram$')
        ax9.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$'%self.dnu)
        ax9.set_ylabel(r'$\rm Collapsed \,\, power$')
        ax9.set_xlim([0., 2.*self.dnu])
        ax9.set_ylim([min(self.yax)-0.025*(max(self.yax)-min(self.yax)), max(self.yax)+0.05*(max(self.yax)-min(self.yax))])

        plt.tight_layout()
        if self.fitbg['save']:
            plt.savefig(self.params[self.target]['path']+'%d_fitbg.png'%self.target, dpi = 300)
        if self.show_plots:
            plt.show() 
        plt.close()

    def plot_mc(self, final_pars):

        plt.figure(figsize = (12,8))

        titles = [r'$\rm Smoothed \,\, \nu_{max} \,\, [\mu Hz]$', r'$\rm Smoothed \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$', r'$\rm Gaussian \,\, \nu_{max} \,\, [\mu Hz]$', r'$\rm Gaussian \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$', r'$\rm Gaussian \,\, FWHM \,\, [\mu Hz]$', r'$\rm \Delta\nu \,\, [\mu Hz]$']

        for i in range(6):
        
            ax = plt.subplot(2,3,i+1)
            ax.hist(final_pars[:,2*self.nlaws+(i+1)], color = 'cyan', histtype = 'step', lw = 2.5, facecolor = '0.75')
            ax.set_title(titles[i])

        plt.tight_layout()
        if self.findex['save']:
            plt.savefig(self.params[self.target]['path']+'%d_mc.png'%self.target, dpi = 300)
        if self.show_plots:
            plt.show()
        plt.close()

##########################################################################################
#                                                                                        #
#                                       FUNCTIONS                                        #
#                                                                                        #
##########################################################################################

    def get_ridges(self, start=0.):

        if self.i == 0:
            self.get_best_dnu()

        ech, gridx, gridy, extent = self.echelle()
        N, M = ech.shape[0], ech.shape[1]
        ech_copy = np.array(list(ech.reshape(-1)))

        n = int(np.ceil(self.dnu/self.resolution))
        xax = np.zeros(n)
        yax = np.zeros(n)
        modx = self.original_freq%self.dnu

        for k in range(n):
            use = np.where((modx >= start)&(modx < start+self.resolution))[0]
            if len(use) == 0:
                continue
            xax[k] = np.median(modx[use])
            yax[k] = np.sum(self.bg_corr[use])
            start += self.resolution

        xax = np.array(list(xax)+list(xax+self.dnu))
        yax = np.array(list(yax)+list(yax))-min(yax)

        if self.fitbg['clip']:
            if self.fitbg['clip_value'] != 0.:
                cut = self.fitbg['clip_value']
            else:
                cut = np.nanmax(ech_copy)-(np.nanmax(ech_copy)-np.nanmedian(ech_copy))/2.
            ech_copy[ech_copy > cut] = cut

        if self.i == 0:
            self.ech_copy = ech_copy
            self.ech = ech_copy.reshape((N,M))
            self.extent = extent
            self.xax = xax
            self.yax = yax

    def get_best_dnu(self):

        dnus = np.arange(self.dnu-0.05*self.dnu, self.dnu+0.05*self.dnu, 0.01)
        difference = np.zeros_like(dnus)

        for x, d in enumerate(dnus):
            start = 0.
            n = int(np.ceil(d/self.resolution))
            xax = np.zeros(n)
            yax = np.zeros(n)
            modx = self.original_freq%d

            for k in range(n):
                use = np.where((modx >= start)&(modx < start+self.resolution))[0]
                if len(use) == 0:
                    continue
                xax[k] = np.median(modx[use])
                yax[k] = np.sum(self.bg_corr[use])
                start += self.resolution

            difference[x] = np.max(yax)-np.mean(yax)

        idx = self.return_max(difference, index=True)
        self.dnu = dnus[idx]

    def echelle(self, n_across=20, startx=0.):

        if self.fitbg['ech_smooth']:
            boxkernel = Box1DKernel(int(np.ceil(self.fitbg['ech_smooth']/self.resolution)))
            smooth_y = convolve(self.bg_corr, boxkernel)

        nox = n_across
        noy = int(np.ceil((max(self.original_freq)-min(self.original_freq))/self.dnu))

        if nox > 2 and noy > 5:
            xax = np.arange(0., self.dnu+(self.dnu/n_across)/2., self.dnu/n_across)
            yax = np.arange(min(self.original_freq), max(self.original_freq), self.dnu)

            arr = np.zeros((len(xax),len(yax)))
            gridx = np.zeros(len(xax))
            gridy = np.zeros(len(yax))

            modx = self.original_freq%self.dnu
            starty = min(self.original_freq)

            for ii in range(len(gridx)):
                for jj in range(len(gridy)):
                    use = np.where((modx >= startx)&(modx < startx+self.dnu/n_across)&(self.original_freq >= starty)&(self.original_freq < starty+self.dnu))[0]
                    if len(use) == 0:
                        arr[ii,jj] = np.nan
                    else:
                        arr[ii,jj] = np.sum(self.bg_corr[use])
                    gridy[jj] = starty + self.dnu/2.
                    starty += self.dnu
                gridx[ii] = startx + self.dnu/n_across/2.
                starty = min(self.original_freq)
                startx += self.dnu/n_across
            smoothed = arr
            dim = smoothed.shape

            smoothed_2 = np.zeros((2*dim[0],dim[1]))
            smoothed_2[0:dim[0],:] = smoothed
            smoothed_2[dim[0]:(2*dim[0]),:] = smoothed
            smoothed = np.swapaxes(smoothed_2, 0, 1)
            extent = [min(gridx)-self.dnu/n_across/2., 2*max(gridx)+self.dnu/n_across/2., min(gridy)-self.dnu/2., max(gridy)+self.dnu/2.]
        
            return smoothed, np.array(list(gridx)+list(gridx+self.dnu)), gridy, extent

    def gaussian_bounds(self, x, y, best_x=None, sigma=None):

        if sigma is None:
            sigma = (max(x)-min(x))/8./np.sqrt(8.*np.log(2.))
        bb = []
        b = np.zeros((2,4)).tolist()
        b[1][0] = np.inf
        b[1][1] = 2.*np.max(y)
        if not int(np.max(y)):
            b[1][1] = np.inf
        if best_x is not None:
            b[0][2] = best_x - 0.001*best_x
            b[1][2] = best_x + 0.001*best_x
        else:
            b[0][2] = np.min(x)
            b[1][2] = np.max(x)
        b[0][3] = sigma
        b[1][3] = np.max(x)-np.min(x)
        bb.append(tuple(b))
        return bb

    def max_elements(self, x, y, limit = [False, None]):

        tempy = y[:]
        indices = []
        while len(indices) < self.fitbg['n_peaks']:
            new_max = max(tempy)
            idx = tempy.index(new_max)
            add = True
            if indices != [] and limit[0]:
                for index in indices:
                    if np.absolute((index-idx)*self.resolution) < limit[1]:
                        add = False
                        break
            if add:
                indices.append(idx)
            tempy[idx] = 0.

        x = np.array(x)
        y = np.array(y)

        peaks_x = x[indices]
        peaks_y = y[indices]
        ss = np.argsort(peaks_x)

        return peaks_x[ss], peaks_y[ss]

    def return_max(self, array, index=False, dnu=False):

        if dnu:
            exp_dnu = 0.22*(self.numax**0.797)
            lst = list(np.absolute(np.copy(array)-exp_dnu))
            idx = lst.index(min(lst))
        else:
            lst = list(array)
            idx = lst.index(max(lst))

        if index:
            return idx
        else:
            return lst[idx]

    def corr(self, frequency, power):

        lag = np.arange(0., len(power))*self.resolution
        auto = np.real(np.fft.fft(np.fft.ifft(power)*np.conj(np.fft.ifft(power))))

        lower_limit = self.dnu/4.
        upper_limit = 2.*self.dnu + self.dnu/4.

        mask = np.ma.getmask(np.ma.masked_inside(lag, lower_limit, upper_limit))
        lag = lag[mask]
        auto = auto[mask]
        auto -= min(auto)
        auto /= max(auto)
    
        return lag, auto

##########################################################################################
#                                                                                        #
#                                        INITIATE                                        #
#                                                                                        #
##########################################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'This is a script to run the new version of the SYD PYpeline')
    parser.add_argument('-e', '-x', '--e', '--x', '-ex', '--ex', '-excess', '--excess', help = 'Use this to turn the find excess function off', dest = 'ex', action = 'store_false')
    parser.add_argument('-b', '--b', '-bg', '--bg', '-background', '--background', help = 'Use this to disable the background fitting process (although not highly recommended)', dest = 'bg', action = 'store_false')
    parser.add_argument('-v', '--v', '-verbose', '--verbose', help = 'Turn on verbose', dest = 'verbose', action = 'store_false')
    parser.add_argument('-s', '--s', '-show', '--show', help = 'Show plots', dest = 'show', action = 'store_false')
    parser.add_argument('-i', '--i', '-ignore', '--ignore', help = 'Ignore multiple target output supression', dest = 'ignore', action = 'store_true')


    args = parser.parse_args()
    ex = args.ex
    bg = args.bg
    verbose = args.verbose
    show = args.show
    ignore = args.ignore

    main(findex = ex, fitbg = bg, verbose = verbose, show_plots = show, ignore = ignore)