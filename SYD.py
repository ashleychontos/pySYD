import os
import glob
import argparse
import subprocess
import matplotlib
import numpy as np
import pandas as pd
import time as clock
from functions import *
import multiprocessing as mp
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
import pdb


def main(args, parallel=False, nthreads=None):

    args = get_info(args)
    set_plot_params()

    for target in args.params['todo']:
        args.target = target
        Target(args)

    if args.verbose:
        print('Combining results into single csv file.')
        print()
    subprocess.call(['python scrape_output.py'], shell=True)

    return

class Target:
    
    def __init__(self, args):
        self.target = args.target
        self.params = args.params
        self.findex = args.findex
        self.fitbg = args.fitbg
        self.verbose = args.verbose
        self.show_plots = args.show
        self.ignore = args.ignore
        self.keplercorr = args.keplercorr
        self.filter = args.filter
        if len(self.params['todo']) > 1 and not self.ignore:
            self.verbose = False
            self.show_plots = False
        self.run_syd()

    def run_syd(self):
        if self.load_data():
            if self.findex['do']:
                self.find_excess()
            if self.fitbg['do']:
                self.fit_background()

##########################################################################################
#                                                                                        #
#                               READING/WRITING TO/FROM FILES                            #
#                                                                                        #
##########################################################################################

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
                if self.cadence/60. < 10.:
                    self.short_cadence = True
                else:
                    self.short_cadence = False
                self.nyquist = 10**6/(2.*self.cadence)
                if self.verbose:
                    print('# LIGHT CURVE: %d lines of data read'%len(self.time))
                if self.short_cadence:
                    self.fitbg['smooth_ps'] = 2.5
            # load power spectrum
            if not os.path.exists(self.params['path']+'%d_PS.txt'%self.target):
                if self.verbose:
                    print('Error: %s%d_PS.txt not found'%(self.params['path'], self.target))
                return False
            else:
                self.get_file(self.params['path']+'%d_PS.txt'%self.target)
                self.frequency = np.copy(self.x)
                self.power = np.copy(self.y)
                if self.keplercorr == True:
                    self.remove_artefact(self.frequency,self.power)
                    self.power = np.copy(self.y)
                    if self.verbose:
                        print('## Removing Kepler artefacts ##')
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
            # create critically sampled PS
            if self.oversample != 1:
                self.freq = np.copy(self.frequency)
                self.pow = np.copy(self.power)
                self.frequency = np.array(self.frequency[self.oversample-1::self.oversample])
                self.power = np.array(self.power[self.oversample-1::self.oversample])
            else:
                self.freq = np.copy(self.frequency)
                self.pow = np.copy(self.power)
                self.frequency = np.copy(self.frequency)
                self.power = np.copy(self.power)
            if hasattr(self, 'findex'):
                if self.findex['do']:
                    mask = np.ones_like(self.freq, dtype=bool)
                    if self.params[self.target]['lowerx'] is not None:
                        mask *= np.ma.getmask(np.ma.masked_greater_equal(self.freq, self.params[self.target]['lowerx']))
                    else:
                        mask *= np.ma.getmask(np.ma.masked_greater_equal(self.freq, self.findex['lower']))
                    if self.params[self.target]['upperx'] is not None:
                        mask *= np.ma.getmask(np.ma.masked_less_equal(self.freq, self.params[self.target]['upperx']))
                    else:
                        mask *= np.ma.getmask(np.ma.masked_less_equal(self.freq, self.findex['upper']))
                    self.freq = self.freq[mask]
                    self.pow = self.pow[mask]
            return True
        else:
            print('Error: data not found for target %d'%self.target)
            return False

    def get_file(self, path):

        f = open(path, "r")
        lines = f.readlines()
        f.close()

        self.x = np.array([float(line.strip().split()[0]) for line in lines])
        self.y = np.array([float(line.strip().split()[1]) for line in lines])

    def save(self):
        df = pd.DataFrame(self.final_pars)
        self.df = df.copy()
        if self.fitbg['num_mc_iter'] > 1:
            for column in self.df.columns.values.tolist():
                self.df['%s_err'%column] = np.array([mad_std(self.df[column].values)]*len(self.df))
        new_df = pd.DataFrame(columns=['parameter','value','uncertainty'])
        for c, col in enumerate(df.columns.values.tolist()):
            new_df.loc[c,'parameter'] = col
            new_df.loc[c,'value'] = self.final_pars[col][0]
            if self.fitbg['num_mc_iter'] > 1:
                new_df.loc[c,'uncertainty'] = mad_std(self.final_pars[col])
            else:
                new_df.loc[c,'uncertainty'] = '--'
        new_df.to_csv(self.params[self.target]['path']+'%d_globalpars.csv'%self.target,index=False)
        if self.fitbg['samples']:
            self.df.to_csv(self.params[self.target]['path']+'%d_globalpars_all.csv'%self.target,index=False)

    def check(self):
        # SYD needs some prior knowledge about numax to work well 
        # (either from findex module or from star info csv)
        if 'numax' not in self.params[self.target].keys():
            print("""WARNING: Suggested use of this pipeline requires either
                         stellar properties to estimate a numax or running the entire
                         pipeline from scratch (i.e. find_excess) first to
                         statistically determine a starting point for nuMax.""")
            return False
        else:
            return True
 

    def get_initial_guesses(self):
        # check whether output from findex module exists; if yes, let that override star info guesses
        if glob.glob(self.params[self.target]['path']+'%d_findex.csv'%self.target) != []:
            df = pd.read_csv(self.params[self.target]['path']+'%d_findex.csv'%self.target)
            for col in ['numax', 'dnu', 'snr']:
            		  self.params[self.target][col] = df.loc[0,col]
        else:
            # if no output from findex module exists, assume SNR is high enough to run fit_background
            self.params[self.target]['snr']=10.
        # mask power spectrum for fitbg module based on estimated/fitted numax
        mask = np.ones_like(self.frequency, dtype=bool)
        if self.params[self.target]['lowerb'] is not None:
            mask *= np.ma.getmask(np.ma.masked_greater_equal(self.frequency, self.params[self.target]['lowerb']))
            if self.params[self.target]['upperb'] is not None:
                mask *= np.ma.getmask(np.ma.masked_less_equal(self.frequency, self.params[self.target]['upperb']))
            else:
                mask *= np.ma.getmask(np.ma.masked_less_equal(self.frequency, self.nyquist))
        else:
            if self.params[self.target]['numax'] > 300.:
                mask = np.ma.getmask(np.ma.masked_inside(self.frequency, 100., self.nyquist))
            else:
                mask = np.ma.getmask(np.ma.masked_inside(self.frequency, 1., 500.))
        self.frequency = self.frequency[mask]
        self.power = self.power[mask]
        self.width = self.params['width_sun']*(self.params[self.target]['numax']/self.params['numax_sun'])
        self.times = self.width/self.params[self.target]['dnu']
        # arbitrary snr cut for leaving region out of background fit, ***statistically validate later?
        if self.fitbg['lower_numax'] is not None:
            self.maxpower = [self.fitbg['lower_numax'], self.fitbg['upper_numax']]
        else:
            if self.params[self.target]['snr'] < 2.:
                self.maxpower = [self.params[self.target]['numax']-self.width/2., self.params[self.target]['numax']+self.width/2.]
            else:
                self.maxpower = [self.params[self.target]['numax']-self.times*self.params[self.target]['dnu'],self.params[self.target]['numax']+self.times*self.params[self.target]['dnu']]
        # adjust the lower frequency limit given numax
        if self.params[self.target]['numax'] > 300.:
           	self.frequency=self.frequency[self.frequency>100.]
           	self.power=self.power[self.frequency>100.]
           	self.fitbg['lower']=100.
        # use scaling relation from sun to get starting points
        scale = self.params['numax_sun']/((self.maxpower[1]+self.maxpower[0])/2.)
        taus = np.array(self.params['tau_sun'])*scale
        b = 2.*np.pi*(taus*1e-6)
        mnu = (1./taus)*1e5
        self.b = b[mnu>=min(self.frequency)]
        self.mnu = mnu[mnu>=min(self.frequency)]
        self.nlaws = len(self.mnu)
        self.mnu_orig = np.copy(self.mnu)
        self.b_orig = np.copy(self.b)

    def write_excess(self, results):

        variables = ['target', 'numax', 'dnu', 'snr']
        ascii.write(np.array(results),self.params[self.target]['path']+'%d_findex.csv'%self.target,names=variables,delimiter=',',overwrite=True)
    
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

        N = int(self.findex['n_trials']+3)
        if N%3 == 0:
            self.nrows = (N-1)//3
        else:
            self.nrows = N//3

        if self.findex['binning'] is not None:
            bin_freq, bin_pow = bin_data(self.freq, self.pow, self.findex)
            self.bin_freq = bin_freq
            self.bin_pow = bin_pow
            if self.verbose:
                print('binned to %d datapoints'%len(self.bin_freq))

            boxsize = int(np.ceil(float(self.findex['smooth_width'])/(bin_freq[1]-bin_freq[0])))
            sp = convolve(bin_pow, Box1DKernel(boxsize))
            smooth_freq = bin_freq[int(boxsize/2):-int(boxsize/2)]
            smooth_pow = sp[int(boxsize/2):-int(boxsize/2)]

            s = InterpolatedUnivariateSpline(smooth_freq, smooth_pow, k=1)
            self.interp_pow = s(self.freq)
            self.bgcorr_pow = self.pow/self.interp_pow

            if not self.short_cadence:
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

                    lag = np.arange(0., len(p))*self.resolution
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
                    best_vars, covar = curve_fit(gaussian, self.md[i], self.cumsum[i], p0 = [np.mean(self.cumsum[i]), 1.0-np.mean(self.cumsum[i]), self.md[i][idx], self.params['width_sun']*(self.md[i][idx]/self.params['numax_sun'])])
                except:
                    results.append([self.target, np.nan, np.nan, -np.inf])
                else:
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
            self.plot_findex()


##########################################################################################
#                                                                                        #
#                                  FIT BACKGROUND ROUTINE                                #
#                                                                                        #
##########################################################################################

    def fit_background(self, result=''):

        if self.check():
            self.get_initial_guesses()
            self.final_pars = final_pars={'numax_smooth':[],'amp_smooth':[],'numax_gaussian':[],'amp_gaussian':[],'fwhm_gaussian':[],'dnu':[],'wn':[]}
            # sampling process
            i = 0
            while i < self.fitbg['num_mc_iter']:
                self.i = i
                if self.i == 0:
                    # record original PS information for plotting
                    self.random_pow = np.copy(self.power)
                    bin_freq, bin_pow, bin_err = mean_smooth_ind(self.frequency, self.random_pow, self.fitbg['ind_width'])
                    if self.verbose:
                        print('-------------------------------------------------')
                        print('binned to %d data points'%len(bin_freq))
                else:
                    # randomize power spectrum to get uncertainty on measured values
                    self.random_pow = (np.random.chisquare(2, len(self.frequency))*self.power)/2.
                    bin_freq, bin_pow, bin_err = mean_smooth_ind(self.frequency, self.random_pow, self.fitbg['ind_width'])
                self.bin_freq = bin_freq[~((bin_freq>self.maxpower[0])&(bin_freq<self.maxpower[1]))]
                self.bin_pow = bin_pow[~((bin_freq>self.maxpower[0])&(bin_freq<self.maxpower[1]))]
                self.bin_err = bin_err[~((bin_freq>self.maxpower[0])&(bin_freq<self.maxpower[1]))]
                # estimate white noise level
                self.get_white_noise()

                # exclude region with power excess and smooth to estimate red/white noise components
                boxkernel = Box1DKernel(int(np.ceil(self.fitbg['box_filter']/self.resolution)))
                self.params[self.target]['mask'] = (self.frequency>=self.maxpower[0])&(self.frequency<=self.maxpower[1])
                self.smooth_pow = convolve(self.random_pow[~self.params[self.target]['mask']], boxkernel)
                
                # temporary array for inputs into model optimization (changes with each iteration) 
                pars = np.zeros((self.nlaws*2+1))
                # estimate amplitude for each harvey component
                for n, nu in enumerate(self.mnu):
                    diff = list(np.absolute(self.frequency-nu))
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

                # smooth power spectrum - ONLY for plotting purposes, not used in subsequent analyses
                self.smooth_power = convolve(self.power, Box1DKernel(int(np.ceil(self.fitbg['box_filter']/self.resolution))))
                # if optimization does not converge, the rest of the code will not run
                if self.get_red_noise():
                    continue
                # save final values for Harvey laws from model fit
                for n in range(self.nlaws):
                    self.final_pars['a_%d'%(n+1)].append(self.pars[2*n])
                    self.final_pars['b_%d'%(n+1)].append(self.pars[2*n+1])
                self.final_pars['wn'].append(self.pars[2*self.nlaws])

                # estimate numax by 1) smoothing power and 2) fitting Gaussian
                self.get_numax_smooth()
                if list(self.region_freq) != []:
                    self.get_numax_gaussian()
                self.bg_corr = self.random_pow/harvey(self.frequency, self.pars, total=True)

                # optional smoothing of PS to remove fine structure before computing ACF
                if self.fitbg['smooth_ps'] is not None:
                    boxkernel = Box1DKernel(int(np.ceil(self.fitbg['smooth_ps']/self.resolution)))
                    self.bg_corr_smooth = convolve(self.bg_corr, boxkernel)
                else:
                    self.bg_corr_smooth = np.copy(self.bg_corr)

                # calculate ACF using ffts (default) and estimate large frequency separation
                dnu = self.get_frequency_spacing()
                self.final_pars['dnu'].append(dnu)
                if self.i == 0:
                    self.get_ridges()
                    self.plot_fitbg()
                i+=1

            self.save()
            if self.fitbg['num_mc_iter'] > 1:
                self.plot_mc() 
                if self.verbose:
                    print('numax (smoothed): %.2f +/- %.2f muHz'%(self.final_pars['numax_smooth'][0],mad_std(self.final_pars['numax_smooth'])))
                    print('maxamp (smoothed): %.2f +/- %.2f ppm^2/muHz'%(self.final_pars['amp_smooth'][0],mad_std(self.final_pars['amp_smooth'])))
                    print('numax (gaussian): %.2f +/- %.2f muHz'%(self.final_pars['numax_gaussian'][0],mad_std(self.final_pars['numax_gaussian'])))
                    print('maxamp (gaussian): %.2f +/- %.2f ppm^2/muHz'%(self.final_pars['amp_gaussian'][0],mad_std(self.final_pars['amp_gaussian'])))
                    print('fwhm (gaussian): %.2f +/- %.2f muHz'%(self.final_pars['fwhm_gaussian'][0],mad_std(self.final_pars['fwhm_gaussian'])))
                    print('dnu: %.2f +/- %.2f muHz'%(self.final_pars['dnu'][0],mad_std(self.final_pars['dnu'])))
                    print('-------------------------------------------------')
                    print()
            else:
                if self.verbose:
                    print('numax (smoothed): %.2f muHz'%(self.final_pars['numax_smooth'][0]))
                    print('maxamp (smoothed): %.2f ppm^2/muHz'%(self.final_pars['amp_smooth'][0]))
                    print('numax (gaussian): %.2f muHz'%(self.final_pars['numax_gaussian'][0]))
                    print('maxamp (gaussian): %.2f ppm^2/muHz'%(self.final_pars['amp_gaussian'][0]))
                    print('fwhm (gaussian): %.2f muHz'%(self.final_pars['fwhm_gaussian'][0]))
                    print('dnu: %.2f'%(self.final_pars['dnu'][0]))
                    print('-------------------------------------------------')
                    print()  

##########################################################################################
#                                                                                        #
#                                SYD-RELATED  FUNCTIONS                                  #
#                                                                                        #
##########################################################################################

    def remove_artefact(self, frequency, power, lc=29.4244*60*1e-6):
        '''
        Removes SC artefacts in Kepler power spectra by replacing them with noise (using linear interpolation)
        following an exponential distribution; known artefacts are:
        1) 1./LC harmonics
        2) unknown artefacts at high frequencies (>5000 muHz)
        3) excess power between 250-400 muHz (in Q0 and Q3 data only??)
        @input: frequency and power
        @output: frequency and power where artefact frequencies are filled with noise
        '''
        f,a=self.frequency,self.power
        oversample = int(round((1./((max(self.time)-min(self.time))*0.0864))/(self.frequency[1]-self.frequency[0])))
        resolution = (self.frequency[1]-self.frequency[0])*oversample

        lcp = 1./lc #LC period in Msec -> 1/LC ~muHz
        art = (1.+np.arange(14))*lcp

        un1=[4530.,5011.,5097.,5575.,7020.,7440.,7864.] #lower limit of artefact
        un2=[4534.,5020.,5099.,5585.,7030.,7450.,7867.] #upper limit of artefact 
        noisefl  = np.mean(a[(f>=max(f)-100.)&(f<=max(f)-50.)])  #estimate white noise 

        # Routine 1: remove 1/LC artefacts by subtracting +/- 5 muHz given each artefact
        for i in range(len(art)):
            if art[i] < np.max(f):
                use=np.where((f>art[i]-5.*resolution)&(f<art[i]+5.*resolution))[0]
                if use[0] != -1:
                    a[use] = noisefl*np.random.chisquare(2,len(use))/2.

        # Routine 2: remove artefacts as identified in un1 & un2
        for i in range(0,len(un1)):
            if un1[i] < np.max(f):
                use=np.where( (f > un1[i]) & (f < un2[i]) )[0]
                if use[0] != -1:
                    a[use] = noisefl*np.random.chisquare(2,len(use))/2.

        # Routine 3: remove two wider artefacts as identified in un1 & un2
        un1=[240.,500.]
        un2=[380.,530.]

        for i in range(0,len(un1)):
            # un1[i] : freq where artefact starts
            # un2[i] : freq where artefact ends
            # un_lower : initial freq to start fitting routine (aka un1[i]-20)
            # un_upper : final freq to end fitting routine     (aka un2[i]+20)
            flower,fupper=un1[i]-20,un2[i]+20   
            usenoise = np.where(((f>=flower)&(f<=un1[i]))|\
                                ((f>=un2[i])&(f<=fupper)))[0]
            m,b =np.polyfit(f[usenoise],a[usenoise],1) # coeffs for linear fit
            use =np.where((f>=un1[i])&(f<=un2[i]))[0]       #index of artefact frequencies (ie. 240-380 muHz)
            a[use]=(f[use]*m+b)*np.random.chisquare(2,len(use))/2.  #fill artefact frequencies with noise
        self.y=a

    def get_white_noise(self):

        if self.nyquist < 400.:
            mask = (self.frequency>200.)&(self.frequency<270.)
            self.noise = np.mean(self.random_pow[mask])
        elif self.nyquist > 400. and self.nyquist < 5000.:
            mask = (self.frequency>4000.)&(self.frequency<4167.)
            self.noise = np.mean(self.random_pow[mask])
        elif self.nyquist > 5000. and self.nyquist < 9000.:
            mask = (self.frequency>8000.)&(self.frequency<8200.)
            self.noise = np.mean(self.random_pow[mask])
        else:
            mask = (self.frequency>(max(self.frequency)-0.1*mask(self.frequency)))&(self.frequency<max(self.frequency))
            self.noise = np.mean(self.random_pow[mask])

    def get_red_noise(self, names=['one', 'one', 'two', 'two', 'three', 'three', 'four', 'four', 'five', 'five', 'six', 'six'],  
                      bounds=[], reduced_chi2=[], paras=[], a=[]):
        # get best fit model 
        if self.i == 0:
            reduced_chi2=[]
            bounds=[]
            a=[]
            paras=[]
            for n in range(self.nlaws):
                a.append(self.pars[2*n])
            self.a_orig=np.array(a)
            if self.verbose:
                print('Comparing %d different models:'%(self.nlaws*2))
            for law in range(self.nlaws):
                bb = np.zeros((2,2*(law+1)+1)).tolist()
                for z in range(2*(law+1)):
                    bb[0][z] = -np.inf
                    bb[1][z] = np.inf
                bb[0][-1] = 0.
                bb[1][-1] = np.inf
                bounds.append(tuple(bb))
            dict1 = dict(zip(np.arange(2*self.nlaws),names[:2*self.nlaws]))
            for t in range(2*self.nlaws):
                if t%2 == 0:  
                    if self.verbose:
                        print('%d: %s harvey model w/ white noise free parameter'%(t+1, dict1[t]))
                    delta = 2*(self.nlaws-(t//2+1))
                    pams = list(self.pars[:(-delta-1)])
                    pams.append(self.pars[-1])
                    try:
                        pp, cv = curve_fit(self.fitbg['functions'][t//2+1], self.bin_freq, self.bin_pow, p0=pams, sigma=self.bin_err)
                    except RuntimeError:
                        paras.append([])
                        reduced_chi2.append(np.inf)
                    else:
                        paras.append(pp)
                        chi, p = chisquare(f_obs=self.random_pow[~self.params[self.target]['mask']], f_exp=harvey(self.frequency[~self.params[self.target]['mask']], pp, total=True))
                        reduced_chi2.append(chi/(len(self.frequency[~self.params[self.target]['mask']])-len(pams)))
                else:
                    if self.verbose:
                        print('%d: %s harvey model w/ white noise fixed'%(t+1, dict1[t]))
                    delta = 2*(self.nlaws-(t//2+1))
                    pams = list(self.pars[:(-delta-1)])
                    pams.append(self.pars[-1])
                    try:
                        pp, cv = curve_fit(self.fitbg['functions'][t//2+1], self.bin_freq, self.bin_pow, p0=pams, sigma=self.bin_err, bounds=bounds[t//2])
                    except RuntimeError:
                        paras.append([])
                        reduced_chi2.append(np.inf)
                    else:
                        paras.append(pp)
                        chi, p = chisquare(f_obs=self.random_pow[~self.params[self.target]['mask']], f_exp=harvey(self.frequency[~self.params[self.target]['mask']], pp, total=True))
                        reduced_chi2.append(chi/(len(self.frequency[~self.params[self.target]['mask']])-len(pams)+1))
            if np.isfinite(min(reduced_chi2)):
                model = reduced_chi2.index(min(reduced_chi2))+1
                if self.nlaws != (((model-1)//2)+1):
                    self.nlaws = ((model-1)//2)+1
                    self.mnu = self.mnu[:(self.nlaws)]
                    self.b = self.b[:(self.nlaws)]
                if self.verbose:
                    print('Based on reduced chi-squared statistic: model %d'%model)
                self.bounds = bounds[self.nlaws-1]
                self.pars = paras[model-1] 
                self.exp_numax = self.params[self.target]['numax']
                self.exp_dnu = self.params[self.target]['dnu']
                self.sm_par = 4.*(self.exp_numax/self.params['numax_sun'])**0.2
                if self.sm_par < 1.:
                    self.sm_par = 1.
                for n in range(self.nlaws):
                    self.final_pars['a_%d'%(n+1)]=[]
                    self.final_pars['b_%d'%(n+1)]=[]
                again = False
            else:
                again = True
        else:
            try:
                pars, cv = curve_fit(self.fitbg['functions'][self.nlaws], self.bin_freq, self.bin_pow, p0=self.pars, sigma=self.bin_err, bounds=self.bounds)
            except RuntimeError:
                again = True
            else:
                self.pars = pars
                self.sm_par = 4.*(self.exp_numax/self.params['numax_sun'])**0.2
                if self.sm_par < 1.:
                    self.sm_par = 1.
                again = False
        return again

    def get_numax_smooth(self):
        sig = (self.sm_par*(self.exp_dnu/self.resolution))/np.sqrt(8.*np.log(2.))
        pssm = convolve_fft(np.copy(self.random_pow), Gaussian1DKernel(int(sig)))
        model = harvey(self.frequency, self.pars, total=True)
        inner_freq = list(self.frequency[self.params[self.target]['mask']])
        inner_obs = list(pssm[self.params[self.target]['mask']])
        outer_freq = list(self.frequency[~self.params[self.target]['mask']])
        outer_mod = list(model[~self.params[self.target]['mask']])
        if self.fitbg['slope']:
            # correct for edge effects and residual slope in Gaussian fit
            inner_mod = model[self.params[self.target]['mask']]
            delta_y = inner_obs[-1]-inner_obs[0]
            delta_x = inner_freq[-1]-inner_freq[0]
            slope = delta_y/delta_x
            b = slope*(-1.*inner_freq[0])+inner_obs[0]
            corrected = np.array([inner_freq[z]*slope+b for z in range(len(inner_freq))])
            corr_pssm = [inner_obs[z]-corrected[z]+inner_mod[z] for z in range(len(inner_obs))]
            final_y = np.array(corr_pssm+outer_mod)
        else:
            outer_freq = list(self.frequency[~self.params[self.target]['mask']])
            outer_mod = list(model[~self.params[self.target]['mask']])
            final_y = np.array(inner_obs+outer_mod)
        final_x = np.array(inner_freq+outer_freq)
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
        self.guesses = [0., max(self.region_pow), self.region_freq[idx], (max(self.region_freq)-min(self.region_freq))/np.sqrt(8.*np.log(2.))]

    def get_numax_gaussian(self):
        bb = gaussian_bounds(self.region_freq, self.region_pow)
        p_gauss1, p_cov = curve_fit(gaussian, self.region_freq, self.region_pow, p0=self.guesses, bounds=bb[0])
        # create array with finer resolution for purposes of quantifying uncertainty
        new_freq = np.linspace(min(self.region_freq),max(self.region_freq),10000)
        numax_fit = list(gaussian(new_freq, p_gauss1[0], p_gauss1[1], p_gauss1[2], p_gauss1[3]))
        d=numax_fit.index(max(numax_fit))
        self.final_pars['numax_gaussian'].append(new_freq[d])
        self.final_pars['amp_gaussian'].append(p_gauss1[1])
        self.final_pars['fwhm_gaussian'].append(p_gauss1[3])
        if self.i == 0:
        	# dnu_exp should always be fixed to the input numax
            #self.exp_numax = new_freq[d]
            #self.exp_dnu = 0.22*(self.exp_numax**0.797)
            #self.width = self.params['width_sun']*(self.exp_numax/self.params['numax_sun'])/2.
            self.new_freq = np.copy(new_freq)
            self.numax_fit = np.array(numax_fit)

    def get_frequency_spacing(self):
        self.compute_acf()
        dnu = self.estimate_dnu()
        return dnu

    def compute_acf(self, fft=True):
        power = self.bg_corr_smooth[(self.frequency >= self.exp_numax-self.width)&(self.frequency <= self.exp_numax+self.width)]
        lag = np.arange(0., len(power))*self.resolution
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

    def estimate_dnu(self):
        if self.i == 0:
            # get peaks from ACF
            peaks_l, peaks_a = self.max_elements(self.lag, self.auto)
            # pick the peak closest to the modeled numax
            idx = return_max(peaks_l, index=True, dnu=True, exp_dnu=self.exp_dnu)
            bb = gaussian_bounds(self.lag, self.auto, best_x=peaks_l[idx], sigma=10**-2)
            guesses = [np.mean(self.auto), peaks_a[idx], peaks_l[idx], peaks_l[idx]*0.01*2.]
            p_gauss2, p_cov2 = curve_fit(gaussian, self.lag, self.auto, p0=guesses, bounds=bb[0])
            self.fitbg['acf_mask']=[peaks_l[idx]-5.*p_gauss2[3],peaks_l[idx]+5.*p_gauss2[3]]
        # do the same only with the single peak
        zoom_lag = self.lag[(self.lag>=self.fitbg['acf_mask'][0])&(self.lag<=self.fitbg['acf_mask'][1])]
        zoom_auto = self.auto[(self.lag>=self.fitbg['acf_mask'][0])&(self.lag<=self.fitbg['acf_mask'][1])]
        # get peaks from ACF
        peaks_l, peaks_a = self.max_elements(zoom_lag, zoom_auto)
        # pick the peak closest to the modeled numax
#        idx = return_max(peaks_l, index=True, dnu=True, exp_dnu=self.exp_dnu)
        best_lag = peaks_l[0]
        best_auto = peaks_a[0]
        bb = gaussian_bounds(zoom_lag, zoom_auto, best_x=best_lag, sigma=10**-2)
        guesses = [np.mean(zoom_auto), best_auto, best_lag, best_lag*0.01*2.]
        p_gauss3, p_cov3 = curve_fit(gaussian, zoom_lag, zoom_auto, p0=guesses, bounds=bb[0])
        # create array with finer resolution for purposes of quantifying dnu uncertainty
        new_lag = np.linspace(min(zoom_lag),max(zoom_lag),2000)
        dnu_fit = list(gaussian(new_lag, p_gauss3[0], p_gauss3[1], p_gauss3[2], p_gauss3[3]))
        d=dnu_fit.index(max(dnu_fit))
        if self.fitbg['force']:
            dnu = self.fitbg['guess']
        else:
            dnu = new_lag[d]
        if self.i == 0:
            peaks_l[idx] = np.nan
            peaks_a[idx] = np.nan
            self.peaks_l = peaks_l
            self.peaks_a = peaks_a
            self.best_lag = best_lag
            self.best_auto = best_auto
            self.zoom_lag = zoom_lag
            self.zoom_auto = zoom_auto
            self.obs_dnu = dnu
            self.new_lag = new_lag
            self.dnu_fit = dnu_fit
        return dnu

    def get_ridges(self, start=0.):

#        self.get_best_dnu()
        ech, gridx, gridy, extent = self.echelle()
        N, M = ech.shape[0], ech.shape[1]
        ech_copy = np.array(list(ech.reshape(-1)))

        n = int(np.ceil(self.obs_dnu/self.resolution))
        xax = np.zeros(n)
        yax = np.zeros(n)
        modx = self.frequency%self.obs_dnu
        for k in range(n):
            use = np.where((modx >= start)&(modx < start+self.resolution))[0]
            if len(use) == 0:
                continue
            xax[k] = np.median(modx[use])
            yax[k] = np.sum(self.bg_corr[use])
            start += self.resolution
        xax = np.array(list(xax)+list(xax+self.obs_dnu))
        yax = np.array(list(yax)+list(yax))-min(yax)
        mask = np.ma.getmask(np.ma.masked_where(yax==0.,yax))
        if self.fitbg['clip']:
            if self.fitbg['clip_value'] != 0.:
                cut = self.fitbg['clip_value']
            else:
                cut = np.nanmedian(ech_copy)+(3.*np.nanmedian(ech_copy))
            ech_copy[ech_copy > cut] = cut
        self.ech_copy = ech_copy
        self.ech = ech_copy.reshape((N,M))
        self.extent = extent
        self.xax = xax[~mask]
        self.yax = yax[~mask]

    def get_best_dnu(self):
        dnus = np.arange(self.obs_dnu-0.05*self.obs_dnu, self.obs_dnu+0.05*self.obs_dnu, 0.01)
        difference = np.zeros_like(dnus)
        for x, d in enumerate(dnus):
            start = 0.
            n = int(np.ceil(d/self.resolution))
            xax = np.zeros(n)
            yax = np.zeros(n)
            modx = self.frequency%d
            for k in range(n):
                use = np.where((modx >= start)&(modx < start+self.resolution))[0]
                if len(use) == 0:
                    continue
                xax[k] = np.median(modx[use])
                yax[k] = np.sum(self.bg_corr[use])
                start += self.resolution
            difference[x] = np.max(yax)-np.mean(yax)
        idx = return_max(difference, index=True)
        self.obs_dnu = dnus[idx]

    def echelle(self, n_across=50, startx=0.):

        if self.fitbg['ech_smooth']:
            boxkernel = Box1DKernel(int(np.ceil(self.fitbg['ech_filter']/self.resolution)))
            smooth_y = convolve(self.bg_corr, boxkernel)
        nox = n_across
        noy = int(np.ceil((max(self.frequency)-min(self.frequency))/self.obs_dnu))
        if nox > 2 and noy > 5:
            xax = np.arange(0., self.obs_dnu+(self.obs_dnu/n_across)/2., self.obs_dnu/n_across)
            yax = np.arange(min(self.frequency), max(self.frequency), self.obs_dnu)
            arr = np.zeros((len(xax),len(yax)))
            gridx = np.zeros(len(xax))
            gridy = np.zeros(len(yax))

            modx = self.frequency%self.obs_dnu
            starty = min(self.frequency)
            for ii in range(len(gridx)):
                for jj in range(len(gridy)):
                    use = np.where((modx >= startx)&(modx < startx+self.obs_dnu/n_across)&(self.frequency >= starty)&(self.frequency < starty+self.obs_dnu))[0]
                    if len(use) == 0:
                        arr[ii,jj] = np.nan
                    else:
                        arr[ii,jj] = np.sum(self.bg_corr[use])
                    gridy[jj] = starty + self.obs_dnu/2.
                    starty += self.obs_dnu
                gridx[ii] = startx + self.obs_dnu/n_across/2.
                starty = min(self.frequency)
                startx += self.obs_dnu/n_across
            smoothed = arr
            dim = smoothed.shape

            smoothed_2 = np.zeros((2*dim[0],dim[1]))
            smoothed_2[0:dim[0],:] = smoothed
            smoothed_2[dim[0]:(2*dim[0]),:] = smoothed
            smoothed = np.swapaxes(smoothed_2, 0, 1)
            extent = [min(gridx)-self.obs_dnu/n_across/2., 2*max(gridx)+self.obs_dnu/n_across/2., min(gridy)-self.obs_dnu/2., max(gridy)+self.obs_dnu/2.]
            return smoothed, np.array(list(gridx)+list(gridx+self.obs_dnu)), gridy, extent

    def max_elements(self, x, y):
        s = np.argsort(y)
        peaks_y = y[s][-int(self.fitbg['n_peaks']):][::-1]
        peaks_x = x[s][-int(self.fitbg['n_peaks']):][::-1]
        return peaks_x, peaks_y

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
        ax2.loglog(self.freq, self.pow, 'w-')
        ax2.set_xlim([min(self.freq), max(self.freq)])
        ax2.set_ylim([min(self.pow), max(self.pow)*1.25])
        ax2.set_title(r'$\rm Crude \,\, background \,\, fit$')
        ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
        if self.findex['binning'] is not None:
            ax2.loglog(self.bin_freq, self.bin_pow, 'r-')
        ax2.loglog(self.freq, self.interp_pow, color = 'lime', linestyle = '-', lw = 2.)

        # crude background-corrected power spectrum
        ax3 = plt.subplot(1+self.nrows,3,3)
        ax3.plot(self.freq, self.bgcorr_pow, 'w-')
        ax3.set_xlim([min(self.freq), max(self.freq)])
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
        ax2.plot(self.frequency[self.frequency<self.maxpower[0]], self.power[self.frequency<self.maxpower[0]], 'w-', zorder = 0)
        ax2.plot(self.frequency[self.frequency>self.maxpower[1]], self.power[self.frequency>self.maxpower[1]], 'w-', zorder = 0)
        ax2.plot(self.frequency[self.frequency<self.maxpower[0]], self.smooth_power[self.frequency<self.maxpower[0]], 'r-', linewidth = 0.75, zorder = 1)
        ax2.plot(self.frequency[self.frequency>self.maxpower[1]], self.smooth_power[self.frequency>self.maxpower[1]], 'r-', linewidth = 0.75, zorder = 1)
        for r in range(self.nlaws):
            ax2.plot(self.frequency, harvey(self.frequency, [self.a_orig[r], self.b_orig[r], self.noise]), color = 'blue', linestyle = ':', linewidth = 1.5, zorder = 3)
        ax2.plot(self.frequency, harvey(self.frequency, self.pars, total=True), color = 'blue', linewidth = 2., zorder = 4)
        ax2.errorbar(self.bin_freq, self.bin_pow, yerr = self.bin_err, color = 'lime', markersize = 0., fillstyle = 'none', ls = 'None', marker = 'D', capsize = 3, ecolor = 'lime', elinewidth = 1, capthick = 2, zorder = 2)
        for m, n in zip(self.mnu_orig, self.a_orig):
            ax2.plot(m, n, color = 'blue', fillstyle = 'none', mew = 3., marker = 's', markersize = 5.)
        ax2.axvline(self.maxpower[0], color = 'darkorange', linestyle = 'dashed', linewidth = 2., zorder = 1, dashes = (5,5))
        ax2.axvline(self.maxpower[1], color = 'darkorange', linestyle = 'dashed', linewidth = 2., zorder = 1, dashes = (5,5))
        ax2.axhline(self.noise, color = 'blue', linestyle = 'dashed', linewidth = 1.5, zorder = 3, dashes = (5,5))
        ax2.set_xlim([min(self.frequency), max(self.frequency)])
        ax2.set_ylim([min(self.power), max(self.power)*1.25])
        ax2.set_title(r'$\rm Initial \,\, guesses$')
        ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        # fitted background
        ax3 = fig.add_subplot(3,3,3)
        ax3.plot(self.frequency[self.frequency<self.maxpower[0]], self.power[self.frequency<self.maxpower[0]], 'w-', linewidth = 0.75, zorder = 0)
        ax3.plot(self.frequency[self.frequency>self.maxpower[1]], self.power[self.frequency>self.maxpower[1]], 'w-', linewidth = 0.75, zorder = 0)
        ax3.plot(self.frequency[self.frequency<self.maxpower[0]], self.smooth_power[self.frequency<self.maxpower[0]], 'r-', linewidth = 0.75, zorder = 1)
        ax3.plot(self.frequency[self.frequency>self.maxpower[1]], self.smooth_power[self.frequency>self.maxpower[1]], 'r-', linewidth = 0.75, zorder = 1)
        for r in range(self.nlaws):
            ax3.plot(self.frequency, harvey(self.frequency, [self.pars[2*r], self.pars[2*r+1], self.pars[-1]]), color = 'blue', linestyle = ':', linewidth = 1.5, zorder = 3)
        ax3.plot(self.frequency, harvey(self.frequency, self.pars, total=True), color = 'blue', linewidth = 2., zorder = 4)
        ax3.errorbar(self.bin_freq, self.bin_pow, yerr = self.bin_err, color = 'lime', markersize = 0., fillstyle = 'none', ls = 'None', marker = 'D', capsize = 3, ecolor = 'lime', elinewidth = 1, capthick = 2, zorder = 2)
        ax3.axvline(self.maxpower[0], color = 'darkorange', linestyle = 'dashed', linewidth = 2., zorder = 1, dashes = (5,5))
        ax3.axvline(self.maxpower[1], color = 'darkorange', linestyle = 'dashed', linewidth = 2., zorder = 1, dashes = (5,5))
        ax3.axhline(self.pars[-1], color = 'blue', linestyle = 'dashed', linewidth = 1.5, zorder = 3, dashes = (5,5))
        ax3.plot(self.frequency, self.pssm, color = 'yellow', linewidth = 2., linestyle = 'dashed', zorder = 5)
        ax3.set_xlim([min(self.frequency), max(self.frequency)])
        ax3.set_ylim([min(self.power), max(self.power)*1.25])
        ax3.set_title(r'$\rm Fitted \,\, model$')
        ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        # smoothed power excess w/ gaussian
        ax4 = fig.add_subplot(3,3,4)
        ax4.plot(self.region_freq, self.region_pow, 'w-', zorder = 0)
        idx = return_max(self.region_pow, index=True)
        ax4.plot([self.region_freq[idx]], [self.region_pow[idx]], color = 'red', marker = 's', markersize = 7.5, zorder = 0)
        ax4.axvline([self.region_freq[idx]], color = 'white', linestyle = '--', linewidth = 1.5, zorder = 0)
        ax4.plot(self.new_freq, self.numax_fit, 'b-', zorder = 3)
        ax4.axvline(self.exp_numax, color = 'blue', linestyle = ':', linewidth = 1.5, zorder = 2)
        ax4.plot([self.exp_numax], [max(self.numax_fit)], color = 'b', marker = 'D', markersize = 7.5, zorder = 1)
        ax4.set_title(r'$\rm Smoothed \,\, bg$-$\rm corrected \,\, PS$')
        ax4.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax4.set_xlim([min(self.region_freq), max(self.region_freq)])

        # background-corrected ps with n highest peaks
        peaks_f, peaks_p = self.max_elements(self.freq, self.psd)
        ax5 = fig.add_subplot(3,3,5)
        ax5.plot(self.freq, self.psd, 'w-', zorder = 0, linewidth = 1.0)
        ax5.scatter(peaks_f, peaks_p, s = 25., edgecolor = 'r', marker = 's', facecolor = 'none', linewidths = 1.)
        ax5.set_title(r'$\rm Bg$-$\rm corrected \,\, PS$')
        ax5.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax5.set_ylabel(r'$\rm Power$')
        ax5.set_xlim([min(self.region_freq), max(self.region_freq)])
        ax5.set_ylim([min(self.psd)-0.025*(max(self.psd)-min(self.psd)), max(self.psd)+0.1*(max(self.psd)-min(self.psd))])

        # ACF for determining dnu
        ax6 = fig.add_subplot(3,3,6)
        ax6.plot(self.lag, self.auto, 'w-', zorder = 0, linewidth = 1.)
        ax6.scatter(self.peaks_l, self.peaks_a, s = 30., edgecolor = 'r', marker = '^', facecolor = 'none', linewidths = 1.)
        ax6.axvline(self.exp_dnu, color = 'lime', linestyle='--', linewidth=1.5, zorder=5)
#        ax6.axvline(self.best_lag, color = 'red', linestyle='--', linewidth=1.5, zorder=2)
        ax6.scatter(self.best_lag, self.best_auto, s = 45., edgecolor = 'lime', marker = 's', facecolor = 'none', linewidths = 1.)
        ax6.plot(self.zoom_lag, self.zoom_auto, 'r-', zorder = 5, linewidth = 1.)
        ax6.set_title(r'$\rm ACF \,\, for \,\, determining \,\, \Delta\nu$')
        ax6.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
        ax6.set_xlim([min(self.lag), max(self.lag)])
        ax6.set_ylim([min(self.auto)-0.05*(max(self.auto)-min(self.auto)), max(self.auto)+0.1*(max(self.auto)-min(self.auto))])

        # dnu fit
        ax7 = fig.add_subplot(3,3,7)
        ax7.plot(self.zoom_lag, self.zoom_auto, 'w-', zorder = 0, linewidth = 1.0)
        ax7.axvline(self.obs_dnu, color = 'red', linestyle = '--', linewidth=1.5, zorder=2)
        ax7.plot(self.new_lag, self.dnu_fit, color = 'red', linewidth = 1.5)
        ax7.axvline(self.exp_dnu, color='blue', linestyle=':', linewidth=1.5, zorder=5)
        ax7.set_title(r'$\rm \Delta\nu \,\, fit$')
        ax7.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
        ax7.annotate(r'$\Delta\nu = %.2f$'%self.obs_dnu, xy = (0.025, 0.85), xycoords = "axes fraction", fontsize = 18, color = 'lime')
        ax7.set_xlim([min(self.zoom_lag), max(self.zoom_lag)])

        cmap = plt.get_cmap('binary')
#        new_cmap = cmap(np.linspace(0.1, 0.9, 100))
        colors=truncate_colormap(cmap,0.1,0.8,100)
        # echelle diagram
        ax8 = fig.add_subplot(3,3,8)
        ax8.imshow(self.ech, extent=self.extent, interpolation='none', aspect='auto', origin='lower', cmap=colors)
        ax8.axvline([self.obs_dnu], color = 'white', linestyle = '--', linewidth = 1., dashes=(5,5))
        ax8.set_title(r'$\rm \grave{E}chelle \,\, diagram$')
        ax8.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$'%self.obs_dnu)
        ax8.set_ylabel(r'$\rm \nu \,\, [\mu Hz]$')
        ax8.set_xlim([0., 2.*self.obs_dnu])
        ax8.set_ylim([self.maxpower[0], self.maxpower[1]])

        ax9 = fig.add_subplot(3,3,9)
        ax9.plot(self.xax, self.yax, color = 'white', linestyle = '-', linewidth = 0.75)
        ax9.set_title(r'$\rm Collapsed \,\, \grave{e}chelle \,\, diagram$')
        ax9.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$'%self.obs_dnu)
        ax9.set_ylabel(r'$\rm Collapsed \,\, power$')
        ax9.set_xlim([0., 2.*self.obs_dnu])
        ax9.set_ylim([min(self.yax)-0.025*(max(self.yax)-min(self.yax)), max(self.yax)+0.05*(max(self.yax)-min(self.yax))])

        plt.tight_layout()
        if self.fitbg['save']:
            plt.savefig(self.params[self.target]['path']+'%d_fitbg.png'%self.target, dpi=300)
        if self.show_plots:
            plt.show() 
        plt.close()

    def plot_mc(self):
        plt.figure(figsize = (12,8))
        panels = ['numax_smooth','amp_smooth','numax_gaussian','amp_gaussian','fwhm_gaussian','dnu']
        titles = [r'$\rm Smoothed \,\, \nu_{max} \,\, [\mu Hz]$', r'$\rm Smoothed \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$', r'$\rm Gaussian \,\, \nu_{max} \,\, [\mu Hz]$', r'$\rm Gaussian \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$', r'$\rm Gaussian \,\, FWHM \,\, [\mu Hz]$', r'$\rm \Delta\nu \,\, [\mu Hz]$']
        for i in range(6):
            ax = plt.subplot(2,3,i+1)
            ax.hist(self.df[panels[i]], bins=20, color='cyan', histtype='step', lw=2.5, facecolor='0.75')
            ax.set_title(titles[i])
        plt.tight_layout()
        if self.findex['save']:
            plt.savefig(self.params[self.target]['path']+'%d_mc.png'%self.target, dpi = 300)
        if self.show_plots:
            plt.show()
        plt.close()

##########################################################################################
#                                                                                        #
#                                        INITIATE                                        #
#                                                                                        #
##########################################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        description = """This is the new python version of asteroseismic IDL 
                                         pipeline 'SYD' created by Daniel Huber (see Huber+2009) 
                                         during his Ph.D. in Sydney. This script will initialize 
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
                                         [The ReadTheDocs page is currently under construction.]""")
    parser.add_argument('-ex', '--ex', '-findex', '--findex', '-excess', '--excess',  
                        help = """Turn off the find excess module. This is only recommended when a list
                                  of numaxes or a list of stellar parameters (to estimate the numaxes) 
                                  are provided. Otherwise the second module, which fits the background 
                                  will not be able to run properly.""", 
                        default=True, dest='excess', action='store_false')
    parser.add_argument('-bg', '--bg', '-fitbg', '--fitbg', '-background', '--background', 
                        help = """Turn off the background fitting process (although this is not recommended).
                                  Asteroseismic estimates are typically unreliable without properly removing
                                  stellar contributions from granulation processes. Since this is the money
                                  maker, fitbg is set to 'True' by default.""",
                        default=True, dest='background', action='store_false')
    parser.add_argument('-filter', '--filter', '-smooth', '--smooth', 
                        help = 'Box filter width [muHz] for the power spectrum (Default = 2.5 muHz)',
                        default=2.5, dest='filter')
    parser.add_argument('-kc', '--kc', '-keplercorr', '--keplercorr', 
                        help = """Turn on Kepler short-cadence artefact corrections""", 
                        default=False, dest='keplercorr', action='store_true')
    parser.add_argument('-v', '--v', '-verbose', '--verbose', 
                        help = """Turn on the verbose output. This is generally helpful for a single target,
                                  but would probably not be useful for a long list of targets. By default,
                                  if the length of 'todo' targets is more than one, this output will be 
                                  suppressed. If that is not desired, please add the 'ignore' command line
                                  argument. See help below for 'ignore'.
                                  Please note: the defaults is 'False'.""",
                        default=True, dest='verbose', action='store_true')
    parser.add_argument('-show', '--show', '-plot', '--plot', '-plots', '--plots', 
                        help = """Shows the appropriate output figures in real time. If the findex module is 
                                  run, this will show one figure at the end of findex. If the fitbg module is
                                  run, a figure will appear at the end of the first iteration. If the monte
                                  carlo sampling is turned on, this will provide another figure at the end of
                                  the MC iterations. Regardless of this option, the figures will be saved to
                                  the output directory. If running more than one target, this is not 
                                  recommended. As a result, this is 'False' by default.""",
                        default=True, dest='show', action='store_true')
    parser.add_argument('-ignore', '--ignore', 
                        help = """For multiple targets run at once, the typical output is suppressed. This is 
                                  if you would like to 'ignore' this feature and have it print results in real 
                                  time. This is strongly discouraged for targets running in parallel.""", 
                        default=False, dest='ignore', action='store_true')
    parser.add_argument('-mc', '--mc', '-mciter', '--mciter', dest='mciter', default=1, type=int,
                        help = 'Number of MC iterations (Default = 1)')

    main(args = parser.parse_args())