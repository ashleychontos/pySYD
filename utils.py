import os
import glob
import matplotlib
import numpy as np
import pandas as pd
from functions import *
from QUAKES import delta_nu
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter, ScalarFormatter

matplotlib.rcParams['backend'] = 'TkAgg'

plt.style.use('dark_background')
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['lines.linewidth'] = 1.
plt.rcParams['xtick.major.size'] = 7.
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 3.5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 7.
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 3.5
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.formatter.useoffset'] = False


def main(findex = True, fitbg = True, verbose = False):

    params = make_dict(verbose, findex, fitbg)

    if findex:
        params = find_excess(params)

    if fitbg:
        params = fit_background(params)

    return


##########################################################################################
#                                                                                        #
#                                      DICTIONARIES                                      #
#                                                                                        #
##########################################################################################

def make_dict(verbose, findex, fitbg, mass_sun = 1.9891e33, radius_sun = 6.95508e10, 
              numax_sun = 3090., width_sun = 1300., G = 6.67428e-8, teff_sun = 5777., 
              dnu_sun = 135.1, path = '../Info/todo', star_info = '../Info/star_info.csv',
              tau_sun = [5.2e6, 1.8e5, 1.7e4, 2.5e3, 280., 80.], findex_file = 
              'Files/params_findex.txt', tau_sun_single = [3.8e6, 2.5e5, 1.5e5, 1.0e5, 230., 70.],  
              fitbg_file = 'Files/params_fitbg.txt'):

    params = {}
    params['path'] = path

    if findex:
        params = read_excess_params(findex_file, params)

    if fitbg:
        params = read_bg_params(fitbg_file, params)

    todo = []
    with open(path+'.txt', "r") as f:
        for line in f:
            todo.append(int(float(line.strip().split()[0])))

    params.update({'numax_sun':numax_sun, 'dnu_sun':dnu_sun, 'width_sun':width_sun, 'todo':todo, 'G':G, 'mass_sun':mass_sun, 'radius_sun':radius_sun, 'teff_sun':teff_sun, 'do_fitbg':fitbg, 'tau_sun':tau_sun, 'tau_sun_single':tau_sun_single, 'do_findex':findex, 'verbose':verbose})

    params = get_star_info(star_info, params)

    if verbose:
        print(params)

    return params


##########################################################################################
#                                                                                        #
#                               READING/WRITING TO/FROM FILES                            #
#                                                                                        #
##########################################################################################


def get_file(path):

    f = open(path, "r")
    lines = f.readlines()
    f.close()

    x = []
    y = []

    for line in lines:
        x.append(float(line.strip().split()[0]))
        y.append(float(line.strip().split()[1]))

    return np.array(x), np.array(y)


def read_excess_params(findex_file, params):

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

    print()
    print('# FIND EXCESS PARAMS: %d valid lines read'%i)

    params['findex'] = dict(zip(pars, vals))

    if params['findex']['save'] and not os.path.exists(params['path']+'/results'):
        os.makedirs(path+'/results')

    return params


def write_excess(target, results, params):

    variables = ['target', 'numax', 'dnu', 'snr']
    
    f = open(params['path']+'/%d_findex.txt'%target, "w")
    f.write('{:<15}{:<15}{:<15}{:<15} \n'.format(variables[0], variables[1], variables[2], variables[3]))
    
    values = [str(target), results[0], results[1], results[2]]
    formats = ["<15s", "<15.4f", "<15.4f", "<15.4f"]
    text = '{:{}}'*len(values) + '\n'
    fmt = sum(zip(values, formats), ())
    f.write(text.format(*fmt))   
    f.close()
    
    return


def read_bg_params(fitbg_file, params):

    pars = ['lower_limit', 'upper_limit', 'num_MC_iter', 'n_laws', 'lower_noise', 'upper_noise', 'box_filter', 'ind_width', 'n_rms', 'lower_numax', 'upper_numax', 'lower_lag', 'upper_lag', 'n_peaks', 'smooth_PS', 'plot', 'force', 'guess', 'save', 'clip', 'clip_value', 'ech_smooth', 'ech_filter']
    dtype = [False, False, True, True, False, False, False, True, True, False, False, False, False, True, False, True, True, False, True, True, False, True, False]
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

    print('# FIT BACKGROUND PARAMS: %d valid lines read'%i)

    params['fitbg'] = dict(zip(pars,vals))
    params['fitbg']['functions'] = {1:harvey_one, 2:harvey_two, 3:harvey_three}

    if params['fitbg']['save'] and not os.path.exists(params['path']+'/results'):
        os.makedirs(path+'/results')

    return params


def write_bgfit(target, results, params):

    variables = ['target', 'numax', 'a_']
    
    f = open(params['path']+'/%d_bgfit.txt'%target, "w")
    f.write('{:<15}{:<15}{:<15}{:<15} \n'.format(variables[0], variables[1], variables[2], variables[3]))
    
    values = [str(target), results[0], results[1], results[2]]
    formats = ["<15s", "<15.4f", "<15.4f", "<15.4f"]
    text = '{:{}}'*len(values) + '\n'
    fmt = sum(zip(values, formats), ())
    f.write(text.format(*fmt))   
    f.close()
    
    return


def get_star_info(star_info, params, cols = ['rad', 'logg', 'teff']):

    if os.path.exists(star_info):
        df = pd.read_csv(star_info)
        targets = df.targets.values.tolist()
        for todo in params['todo']:
            if todo in targets:
                params[todo] = {}
                idx = targets.index(todo)
                for col in cols:
                    params[todo][col] = df.loc[idx,col]
                params[todo]['mass'] = (((params[todo]['rad']*params['radius_sun'])**(2.))*10**(params[todo]['logg'])/params['G'])/params['mass_sun']
                params[todo]['nuMax'] = params['numax_sun']*params[todo]['mass']*(params[todo]['rad']**(-2.))*((params[todo]['teff']/params['teff_sun'])**(-0.5))
                params[todo]['dNu'] = params['dnu_sun']*(params[todo]['mass']**(0.5))*(params[todo]['rad']**(-1.5))  
            else:
                print('Target %d did not have star info supplied.'%todo)

        print('# STAR INFO: %d valid lines read'%len(df.targets.values.tolist()))
        return params

    else:
        print('%s file does not exist.'%star_info)
        return


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

def find_excess(params):

    for target in params['todo']:

        if not os.path.exists(params['path']+'/%s_LC.txt'%str(target)):
            print('Error: %s/%s_LC.txt not found'%(params['path'], str(target)))
            break
        else:
            time, flux = get_file(params['path']+'/%s_LC.txt'%str(target))
            print('# LIGHT CURVE: %d lines of data read'%len(time))

        if not os.path.exists(params['path']+'/%s_PS.txt'%str(target)):
            print('Error: %s/%s_PS.txt not found'%(params['path'], str(target)))
            break
        else:
            frequency, power = get_file(params['path']+'/%s_PS.txt'%str(target))
            print('# POWER SPECTRUM: %d lines of data read'%len(frequency))
        print('------------')
        print(target)

        N = int(params['findex']['n_trials']+3)
        if N%3 == 0:
            nrows = (N-1)//3
        else:
            nrows = N//3

        if params['findex']['plot']:

            fig = plt.figure(figsize = (12,8))
            plt.ion()
            plt.show()
        
            ax1 = fig.add_subplot(1+nrows,3,1)
            ax1.plot(time, flux, 'w-')
            ax1.set_xlim([min(time), max(time)])
            ax1.set_title(r'$\rm Time \,\, series$', fontsize = 18)
            ax1.set_xlabel(r'$\rm Time \,\, [days]$', fontsize = 16)
            ax1.set_ylabel(r'$\rm Flux$', fontsize = 16)
            ax1.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
            ax1.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
            ax1.xaxis.set_minor_locator(MultipleLocator(25))
            plt.draw()
            plt.show()

        if params['findex']['lower_limit'] is not None:
            if params['findex']['upper_limit'] is not None:
                mask = np.ma.getmask(np.ma.masked_inside(frequency, params['findex']['lower_limit'], params['findex']['upper_limit']))
                frequency = frequency[mask]
                power = power[mask]
            else:
                mask = np.ma.getmask(np.ma.masked_greater_equal(frequency, params['findex']['lower_limit']))
                frequency = frequency[mask]
                power = power[mask]
        else:
            if params['findex']['upper_limit'] is not None:
                mask = np.ma.getmask(np.ma.masked_less_equal(frequency, params['findex']['upper_limit']))
                frequency = frequency[mask]
                power = power[mask]
    
        if params['findex']['binning'] is not None:
            if params['findex']['plot']:
                ax2 = fig.add_subplot(1+nrows,3,2)
                ax2.loglog(frequency, power, 'w-')
                ax2.set_xlim([min(frequency), max(frequency)])
                ax2.set_ylim([min(power), max(power)*1.25])
                ax2.set_title(r'$\rm Crude \,\, background \,\, fit$', fontsize = 18)
                ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 16)
                ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$', fontsize = 16)
                ax2.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                ax2.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                plt.draw()
                plt.show()

            bin_freq, bin_pow = bin_data(frequency, power, params)
            print('binned to %d datapoints'%len(bin_freq))
            if params['findex']['plot']:
                ax2.loglog(bin_freq, bin_pow, 'r-')
                plt.tight_layout()
                plt.draw()
                plt.show()

            if params['findex']['smooth_width'] is not None:
                smooth_pow = convolve(bin_pow, Box1DKernel(params['findex']['smooth_width']))
            else:
                smooth_pow = bin_pow[:]

            resolution = np.nanmedian((np.diff(frequency)))
            params['resolution'] = resolution
            boxsize = np.ceil(float(params['findex']['smooth_width'])/resolution)
            box_kernel = Box1DKernel(boxsize)
            smooth_pow = convolve(bin_pow, box_kernel)
            sf = bin_freq[int(boxsize/2):-int(boxsize/2)]
            sp = smooth_pow[int(boxsize/2):-int(boxsize/2)]

            s = InterpolatedUnivariateSpline(sf, sp, k = 1)
            interp_pow = s(frequency)
            if params['findex']['plot']:
                ax2.loglog(frequency, interp_pow, color = 'lime', linestyle = '-', lw = 2.)
                plt.tight_layout()
                plt.draw()
                plt.show()
                if params['findex']['check']:
                    answer = input('continue (1-y/0-n)? ')
                    try:
                        answer = int(answer)
                    except:
                        answer = int(input('Did not understand input. Please enter 1 for yes or 0 for no. '))
                    if answer == 0:
                        new_bin = float(input('enter new binlength: '))
                        new_smooth = float(input('enter new smoothlength: '))

            original_power = power[:]
            bgcorr_power = power/interp_pow

            if params['findex']['plot']:
                ax3 = fig.add_subplot(1+nrows,3,3)
                ax3.plot(frequency, bgcorr_power, 'w-')
                ax3.set_xlim([min(frequency), max(frequency)])
                ax3.set_ylim([min(bgcorr_power), max(bgcorr_power)*1.25])
                ax3.set_title(r'$\rm Background \,\, corrected \,\, PS$', fontsize = 18)
                ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 16)
                ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$', fontsize = 16)
                ax3.xaxis.set_minor_locator(MultipleLocator(250))
                ax3.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                ax3.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                plt.tight_layout()
                plt.draw()
                plt.show()

            results = []

            if params['findex']['long_cadence']:
                boxes = np.logspace(np.log10(0.5), np.log10(25.), params['findex']['n_trials'])*params['findex']['box']
            else:
                boxes = np.logspace(np.log10(50.), np.log10(1000.), params['findex']['n_trials'])*params['findex']['box']

            for i, box in enumerate(boxes):

                subset = np.ceil(box/resolution)
                steps = np.ceil((box*params['findex']['step'])/resolution)

                cumsum = np.zeros_like(frequency)
                md = np.zeros_like(frequency)
                j = 0
                start = 0

                while True:

                    if (start+subset) > len(frequency):
                        break

                    freq = frequency[int(start):int(start+subset)]
                    pow = bgcorr_power[int(start):int(start+subset)]

                    f = freq[:]
                    p = pow[:]

                    n = len(p)
                    mean = np.mean(p)
                    var = np.var(p)   
                    N = np.arange(n)
    
                    lag = N*resolution
    
                    auto = np.correlate(p - mean, p - mean, "full")    
                    auto = auto[int(auto.size/2):]

                    mask = np.ma.getmask(np.ma.masked_inside(lag, params['findex']['lower_lag'], params['findex']['upper_lag']))
    
                    l = lag[mask]
                    a = auto[mask]
    
                    lag = l[:]
                    auto = a[:]
                    auto = (auto - np.mean(auto))
                    cor = np.absolute(auto)

                    cumsum[j] = np.sum(cor)
                    md[j] = np.mean(f)

                    start += steps
                    j += 1
            
                mask = np.ma.getmask(np.ma.masked_values(cumsum, 0.0))
                md = md[~mask]
                cumsum = cumsum[~mask] - min(cumsum[~mask])
                cumsum = list(cumsum/max(cumsum))
                idx = cumsum.index(max(cumsum))

                if params['findex']['plot']:
                    ax = fig.add_subplot(1+nrows,3,4+i)
                    ax.plot(md, cumsum, 'w-')
                    ax.axvline(md[idx], linestyle = 'dotted', color = 'r', linewidth = 0.75)
                    ax.set_xlim([min(frequency), max(frequency)])
                    ax.set_ylim([-0.05, 1.15])
                    ax.set_title(r'$\rm collapsed \,\, ACF \,\, [trial \,\, %d]$'%(i+1), fontsize = 18)
                    ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 16)
                    ax.set_ylabel(r'$\rm Arbitrary \,\, units$', fontsize = 16)
                    ax.xaxis.set_minor_locator(MultipleLocator(250))
                    ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                    ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                    plt.tight_layout()
                    plt.draw()
                    plt.show()

                p = [np.mean(cumsum), 1.0-np.mean(cumsum), md[idx], 1000.]
                best_vars, covar = curve_fit(gaussian, md, cumsum, p0 = p)
                x = np.linspace(min(md), max(md), 1001)
                y = gaussian(x, best_vars[0], best_vars[1], best_vars[2], best_vars[3])
                SNR = max(y)/best_vars[0]
                if SNR > 100.:
                    SNR = 100.
                dNu = delta_nu(best_vars[2])
                results.append([best_vars[2], dNu, SNR])
                print('power excess trial %d: numax = %.2f +/- %.2f'%(i+1, best_vars[2], np.absolute(best_vars[3])/2.))
                print('S/N: %.2f'%SNR)

                if params['findex']['plot']:
                    ax.plot(x, y, color = 'lime', linestyle = '-', linewidth = 1.5)
                    ax.axvline(best_vars[2], color = 'lime', linestyle = '--', linewidth = 0.75)
                    ax.annotate(r'$\rm SNR = %3.2f$'%SNR, xy = (500., 0.9), fontsize = 18)
                    plt.tight_layout()
                    plt.draw()
                    plt.show()

                input(':')

            compare = [each[-1] for each in results]
            best = compare.index(max(compare))
            print('picking model %d'%(best+1))
            write_excess(target, results[best], params)

            plt.ioff()
            if params['findex']['save']:
                plt.savefig(params['path']+'/results/%d_findex.png'%target, dpi = 150)
            plt.close()
            input(':')

        else:
            print('This will be extremely difficult to do without binning. Check your input parameters file (findex_params.txt)')

    return params


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

def fit_background(params):

    for target in params['todo']:

        if os.path.exists(params['path']+'/%d_findex.txt'%target):
            f = open(params['path']+'/%d_findex.txt'%target)
            lines = f.readlines()
            line = lines[1]
            for i, var in enumerate(['maxp', 'delnu', 'snr']):
                params[target][var] = float(line.strip().split()[i+1])
        else:
            if target in params.keys():
                params[target]['maxp'] = params[target]['nuMax']
                params[target]['delnu'] = params[target]['dNu']
                params[target]['snr'] = np.nan
            else:
                print("WARNING: Suggested use of this pipeline requires either")
                print("stellar properties to estimate a nuMax or run the entire")
                print("pipeline (i.e. run find_excess first) to statistically")
                print("determine a starting point for nuMax.")
                return
               
        if not os.path.exists(params['path']+'/%s_LC.txt'%str(target)):
            print('Error: %s/%s_LC.txt not found'%(params['path'], str(target)))
            break
        else:
            time, flux = get_file(params['path']+'/%s_LC.txt'%str(target))
            print('# LIGHT CURVE: %d lines of data read'%len(time))

        if not os.path.exists(params['path']+'/%s_PS.txt'%str(target)):
            print('Error: %s/%s_PS.txt not found'%(params['path'], str(target)))
            break
        else:
            frequency, power = get_file(params['path']+'/%s_PS.txt'%str(target))
            print('# POWER SPECTRUM: %d lines of data read'%len(frequency))
        oversample = int(round((1./((max(time)-min(time))*0.0864))/(np.nanmedian(np.diff(frequency)))))
        print('------------')
        print(target)
        if oversample:
            print('critically sampled')
        else:
            print('oversampled by a factor of %d'%oversample)

        if params['fitbg']['lower_limit'] is not None:
            if params['fitbg']['upper_limit'] is not None:
                mask = np.ma.getmask(np.ma.masked_inside(frequency, params['fitbg']['lower_limit'], params['fitbg']['upper_limit']))
                frequency = frequency[mask]
                power = power[mask]
            else:
                mask = np.ma.getmask(np.ma.masked_greater_equal(frequency, params['fitbg']['lower_limit']))
                frequency = frequency[mask]
                power = power[mask]
        else:
            if params['fitbg']['upper_limit'] is not None:
                mask = np.ma.getmask(np.ma.masked_less_equal(frequency, params['fitbg']['upper_limit']))
                frequency = frequency[mask]
                power = power[mask]

        cad = int(np.nanmedian(np.diff(time)*24.*60.*60.))
        nyq = 10**6/(2.*cad)
        res = np.nanmedian(np.diff(frequency))*oversample
        print('time series cadence: %d seconds'%cad)
        print('power spectrum resolution: %.6f muHz'%res)
        width = params['width_sun']*(params[target]['maxp']/params['numax_sun'])
        times = width/params[target]['delnu']
        params.update({'cadence':cad, 'resolution':res, 'nyquist':nyq, 'width':width, 'times':times})

        # Arbitrary SNR cut for leaving region out of background fit, statistically validate later?
        if os.path.exists(params['path']+'/%d_findex.txt'%target):
            if params['fitbg']['lower_numax'] is not None:
                maxpower = [params['fitbg']['lower_numax'], params['fitbg']['upper_numax']]
            else:
                if params[target]['snr'] > 2.:
                    maxpower = [params[target]['maxp']-width/2., params[target]['maxp']+width/2.]
                else:
                    maxpower = [params[target]['maxp']-times*params[target]['delnu'],params[target]['maxp']+times*params[target]['delnu']]
        else:
            if params['fitbg']['lower_numax'] is not None:
                maxpower = [params['fitbg']['lower_numax'], params['fitbg']['upper_numax']]
            else:
                maxpower = [params[target]['maxp']-times*params[target]['delnu'],params[target]['maxp']+times*params[target]['delnu']]

        # Create independent frequency points (need to if oversampled)
        if not oversample:
            orig_freq = frequency[oversample-1::oversample]
            orig_pow = power[oversample-1::oversample]
        else:
            orig_freq = frequency[:]
            orig_pow = power[:]

        nlaws = params['fitbg']['n_laws']

        for i in range(params['fitbg']['num_MC_iter']):

            dnu_exp = params[target]['delnu']
        
            if i == 0:
                synth_pow = orig_pow[:]
            else:
                synth_pow = (np.random.chisquare(2, len(orig_freq))*orig_pow)/2.

            xo = orig_freq[:]
            yo = synth_pow[:]

            xo_bin, yo_bin, erro_bin = mean_smooth_ind(xo, yo, params['fitbg']['ind_width'])
            binned_res = np.nanmedian(np.diff(xo_bin))
            params['binned_resolution'] = binned_res
            print('binned resolution: %.2f muHz'%binned_res)

            if params['fitbg']['upper_noise'] is not None:
                noise = np.mean(yo[(xo>params['fitbg']['lower_noise'])&(xo<params['fitbg']['upper_noise'])])
            else:
                noise = np.mean(yo[(xo>(max(xo)-0.1*max(xo)))])
            pars = np.zeros((nlaws*2+1))
            pars[2*nlaws] = noise

            msk = (xo>maxpower[0])&(xo<maxpower[1])
            x = xo[~msk]
            limits = xo[msk]
            test = list(xo[:])
            lower_idx = test.index(min(limits))
            upper_idx = test.index(max(limits))
            y = yo[~msk]
            msk = (xo_bin>maxpower[0])&(xo_bin<maxpower[1])
            x_bin = xo_bin[~msk]
            y_bin = yo_bin[~msk]
            err_bin = erro_bin[~msk]
            s = smooth(y, params['fitbg']['box_filter'], params, silent = True)

            scale = params['numax_sun']/((maxpower[1]+maxpower[0])/2.)

            if orig_freq.min() > 0.01:
                taus = np.array(params['tau_sun'][int(np.log10(orig_freq.min()*100.)):])*scale
            else:
                taus = np.array(params['tau_sun'])*scale

            ex = nlaws-len(taus)
            if ex <= 0:
                ex = 0
            else:
                taus = np.array(params['tau_sun'][-nlaws:])*scale

            b = 2.*np.pi*(taus*1e-6)
            mnu = (1./taus)*1e5
            a = np.zeros_like(mnu)

            if max(mnu) < min(x_bin):
                scup = min(x_bin)/max(mnu)
                mnu[-1] = scup*mnu[-1]
                taus[-1] = 1e5/mnu[-1]
                b[-1] = 2.*np.pi*(taus[-1]*1e-6)

            for j, nu in enumerate(mnu):
                idx = 0
                while x[idx] < nu:
                    idx += 1
                if idx < params['fitbg']['n_rms']:
                    a[j] = np.mean(s[:params['fitbg']['n_rms']])
                elif (len(s) - idx) < params['fitbg']['n_rms']:
                    a[j] = np.mean(s[-params['fitbg']['n_rms']:])
                else:
                    a[j] = np.mean(s[idx-int(params['fitbg']['n_rms']/2):idx+int(params['fitbg']['n_rms']/2)])

            for n in range(nlaws):
                pars[2*n] = a[n]
                pars[2*n+1] = b[n]

            if i == 0:

                if params['fitbg']['plot']:

                    fig = plt.figure(figsize = (12,12))
                    plt.ion()
                    plt.show()

                    ax1 = fig.add_subplot(3,3,1)
                    ax1.plot(time, flux, 'w-')
                    ax1.set_xlim([min(time), max(time)])
                    ax1.set_title(r'$\rm Time \,\, series$', fontsize = 18)
                    ax1.set_xlabel(r'$\rm Time \,\, [days]$', fontsize = 16)
                    ax1.set_ylabel(r'$\rm Flux$', fontsize = 16)
                    ax1.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                    ax1.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                    ax1.xaxis.set_minor_locator(MultipleLocator(25))
                    plt.tight_layout()

                    plt.draw()
                    plt.show()

                    ax2 = fig.add_subplot(3,3,2)
                    ax2.plot(x[x<maxpower[0]], y[x<maxpower[0]], 'w-', zorder = 0)
                    ax2.plot(x[x>maxpower[1]], y[x>maxpower[1]], 'w-', zorder = 0)
                    ax2.plot(x[x<maxpower[0]], s[x<maxpower[0]], 'r-', linewidth = 0.75, zorder = 1)
                    ax2.plot(x[x>maxpower[1]], s[x>maxpower[1]], 'r-', linewidth = 0.75, zorder = 1)
                    for r in range(nlaws):
                        pams = list(pars[2*r:2*r+2])
                        pams.append(pars[-1])
                        ax2.plot(orig_freq, harvey(orig_freq, pams), color = 'cyan', linestyle = ':', linewidth = 1.5, zorder = 3)
                    ax2.plot(orig_freq, harvey(orig_freq, pars), color = 'blue', linewidth = 2., zorder = 4)
                    ax2.errorbar(x_bin, y_bin, yerr = err_bin, color = 'lime', markersize = 0., fillstyle = 'none', ls = 'None', marker = 'D', capsize = 3, ecolor = 'lime', elinewidth = 1, capthick = 2, zorder = 2)
                    for m, n in zip(mnu,a):
                        ax2.plot(m, n, color = 'blue', fillstyle = 'none', mew = 3., marker = 's', markersize = 5.)
                    ax2.axvline(maxpower[0], color = 'darkorange', linestyle = 'dashed', linewidth = 2.5, zorder = 1)
                    ax2.axvline(maxpower[1], color = 'darkorange', linestyle = 'dashed', linewidth = 2.5, zorder = 1)
                    ax2.set_xlim([min(frequency), max(frequency)])
                    ax2.set_ylim([min(power), max(power)*1.25])
                    ax2.set_title(r'$\rm Initial \,\, guesses$', fontsize = 18)
                    ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 16)
                    ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$', fontsize = 16)
                    ax2.set_xscale('log')
                    ax2.set_yscale('log')
                    ax2.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                    ax2.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                    plt.tight_layout()

                    plt.draw()
                    plt.show()

                print('Comparing %d different models:'%(params['fitbg']['n_laws']*2))
                # Test n different models
                bounds = []
                for law in range(params['fitbg']['n_laws']):
                    b = np.zeros((2,2*(law+1)+1)).tolist()
                    for z in range(2*(law+1)):
                        b[0][z] = -np.inf
                        b[1][z] = np.inf
                    b[0][-1] = pars[-1]-0.1
                    b[1][-1] = pars[-1]+0.1
                    bounds.append(tuple(b))

                reduced_chi2 = []
                paras = []
                paras_errs = []
                obs = y[:]
                tt = np.arange(2*params['fitbg']['n_laws'])
                names = ['one', 'one', 'two', 'two', 'three', 'three']
                dict1 = dict(zip(tt,names[:2*params['fitbg']['n_laws']]))
                for t in range(2*params['fitbg']['n_laws']):
                    if t%2 == 0:
                        print('%d: %s harvey model w/ white noise free parameter'%(t+1, dict1[t]))
                        delta = 2*(params['fitbg']['n_laws']-(t//2+1))
                        pams = list(pars[:(-delta-1)])
                        pams.append(pars[-1])
                        pp, cv = curve_fit(params['fitbg']['functions'][t//2+1], x_bin, y_bin, p0 = pams, sigma = err_bin)
                        pe = np.sqrt(np.diag(cv))
                        paras.append(pp)
                        paras_errs.append(pe)
                        exp = harvey(x, pp, total = True)
                        chi, p = chisquare(f_obs = obs, f_exp = exp)
                        reduced_chi2.append(chi/(len(x)-len(pams)))
                    else:
                        print('%d: %s harvey model w/ white noise fixed'%(t+1, dict1[t]))
                        delta = 2*(params['fitbg']['n_laws']-(t//2+1))
                        pams = list(pars[:(-delta-1)])
                        pams.append(pars[-1])
                        pp, cv = curve_fit(params['fitbg']['functions'][t//2+1], x_bin, y_bin, p0 = pams, sigma = err_bin, bounds = bounds[t//2])
                        pe = np.sqrt(np.diag(cv))
                        paras.append(pp)
                        paras_errs.append(pe)
                        exp = harvey(x, pp, total = True)
                        chi, p = chisquare(f_obs = obs, f_exp = exp)
                        reduced_chi2.append(chi/(len(x)-len(pams)+1))

                model = reduced_chi2.index(min(reduced_chi2))+1
                print('Based on reduced chi-squared statistic: choosing model %d'%model)
                if model == 5 or model == 6:
                    nlaws = 3
                elif model == 3 or model == 4:
                    nlaws = 2
                else:
                    nlaws = 1
                pars = paras[model-1]
                pars_errs = paras_errs[model-1]
                final_pars = np.zeros((params['fitbg']['num_MC_iter'],nlaws*2+1+11))
                params.update({'best_model':model-1, 'n_laws':nlaws})

                # get rid of edge effects if numax is close to nyquist frequency
                if (params['nyquist']-params[target]['maxp']) < 1000.:
                    x_add = np.arange(orig_freq[-1], params['nyquist']+2000., params['resolution'])
                    y_add = generate_model(x_add, pars, pars_errs, params['nyquist'])
                    params.update({'edge':[True, len(x_add)]})
                else:
                    y_add = []
                    params.update({'edge':[False, None]})

                sm_par = 4.*(params[target]['maxp']/params['numax_sun'])**0.2
                if sm_par < 1.:
                    sm_par = 1.

                if params['fitbg']['plot']:

                    ax3 = fig.add_subplot(3,3,3)
                    ax3.plot(x[x<maxpower[0]], y[x<maxpower[0]], 'w-', zorder = 0)
                    ax3.plot(x[x>maxpower[1]], y[x>maxpower[1]], 'w-', zorder = 0)
                    ax3.plot(x[x<maxpower[0]], s[x<maxpower[0]], 'r-', linewidth = 0.75, zorder = 1)
                    ax3.plot(x[x>maxpower[1]], s[x>maxpower[1]], 'r-', linewidth = 0.75, zorder = 1)
                    for r in range(nlaws):
                        ax3.plot(orig_freq, harvey(orig_freq, [pars[2*r], pars[2*r+1], pars[-1]]), color = 'cyan', linestyle = ':', linewidth = 1.5, zorder = 3)
                    ax3.plot(orig_freq, harvey(orig_freq, pars, total = True), color = 'blue', linewidth = 3., zorder = 4)
                    ax3.errorbar(x_bin, y_bin, yerr = err_bin, color = 'lime', markersize = 0., fillstyle = 'none', ls = 'None', marker = 'D', capsize = 3, ecolor = 'lime', elinewidth = 1, capthick = 2, zorder = 2)
                    ax3.axvline(maxpower[0], color = 'darkorange', linestyle = 'dashed', linewidth = 2.5, zorder = 1)
                    ax3.axvline(maxpower[1], color = 'darkorange', linestyle = 'dashed', linewidth = 2.5, zorder = 1)
                    ax3.set_xlim([min(x), max(x)])
                    ax3.set_ylim([min(y), max(y)*1.25])
                    ax3.set_title(r'$\rm Fitted \,\, model$', fontsize = 18)
                    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 16)
                    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$', fontsize = 16)
                    ax3.set_xscale('log')
                    ax3.set_yscale('log')
                    ax3.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                    ax3.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                    plt.tight_layout()
                    plt.draw()
                    plt.show()

                best_fit = harvey(orig_freq, pars, total = True)

            else:

                pars, cv = curve_fit(params['fitbg']['functions'][params['best_model']//2+1], x_bin, y_bin, p0 = pams, sigma = err_bin, bounds = bounds[params['best_model']//2])
                pars_errs = np.sqrt(np.diag(cv))

            xtemp = orig_freq[:]
            ytemp = list(synth_pow[:])

            pssm = smooth_gauss(ytemp+y_add, sm_par*dnu_exp/params['resolution'], params, silent = True)
            model = harvey(xtemp, pars, total = True)
            
            # fix any residual slope in gaussian fit and correct it
            msk = (xtemp >= maxpower[0])&(xtemp <= maxpower[1])
            x0 = list(xtemp[msk])
            t0 = pssm[msk]
            t1 = model[msk]
            delta_y = t0[-1] - t0[0]
            delta_x = x0[-1] - x0[0]
            slope = delta_y/delta_x
            b = slope*(-1.*x0[0]) + t0[0]
            corrected = np.array([x0[z]*slope + b for z in range(len(x0))])
            corr_pssm = [t0[z]-corrected[z] + t1[z] for z in range(len(t0))]
            x2 = list(xtemp[~msk])
            t2 = list(model[~msk])
            final_x = np.array(x0+x2)
            final_y = np.array(corr_pssm+t2)
            ss = np.argsort(final_x)
            final_x = final_x[ss]
            final_y = final_y[ss]

            if i == 0 and params['fitbg']['plot']:

                ax3.plot(final_x, final_y, color = 'yellow', linewidth = 2., linestyle = 'dashed', zorder = 5)
                plt.tight_layout()
                plt.draw()
                plt.show()

            pssm_0 = pssm-model
            pssm_corr = final_y[:]
            pssm_bgcor = pssm_corr-model

            xt = list(xtemp[msk])
            ptt = list(pssm_0[msk])
            idx1 = ptt.index(max(ptt))
            pct = list(pssm_bgcor[msk])
            idx = pct.index(max(pct))
            maxamp = pct[idx]
            numax = xt[idx]
            maxamp1 = ptt[idx1]
            numax1 = xt[idx1]

            # somehow test if numax is close to maxp

            if i == 0:
                maxval = pssm[idx]

            maxpower0 = [numax-params['times']*dnu_exp, numax+params['times']*dnu_exp]
            msk = (xtemp >= maxpower0[0])&(xtemp <= maxpower0[1])
            useg = list(xtemp[msk])

            if useg != []:

                xu = xtemp[msk]
                yu = pssm_bgcor[msk]
                min_sig = (max(xu)-min(xu))/8./np.sqrt(8.*np.log(2.))

                bounds = []
                b = np.zeros((2,4)).tolist()
                b[1][0] = np.inf
                b[1][1] = 2.*np.max(yu)
                b[0][2] = np.min(xu)
                b[1][2] = np.max(xu)
                b[0][3] = min_sig
                b[1][3] = np.max(xu) - np.min(xu)
                bounds.append(tuple(b))

                p_gauss, p_cov = curve_fit(gaussian, xu, yu, p0 = [0., max(yu), numax, min_sig], bounds = bounds[0])

                if i == 0 and params['fitbg']['plot']:

                    ax4 = fig.add_subplot(3,3,4)
                    ax4.plot(xt, ptt, 'r-', zorder = 0)
                    ax4.plot(xt, pct, 'w-', zorder = 1)
                    ax4. plot([numax1], [maxamp1], color = 'red', marker = 's', markersize = 7.5, zorder = 0)
                    ax4.plot([numax], [maxamp], color = 'cyan', marker = 'D', markersize = 7.5, zorder = 1)
                    ax4.axvline([numax1], color = 'red', linestyle = '--', linewidth = 1.5, zorder = 0)
                    ax4.axvline([numax], color = 'cyan', linestyle = '-.', linewidth = 1.5, zorder = 1)
                    gaus = gaussian(xu, p_gauss[0], p_gauss[1], p_gauss[2], p_gauss[3])
                    plot_min = 0.
                    if min(pct) < plot_min:
                        plot_min = min(pct)
                    if min(ptt) < plot_min:
                        plot_min = min(ptt)
                    if min(gaus) < plot_min:
                        plot_min = min(gaus)
                    plot_max = 0.
                    if max(pct) > plot_max:
                        plot_max = max(pct)
                    if max(ptt) > plot_max:
                        plot_max = max(ptt)
                    if max(gaus) > plot_max:
                        plot_max = max(gaus)
                    plot_range = plot_max - plot_min
                    ax4.plot(xu, gaus, 'b-', zorder = 3)
                    ax4.axvline([p_gauss[2]], color = 'blue', linestyle = ':', linewidth = 1.5, zorder = 2)
                    ax4.plot([p_gauss[2]], [p_gauss[1]], color = 'b', marker = 'D', markersize = 7.5, zorder = 1)
                    ax4.set_title(r'$\rm Smoothed \,\, bg-corrected \,\, PS$', fontsize = 18)
                    ax4.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 16)
                    ax4.set_xlim([min(xt), max(xt)])
                    ax4.set_ylim([plot_min - 0.1*plot_range, plot_max + 0.1*plot_range])
                    ax4.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                    ax4.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                    plt.tight_layout()
                    plt.draw()
                    plt.show()

            final_pars[i,0:2*nlaws+1] = pars
            final_pars[i,2*nlaws+2] = numax
            final_pars[i,2*nlaws+3] = maxamp
            final_pars[i,2*nlaws+4] = p_gauss[1]
            final_pars[i,2*nlaws+5] = p_gauss[2]
            final_pars[i,2*nlaws+6] = p_gauss[3]

            f_cor = orig_freq[:]
            a_cor = synth_pow[:]
            a_cor_coadd = synth_pow[:]/model

            # optional smoothing of PS to remove fine structure
            if params['fitbg']['smooth_PS'] is not None:
                a_cor_coadd = smooth(a_cor_coadd, params['fitbg']['smooth_PS'], params, silent = True)

            # redetermine power excess using bg-corrected PS
#            if i == 0:
#                excess_pars = confirm_excess(f_cor, a_cor_coadd, p_gauss[2], target, params) 

            center = p_gauss[2]
            dnu_exp = 0.22*(center**0.797)
            lim_factor = times*dnu_exp
            if params['fitbg']['lower_numax'] is not None:
                msk = (f_cor >= params['fitbg']['lower_numax'])&(f_cor <= params['fitbg']['upper_numax'])
            else:
                msk = (f_cor >= center-lim_factor)&(f_cor <= center+lim_factor)
            freq = f_cor[msk]
            psd = a_cor_coadd[msk]

            indices = max_elements(list(psd), params['fitbg']['n_peaks'], params['resolution'], limit = [True, 2.])
            peaks_f = freq[indices]
            peaks_p = psd[indices]

            ss = np.argsort(peaks_f)
            peaks_f = peaks_f[ss]
            peaks_p = peaks_p[ss]

            if i == 0 and params['fitbg']['plot']:

                ax5 = fig.add_subplot(3,3,5)
                ax5.plot(freq, psd, 'w-', zorder = 0, linewidth = 1.0)
                ax5.scatter(peaks_f, peaks_p, s = 25., edgecolor = 'r', marker = 's', facecolor = 'none', linewidths = 1.)
                ax5.set_title(r'$\rm Bg-corrected \,\, power \,\, spectrum$', fontsize = 18)
                ax5.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 16)
                ax5.set_ylabel(r'$\rm Power$', fontsize = 16)
                ax5.set_xlim([min(freq), max(freq)])
                ax5.set_ylim([min(psd) - 0.1*(max(psd)-min(psd)), max(psd) + 0.1*(max(psd)-min(psd))])
                ax5.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                ax5.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                plt.tight_layout()
                plt.draw()
                plt.show()

            lag, auto = corr(freq, psd, params)
            new_auto = auto - min(auto)
            auto = new_auto[:]
            indices = max_elements(list(auto), params['fitbg']['n_peaks'], params['resolution'], limit = [True, 20.*params['resolution']])
            peaks_l = lag[indices]
            peaks_a = auto[indices]

            ss = np.argsort(peaks_l)
            peaks_l = peaks_l[ss]
            peaks_a = peaks_a[ss]

            temp_a = list(peaks_a)
            temp_idx = temp_a.index(max(temp_a))
            best_lag = peaks_l[temp_idx]
            best_auto = peaks_a[temp_idx]

            if i == 0 and params['fitbg']['plot']:

                ax6 = fig.add_subplot(3,3,6)
                ax6.plot(lag, auto, 'w-', zorder = 0, linewidth = 1.0)
                ax6.scatter(peaks_l[:temp_idx], peaks_a[:temp_idx], s = 25., edgecolor = 'r', marker = '^', facecolor = 'none', linewidths = 1.)
                ax6.scatter(peaks_l[temp_idx+1:], peaks_a[temp_idx+1:], s = 25., edgecolor = 'r', marker = '^', facecolor = 'none', linewidths = 1.)
                ax6.axvline([best_lag], color = 'red', linestyle = '--', linewidth = 1.5, zorder = 2)
                ax6.scatter(best_lag, best_auto, s = 25., edgecolor = 'lime', marker = 's', facecolor = 'none', linewidths = 1.)
                ax6.set_title(r'$\rm ACF \,\, for \,\, determining \,\, \Delta\nu$', fontsize = 18)
                ax6.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$', fontsize = 16)
                ax6.set_xlim([min(lag), max(lag)])
                ax6.set_ylim([min(auto) - 0.1*(max(auto)-min(auto)), max(auto) + 0.1*(max(auto)-min(auto))])
                ax6.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                ax6.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                plt.tight_layout()
                plt.draw()
                plt.show()

            rough_sig = best_lag*0.01*2.

            bounds = []
            b = np.zeros((2,4)).tolist()
            b[0][0] = -np.inf
            b[1][0] = np.inf
            b[0][1] = 2.*np.min(auto)
            b[1][1] = 2.*np.max(auto)
            b[0][2] = best_lag - 0.001*best_lag
            b[1][2] = best_lag + 0.001*best_lag
            b[0][3] = 10**-2
            b[1][3] = np.max(lag) - np.min(lag)
            bounds.append(tuple(b))

            p_gauss, p_cov = curve_fit(gaussian, lag, auto, p0 = [np.mean(auto), best_auto-np.mean(auto), best_lag, rough_sig], bounds = bounds[0])
            p_gauss_errs = np.sqrt(np.diag(p_cov))

            acf_region = [best_lag - 1.5*p_gauss[3]*np.sqrt(8.*np.log(2.)), best_lag + 1.5*p_gauss[3]*np.sqrt(8.*np.log(2.))]
            mask = (lag >= acf_region[0])&(lag <= acf_region[1])
            zoom_lag = lag[mask]
            zoom_auto = auto[mask]
            fit = gaussian(zoom_lag, p_gauss[0], p_gauss[1], p_gauss[2], p_gauss[3])

            plot_lower = min(zoom_auto)
            if min(fit) < plot_lower:
                plot_lower = min(fit)

            plot_upper = max(zoom_auto)
            if max(fit) > plot_upper:
                plot_upper = max(fit)

            gauss = list(fit)
            dnu = zoom_lag[gauss.index(max(gauss))]
            final_pars[i,2*nlaws+7] = dnu

            if i == 0 and params['fitbg']['plot']:

                ax7 = fig.add_subplot(3,3,7)
                ax7.plot(zoom_lag, zoom_auto, 'w-', zorder = 0, linewidth = 1.0)
                ax7.axvline([dnu_exp], color = 'red', linestyle = '--', linewidth = 1.5, zorder = 2)
                ax7.plot(zoom_lag, fit, color = 'lime', linewidth = 1.5)
                ax7.axvline([dnu], color = 'lime', linestyle = '--', linewidth = 1.5)
                ax7.set_title(r'$\rm \Delta\nu \,\, fit$', fontsize = 18)
                ax7.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$', fontsize = 16)
                ax7.annotate(r'$\Delta\nu = %.2f$'%dnu, xy = (0.05, 0.05), xycoords = "axes fraction", fontsize = 18, color = 'lime')
                ax7.set_xlim([min(zoom_lag), max(zoom_lag)])
                ax7.set_ylim([plot_lower - 0.1*(plot_upper-plot_lower), plot_upper + 0.1*(plot_upper-plot_lower)])
                ax7.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
                ax7.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
                plt.tight_layout()
                plt.draw()
                plt.show()

            if params['fitbg']['force']:
                dnu = params['fitbg']['guess']

            freq = f_cor[:]
            psd = a_cor[:]/model

            fig = get_ridges(freq, psd, dnu, params, msk, fig, i)
            if params['fitbg']['save']:
                plt.savefig(params['path']+'/results/%d_fitbg.png'%target, dpi = 150)
            plt.show() 

            # for the original data set, we try several SNR cuts drawn from a uniform random
            # distribution to determine the best straightened echelle diagram; for synthetic
            # datasets, we only do this once 

    return


def echelle(x, y, dnu, params, n_per_bin = 20):

    if params['fitbg']['ech_smooth']:
        y = smooth(y, params['fitbg']['ech_filter'], params, silent = True)

    x0 = x[:]
    y0 = y[:]

    boxx = params['resolution']*n_per_bin
    boxy = dnu

    nox = int(np.ceil(dnu/boxx))
    noy = int(np.ceil((max(x)-min(x))/boxy))

    if nox > 2 and noy > 5:

        xax = np.arange(0., dnu+boxx/2., boxx)
        yax = np.arange(min(x), max(x), dnu)

        arr = np.zeros((len(xax),len(yax)))
        gridx = np.zeros(len(xax))
        gridy = np.zeros(len(yax))

        modx = x%dnu

        startx = 0.
        starty = min(x)

        for i in range(len(gridx)):
 
            for j in range(len(gridy)):

                use = np.where((modx >= startx)&(modx < startx+boxx)&(x >= starty)&(x < starty+dnu))[0]
                if len(use) == 0:
                    arr[i,j] = np.nan
                else:
                    arr[i,j] = np.sum(y[use])
                gridy[j] = starty + dnu/2.
                starty += dnu

            gridx[i] = startx + boxx/2.
            starty = min(x)
            startx += boxx

        smoothed = arr
        dim = smoothed.shape

        smoothed_2 = np.zeros((2*dim[0],dim[1]))
        smoothed_2[0:dim[0],:] = smoothed
        smoothed_2[dim[0]:(2*dim[0]),:] = smoothed
        smoothed = np.swapaxes(smoothed_2, 0, 1)

        extent = [min(gridx)-boxx/2., 2*max(gridx)+boxx/2., min(gridy)-dnu/2., max(gridy)+dnu/2.]
        
        return smoothed, np.array(list(gridx)+list(gridx+dnu)), gridy, extent

    else:
        print('Please check your bounds and values for the echelle diagram.')
        return


def get_ridges(freq, psd, dnu, params, msk, fig, i):

    ech, gridx, gridy, extent = echelle(freq, psd, dnu, params)
    N, M = ech.shape[0], ech.shape[1]
    ech_copy = np.array(list(ech.reshape(-1)))

    n = int(np.ceil(dnu/params['resolution']))

    xax = np.zeros(n)
    yax = np.zeros(n)
    modx = freq%dnu

    start = 0.
    for k in range(n):
        use = np.where((modx >= start)&(modx < start+params['resolution']))[0]
        if len(use) == 0:
            continue
        xax[k] = np.median(modx[use])
        yax[k] = np.mean(psd[use])
        start += params['resolution']

    xax = np.array(list(xax)+list(xax+dnu))
    yax = np.array(list(yax)+list(yax))-min(yax)

    if params['fitbg']['clip']:
        if params['fitbg']['clip_value'] != 0.:
            cut = params['fitbg']['clip_value']
        else:
            cut = np.nanmax(ech_copy) - (np.nanmax(ech_copy)-np.nanmedian(ech_copy))/5.
        print('clipping at %.1f for echelle diagram'%cut)
        ech_copy[ech_copy > cut] = cut
        ech = ech_copy.reshape((N,M))

    # check for nans in echelle diagram before plotting
    first_order = ech[0,:]
    last_order = ech[-1,:]
    first = int(np.sum(np.isnan(first_order)))
    last = int(np.sum(np.isnan(last_order)))

    if first != 0:
        lower_y = gridy[0] + dnu/2.
    else:
        lower_y = min(freq)
    if last != 0:
        upper_y = gridy[-1] - dnu/2.
    else:
        upper_y = max(freq)

    if i == 0 and params['fitbg']['plot']:

        ax8 = fig.add_subplot(3,3,8)
        ax8.imshow(ech, extent = extent, interpolation = 'kaiser', aspect = 'auto', origin = 'lower', cmap = 'jet', norm = PowerNorm(0.5, vmin = np.nanmedian(ech_copy), vmax = np.nanmax(ech_copy), clip = True))
#                ax8.imshow(ech, extent = extent, interpolation = 'kaiser', aspect = 'auto', origin = 'lower', cmap = 'jet')
        ax8.axvline([dnu], color = 'white', linestyle = ':', linewidth = 0.75)
        ax8.set_title(r'$\rm \grave{E}chelle \,\, diagram$', fontsize = 18)
        ax8.set_xlabel(r'$\rm Frequency \,\, mod \,\, %.2f \, \mu Hz$'%dnu, fontsize = 16)
        ax8.set_ylabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 16)
        if 2.*dnu > 100.:
            ax8.xaxis.set_major_locator(MultipleLocator(50))
            ax8.xaxis.set_minor_locator(MultipleLocator(10))
        else:
            ax8.xaxis.set_major_locator(MultipleLocator(10))
            ax8.xaxis.set_minor_locator(MultipleLocator(5))
        ax8.set_xlim([0., 2.*dnu])
        ax8.set_ylim([min(freq[msk]), max(freq[msk])])
        ax8.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
        ax8.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
        plt.tight_layout()
        plt.draw()
        plt.show()

        ax9 = fig.add_subplot(3,3,9)
        ax9.plot(xax, yax, color = 'white', linestyle = '-', linewidth = 1.)
        ax9.set_title(r'$\rm Collapsed \,\, \grave{e}chelle \,\, diagram$', fontsize = 18)
        ax9.set_xlabel(r'$\rm Frequency \,\, mod \,\, %.2f \, \mu Hz$'%dnu, fontsize = 16)
        ax9.set_ylabel(r'$\rm Collapsed \,\, power$', fontsize = 16)
        if 2.*dnu > 100.:
            ax9.xaxis.set_major_locator(MultipleLocator(50))
            ax9.xaxis.set_minor_locator(MultipleLocator(10))
        else:
            ax9.xaxis.set_major_locator(MultipleLocator(10))
            ax9.xaxis.set_minor_locator(MultipleLocator(5))
        ax9.set_xlim([0., 2.*dnu])
        ax9.set_ylim([min(yax), max(yax)])
        ax9.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
        ax9.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
        plt.tight_layout()
        plt.draw()
        plt.show()

        plt.ioff()
        return fig

#def confirm_excess(frequency, power, numax, target, params):
