from __future__ import division

import csv
import os, glob
import subprocess
import matplotlib
import numpy as np
import pandas as pd
from zipfile import ZipFile
from astropy.io import fits, ascii
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
from astropy.timeseries import LombScargle as lomb
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

matplotlib.rcParams['xtick.major.size'] = 7.
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.minor.size'] = 3.5
matplotlib.rcParams['xtick.minor.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 7.
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.minor.size'] = 3.5
matplotlib.rcParams['ytick.minor.width'] = 1
matplotlib.rcParams['axes.formatter.useoffset'] = False
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.rcParams["legend.columnspacing"] = 1.0
matplotlib.rcParams["legend.handletextpad"] = 0.25
matplotlib.rcParams['mathtext.fontset'] = 'stix'

def run_QUAKES(target, inst, sectors = None, guess = True, show = True, restrict = False, ask = False,
               filters = ["1.0", "2.5", "5.0"], colors = ['0.8', 'k', 'r'], verbose = False, chop = False,
               planet = False, manual = False, oversample = False, short_cadence = True, clip = False, 
               presentation = False, bound = None, inset = True, original = False, simulate = False, 
               baseline = None, pdc = True, SYD = True, est = None, zoom = False, filter = True, 
               quality_flag = True, save_files = True, save_plots = True, peaks = 10, fname = None,
               hp_filter = 0.2, lp_filter = 30.):

    if verbose:
        print('')
        print(inst + ' target: ' + str(target))
        print('')

    params = make_dict(target, inst, pdc, planet, manual, short_cadence, sigma = 3.5)
    filter_labels = [r'%s $\mu$Hz'%(filters[0]), r'%s $\mu$Hz'%(filters[1]), r'%s $\mu$Hz'%(filters[2])]
    params.update({'sectors':sectors, 'quality_flag':quality_flag, 'save_plots':save_plots, 'show':show, 
                   'verbose':verbose, 'clip':clip, 'hp_filter':hp_filter/(params['cadence']/60./60./24.), 
                   'lp_filter':lp_filter*24.*60.*60., 'cutoff':1./(lp_filter*24.*60.*60.)/params['nyq'],
                   'simulate':simulate,'baseline':baseline,'bounds':bound,'save_files':save_files,
                   'zoom':zoom,'presentation':presentation,'filter':filter,'chop':chop,'filters':filters,
                   'filter_colors':colors,'filter_labels':filter_labels,'oversample':oversample,
                   'peaks':peaks,'fname':fname})

    if verbose:
        print('target dictionary:')
        print(params)

    totalTime, totalFlux, clipTime, clipFlux, qualityTime, qualityFlux, params = make_light_curve(params)

    if planet:
        time = clipTime[:]
        flux = clipFlux[:]
    else:
        time = totalTime[:]
        flux = totalFlux[:]

    """

    plt.figure(figsize = (10,8))

    plt.plot(time, flux, 'k.')
    plt.tight_layout()
    plt.show()
    plt.close()

    probabilities = [0.05, 0.01, 0.001]
    colors = ['red', 'blue', 'green']
    styles = ['--', ':', '-.']

    ls = lomb(time, flux)

    freq = np.arange(0.01, 0.5, 0.005)

#    frequency, power = ls.autopower(minimum_frequency = 0.01, maximum_frequency = 0.5, samples_per_peak = 10)
    power = ls.power(freq)
    probs = ls.false_alarm_level(probabilities)
    print(probs)

    fap = ls.false_alarm_probability(power.max())
    print(power.max(), fap)

    freq = [1./f for f in freq]

    plt.figure(figsize = (10,8))

    ax = plt.subplot(1,1,1)
    plt.plot(freq, power, 'k-')
    plt.tick_params(labelsize = 16)
    for i in range(3):
        ax.axhline(probs[i], color = colors[i], linestyle = styles[i], linewidth = 1.5, label = r'$\rm p = %s$'%probabilities[i])
    ax.axvline(47.8, color = 'r', linestyle = '-', linewidth = 1.5)
    ax.minorticks_on()
    ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
    ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
    ax.set_xticks([20., 40., 60., 80., 100.])
    ax.set_xticklabels([r'$20$', r'$40$', r'$60$', r'$80$', r'$100$'])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.20])
    ax.set_yticklabels([r'$0.00$', r'$0.05$', r'$0.10$', r'$0.15$', r'$0.20$'])
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.tick_params(labelsize = 20)
    plt.xlabel(r'$\rm Period \,\, [days]$', fontsize = 22)
    plt.ylabel(r'$\rm Power$', fontsize = 22)
    plt.xlim([min(freq),max(freq)])
    plt.ylim([0., 0.205])
    plt.legend(fontsize = 18, handletextpad = 0.75, facecolor = 'w', framealpha = 1.0, markerscale = 2., labelspacing = 0.25, columnspacing = 1.0)
    plt.tight_layout()
    plt.savefig('../Targets/tess/141810080/Plots/TESS_LS_SAP.png', dpi = 150)
    plt.show()
    plt.close()

    """

    params, data = time_to_frequency(time, flux, params)

    if simulate:
        simulate_PS(data, baseline, est, params, verbose, ask, restrict, inset, bound)
    else:
        time = np.array(data['time'])
        flux = np.array(data['flux'])
        frequency = np.array(data['frequency'])
        psd = np.array(data['power'])
        psd1 = np.array(data[filters[0]])
        psd2 = np.array(data[filters[1]])
        psd3 = np.array(data[filters[2]])

        if params['save_files']:
            save_all(np.array(data['time']), np.array(data['flux']), frequency, psd, params)
            save_file(frequency, psd, params['path'] + '/' + str(params['target']) + '_PS.txt', [">14.10f", ">20.10f"])

        plt.figure(figsize = (10,6))
        plt.loglog(frequency, psd, color = colors[0], linestyle = '-', label = 'No filter')
        plt.loglog(frequency, psd1, color = colors[1], linestyle = '-', label = params['filter_labels'][0])
        plt.loglog(frequency, psd2, color = colors[2], linestyle = '-', label = params['filter_labels'][1])
        plt.axhline(params['noise'], color = 'b', linestyle = '--')
        plt.xlabel(r'Frequency ($\mu$Hz)', fontsize = 22)
        plt.ylabel(r'Power Density (ppm$^2$ $\mu$Hz$^{-1}$)', fontsize = 22)
        plt.xlim([1., max(frequency)])
        plt.legend(loc = 'upper right', fontsize = 16, facecolor = 'w', framealpha = 1.0)
        plt.tick_params(labelsize = 16)
        plt.tight_layout()
        if params['save_plots']:
            plt.savefig(params['path'] + '/power_spectrum_loglog.png', dpi = 150)
        if guess:
            plt.show()
        plt.close()

        plt.figure(figsize = (10,6))
        plt.plot(frequency, psd, color = colors[0], linestyle = '-', label = 'No filter')
        plt.plot(frequency, psd1, color = colors[1], linestyle = '-', label = params['filter_labels'][0])
        plt.plot(frequency, psd2, color = colors[2], linestyle = '-', label = params['filter_labels'][1])
        plt.axhline(params['noise'], color = 'b', linestyle = '--')
        plt.xlabel(r'Frequency ($\mu$Hz)', fontsize = 22)
        plt.ylabel(r'Power Density (ppm$^2$ $\mu$Hz$^{-1}$)', fontsize = 22)
        plt.ylim([min(psd), max(psd)])
        plt.xlim([1., max(frequency)])
        plt.legend(loc = 'upper right', fontsize = 16, facecolor = 'w', framealpha = 1.0)
        plt.tick_params(labelsize = 16)
        plt.tight_layout()
        if params['save_plots']:
            plt.savefig(params['path'] + '/power_spectrum.png', dpi = 150)
        if guess:
            plt.show()
        plt.close()

        if guess:
            est = float(input("What is your estimate for nuMax? "))
            print('')
            lower, upper = bounds(est)
            dNu = delta_nu(est)
        else:
            if est is None:
                est = params['star']['nuMax']
            lower, upper = bounds(est)
            dNu = delta_nu(est)

        if verbose:
            print('nuMax = ' + str(est) + ' muHz')
            print('dNu = ' + str(round(dNu,2)) + ' muHz')
            print('')

        if est >= 270:
            if inst == 'tess':
                mask = np.ma.getmask(np.ma.masked_inside(frequency, 200., 3500.))
            if inst == 'k2':
                mask = np.ma.getmask(np.ma.masked_inside(frequency, 200., 4000.))
            if inst == 'kepler':
                mask = np.ma.getmask(np.ma.masked_inside(frequency, 400., 4000.))
            if bound is not None:
                params.update({'x_min':bound[0], 'x_max':bound[1]})
            else:
                params.update({'x_min':300., 'x_max':4000.})
        else:
            mask = np.ma.getmask(np.ma.masked_inside(frequency, 50., 400.))
            if bound is not None:
                params.update({'x_min':bound[0], 'x_max':bound[1]})
            else:
                params.update({'x_min':1., 'x_max':1000.})
            inset = False

        pow = psd[mask]
        pow1 = psd1[mask]        
        params.update({'y_max':1.25*pow.max(), 'y_min':0.75*pow.min(), 'original':original, 
                       'y_max_1':1.25*pow1.max(), 'y_min_1':0.75*pow1.min()})

        if ask:
            restrict = yes_no("Did you want to restrict the oscillation region? ")

        if restrict:
            lower = float(input("What is your lower bound? "))
            upper = float(input("What is your upper bound? "))
            print('')

        if verbose:
            print('Oscillation bounds: [' + str(round(lower,2)) + ', ' + str(round(upper,2)) + ']')
            print('')

        params['star']['nuMax'] = est
        params['star']['dNu'] = dNu
        params['upper'] = upper
        params['lower'] = lower

        lag, auto, start, end = corr(frequency, psd2, params)
        params.update({'start':start, 'end':end, 'inset':inset})

        inset_plot_ACF(params, frequency, psd, psd1, psd2, lag, auto)
#        paper_plot(params, time, flux, old_time, old_flux, frequency, psd, psd1, psd2, lag, auto)

    return


def simulate_PS(data, baseline, est, params, verbose, ask, restrict, inset, bound):

    if params['animate']:
        spectra = {}
        acf = {}

    for x, length in enumerate(baseline):

        params.update({'ext':str(int(length))})
        frequency = np.array(data[length]['frequency'])
        psd = np.array(data[length]['power'])
        psd1 = np.array(data[length][filters[0]])
        psd2 = np.array(data[length][filters[1]])
        psd3 = np.array(data[length][filters[2]])

        lower, upper = bounds(est)
        dNu = delta_nu(est)

        if est >= 270:
            if params['inst'] == 'tess':
                mask = np.ma.getmask(np.ma.masked_inside(frequency, 200., 3500.))
            if params['inst'] == 'k2':
                mask = np.ma.getmask(np.ma.masked_inside(frequency, 200., 4000.))
            if params['inst'] == 'kepler':
                mask = np.ma.getmask(np.ma.masked_inside(frequency, 400., 4000.))
            if bound is not None:
                params.update({'x_min':bound[0], 'x_max':bound[1]})
            else:
                params.update({'x_min':300., 'x_max':4000.})
        else:
            mask = np.ma.getmask(np.ma.masked_inside(frequency, 50., 400.))
            if bound is not None:
                params.update({'x_min':bound[0], 'x_max':bound[1]})
            else:
                params.update({'x_min':1., 'x_max':1000.})
            inset = False

        pow = psd[mask]
        pow1 = psd1[mask]        
        params.update({'y_max':1.15*pow.max(), 'y_min':0.95*pow.min(), 'original':original, 
                       'y_max_1':1.15*pow1.max(), 'y_min_1':0.95*pow1.min()})

        if ask:
            restrict = yes_no("Did you want to restrict the oscillation region? ")

        if restrict:
            lower = float(input("What is your lower bound? "))
            upper = float(input("What is your upper bound? "))
            print('')

        params['star']['nuMax'] = est
        params['star']['dNu'] = dNu
        params['upper'] = upper
        params['lower'] = lower

        lag, auto, start, end = corr(frequency, psd2, params)
        params.update({'start':start, 'end':end, 'inset':inset})

        if params['animate']:
            spectra[x] = {}
            spectra.update({x:{'frequency':frequency,'psd':psd,'psd1':psd1,'psd2':psd2}})
            acf[x] = {}
            acf.update({x:{'lag':lag,'auto':auto}})

        inset_plot_ACF(params, frequency, psd, psd1, psd2, lag, auto)

    if params['animate']:

        width = upper - lower
        x_min = lower - (1./10)*width
        x_max = upper + (9./10)*width

        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_ylabel(r'PSD (ppm$^{2}$/$\mu$Hz)', fontsize = 20)
        ax.set_xlabel(r'Frequency ($\mu$Hz)', fontsize = 20)
        ax.tick_params(axis = 'both', which = 'both', direction = 'inout')
        ax.set_xlim([x_min,x_max])
        ax.tick_params(labelsize = 16)

        line1, = ax.plot([], [], color = params['filter_colors'][0], linestyle = '-')
        line2, = ax.plot([], [], color = params['filter_colors'][1], linestyle = '-')
        line3, = ax.plot([], [], color = params['filter_colors'][2], linestyle = '-')
        title_text = ax.set_title('', fontsize = 24)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            title_text.set_text('')
            return line1, line2, line3, title_text

        def animate(i):
            line1.set_data(spectra[i]['frequency'], spectra[i]['psd'])
            line2.set_data(spectra[i]['frequency'], spectra[i]['psd1'])
            line3.set_data(spectra[i]['frequency'], spectra[i]['psd2'])
            title_text.set_text('Baseline = ' + str(baseline[i]) + ' days')
            return line1, line2, line3, title_text

        ani = animation.FuncAnimation(fig, animate, frames = len(baseline), interval = 2, 
                                      blit = True, init_func = init)
        plt.tight_layout()
        filename = params['path'] + '/' + params['target'] + '.mp4'
        ani.save(filename)
        plt.show()

    return


def save_file(x, y, f_name, formats, mask = None):

    if mask is not None:
        mask_1 = np.ma.getmask(np.ma.masked_inside(x, 60., 4167.))
        x = x[mask_1]
        y = y[mask_1]

    with open(f_name, "w") as f:
        for xx, yy in zip(x, y):
            values = [xx, yy]
            text = '{:{}}'*len(values) + '\n'
            fmt = sum(zip(values, formats), ())
            f.write(text.format(*fmt))
    f.close()
    print(f_name + ' file written.')
    
    return


def save_all(time, flux, frequency, psd, params, SYD = True, mask = True):

    if mask:
        mask_1 = np.ma.getmask(np.ma.masked_inside(frequency, 0., 300.))
        frequency = frequency[mask_1]
        psd = psd[mask_1]

    if params['pdc']:
        lc_file = params['path'] + '/' + str(params['target']) + '_PDC_LC.txt'
        lc_formats = [">15.8f", ">25.16f"]
        save_file(time, flux, lc_file, lc_formats)

        if params['oversample']:
            fft_file = params['path'] + '/' + str(params['target']) + '_PDC_PS_OS.txt'
        else:
            if params['fname'] is not None:
                fft_file = params['fname']
            else:
                fft_file = params['path'] + '/' + str(params['target']) + '_PDC_PS.txt'
        fft_formats = [">14.10f", ">20.10f"]
        save_file(frequency, psd, fft_file, fft_formats)
        fft_formats = [">15.8f", ">18.10e"]

        if SYD:

            if params['sectors'] is not None:

                lc_file = params['path'] + '/' + str(params['target']) + '_' + str(int(params['sectors'])) + '_PDC.dat.ts'
                save_file(time, flux, lc_file, lc_formats)
                fft_file = params['path'] + '/' + str(params['target']) + '_' + str(int(params['sectors'])) + '_PDC.dat.ts.fft'
                save_file(frequency, psd, fft_file, fft_formats)
                zip_file = params['path'] + '/' + str(params['target']) + '_' + str(int(params['sectors'])) + '_PDC.zip'

            else:

                lc_file = params['path'] + '/' + str(params['target']) + '_PDC.dat.ts'
                save_file(time, flux, lc_file, lc_formats)
                fft_file = params['path'] + '/' + str(params['target']) + '_PDC.dat.ts.fft'
                save_file(frequency, psd, fft_file, fft_formats)
                zip_file = params['path'] + '/' + str(params['target']) + '_PDC.zip'

            list_file = fft_file + 'list'
            with open(list_file, "w") as f:
                f.write(lc_file.split('/')[-1] + '\n')
                f.write(fft_file.split('/')[-1])

            with ZipFile(zip_file, "w") as zipObj:
                zipObj.write(list_file)
                zipObj.write(lc_file)
                zipObj.write(fft_file)

            with subprocess.Popen(["scp", zip_file, 'achontos@sarek1.ifa.hawaii.edu:Documents/pipeline/testdir_SC']) as p:
                sts = os.waitpid(p.pid, 0)
            os.remove(zip_file)
            os.remove(lc_file)
            os.remove(fft_file)
            os.remove(list_file)
            print()

    else:
        lc_file = params['path'] + '/' + str(params['target']) + '_SAP_LC.txt'
        lc_formats = [">15.8f", ">25.16f"]
        save_file(time, flux, lc_file, lc_formats)

        if params['oversample']:
            fft_file = params['path'] + '/' + str(params['target']) + '_SAP_PS_OS.txt'
        else:
            if params['fname'] is not None:
                fft_file = params['fname']
            else:
                fft_file = params['path'] + '/' + str(params['target']) + '_SAP_PS.txt'
        fft_formats = [">14.10f", ">20.10f"]
        save_file(frequency, psd, fft_file, fft_formats)
        fft_formats = [">15.8f", ">18.10e"]

        if SYD:

            if params['sectors'] is not None:

                lc_file = params['path'] + '/' + str(params['target']) + '_' + str(int(params['sectors'])) + '_SAP.dat.ts'
                save_file(time, flux, lc_file, lc_formats)
                fft_file = params['path'] + '/' + str(params['target']) + '_' + str(int(params['sectors'])) + '_SAP.dat.ts.fft'
                save_file(frequency, psd, fft_file, fft_formats)
                zip_file = params['path'] + '/' + str(params['target']) + '_' + str(int(params['sectors'])) + '_SAP.zip'

            else:

                lc_file = params['path'] + '/' + str(params['target']) + '_SAP.dat.ts'
                save_file(time, flux, lc_file, lc_formats)
                fft_file = params['path'] + '/' + str(params['target']) + '_SAP.dat.ts.fft'
                save_file(frequency, psd, fft_file, fft_formats)
                zip_file = params['path'] + '/' + str(params['target']) + '_SAP.zip'

            list_file = fft_file + 'list'
            with open(list_file, "w") as f:
                f.write(lc_file.split('/')[-1] + '\n')
                f.write(fft_file.split('/')[-1])

            with ZipFile(zip_file, "w") as zipObj:
                zipObj.write(list_file)
                zipObj.write(lc_file)
                zipObj.write(fft_file)

            with subprocess.Popen(["scp", zip_file, 'achontos@sarek1.ifa.hawaii.edu:Documents/pipeline/testdir_SC']) as p:
                sts = os.waitpid(p.pid, 0)
            os.remove(zip_file)
            os.remove(lc_file)
            os.remove(fft_file)
            os.remove(list_file)
            print()

    return


def ask_int(question):    

    while True:
        answer = input(question)
        try:
            if float(answer).is_integer():
                print()
                break
            else:
                print()
                print("Need an integer.")
                print()
        except ValueError:
            print()
            print("Did not understand that input.")
            print()
    
    return int(answer)
    
    
############################################################################################################
#                                                                                                          #
#                                              DICTIONARIES                                                #
#                                                                                                          #
############################################################################################################
 

def set_planet_dict(target, inst, manual, vars = ["periods", "periods_err", "epochs", "epochs_err", 
                    "durations", "durations_err"]):

    if manual:
        periods = [0.73665]
        periods_err = [0.00018]
        epochs = [1872.16257]
        epochs_err = [0.001101]
        durations = [1.459/24.]
        durations_err = [0.22/24.]

        vals = [periods, periods_err, epochs, epochs_err, durations, durations_err]

        planet_dict = {}
        planet_dict.update(dict(zip(vars,vals)))

    else:
        planet_dict = {}
        df = pd.read_csv('../Info/TOIs.csv')
        table = df.loc[df['TIC'] == target].index.values.tolist()
        cols = df.columns.values.tolist()
        for variable in vars:
            if variable in cols:
                if variable == 'durations' or variable == 'durations_err':
                    dur = df.loc[table][variable].values.tolist()
                    dur = [d/24. for d in dur]
                    planet_dict.update({variable:dur})
                else:
                    planet_dict.update({variable:df.loc[table][variable].values.tolist()})
            else:
                array = np.empty(len(table))
                array[:] = np.nan
                planet_dict.update({variable:array})

    return planet_dict


def set_star_dict(target, inst, manual, vars = ["srad", "srad err", "Ms", "lum", "teff", "teff_err", "dNu", "dNu_err",
                  "nuMax", "nuMax_err", "logg", "logg_err"]):

    if manual:
        rstar = 0.96
        rstar_err = 0.05
        mstar = 0.9
        mstar_err = 0.05
        lum = 0.63
        lum_err = 0.55
        logg = 4.42
        logg_err = 0.08
        rho = None
        rho_err = None
        teff = 5250.
        teff_err = 75.
        dNu = 130.
        dNu_err = 0.55
        nuMax = 3150.
        nuMax_err = 30.0

        vars = ["rstar", "rstar_err", "mstar", "mstar_err", "lum", "lum_err", "rho", "rho_err", "teff", 
                "teff_err", "dNu", "dNu_err", "nuMax", "nuMax_err", "logg", "logg_err"]
        vals = [rstar, rstar_err, mstar, mstar_err, lum, lum_err, rho, rho_err, teff, teff_err, dNu, 
                dNu_err, nuMax, nuMax_err, logg, logg_err]

        star_dict = dict(zip(vars,vals))
        params = seismic_parameters(inst, star_dict)
        star_dict.update(params)

    else:
        star_dict = {}
        path_planet = '../Info/'+inst+'/'+inst+'.csv'
        path_seismic = '../Info/'+inst+'/'+inst+'_seismic.csv'

        if os.path.exists(path_planet) and os.path.exists(path_seismic):
            path = path_seismic
            df = pd.read_csv(path)
            if target not in df['target']:
                path = path_planet
        elif os.path.exists(path_planet) or os.path.exists(path_seismic):
            if os.path.exists(path_planet):
                path = path_planet
            else:
                path = path_seismic
        else:
            path = None

        if path is not None:
            df = pd.read_csv('../Info/TOIs.csv')
            table = df.loc[df['TIC'] == target].index.values.tolist()
            index = table[0]
            cols = df.columns.values.tolist()
            for variable in vars:
                if variable in cols:
                    star_dict.update({variable:df.loc[index][variable]})
                else:
                    star_dict.update({variable:np.nan})
        else:
            for variable in vars:
                star_dict.update({variable:np.nan})

    return star_dict
   

def make_dict(target, inst, pdc, planet, manual, short_cadence, sigma = 2.5, 
              binning = 10., folded = True, multiplier = 1.5, delta_t = 2., 
              rms_timescale = 60, secondary = False, order = 2):

    planet_dict = set_planet_dict(target, inst, manual)
    star_dict = set_star_dict(target, inst, manual)
    
    if inst == 'tess':
        if short_cadence:
            lcs = '/*s_lc.fits'
            cadence = 120.
        else:
            cadence = 1800.
        if pdc:
            flux = 'PDCSAP_FLUX'
        else:
            flux = 'SAP_FLUX'
        quality = 'QUALITY'

    if inst == 'kepler':
        if short_cadence:
            lcs = '/*_slc.fits'
            cadence = 60.
        else:
            lcs = '/*_llc.fits'
            cadence = 1800.
        if pdc:
            flux = 'PDCSAP_FLUX'
        else:
            flux = 'SAP_FLUX'
        quality = 'SAP_QUALITY'

    if inst == 'k2':
        if short_cadence:
            cadence = 60.
        else:
            cadence = 1800.
        lcs = '/*.fits'
        flux = 'PDCSAP_FLUX'
        quality = 'SAP_QUALITY'

    nyq = 0.5*(1./cadence)

    path = '../Targets/' + inst + '/' + str(target)
    if not os.path.exists(path):
        os.makedirs(path)
    
    values = [target, sigma, binning, short_cadence, folded, path, secondary, delta_t, inst, 
              planet, rms_timescale, quality, pdc, lcs, flux, order, cadence, nyq]
              
    variables = ["target", "sigma", "binning", "short_cadence", "folded", "path", "secondary",
                 "delta_t", "inst", "planet", "rms_timescale", "quality", "pdc", "lcs", "flux", 
                 "order", "cadence", "nyq"]
                 
    params = dict(zip(variables,values))
    params.update({'planets':planet_dict, 'star':star_dict})
    print(params)
    
    return params



##########################################################################################
#                                                                                        #
#                                      TIME TOOLS                                        #
#                                                                                        #
##########################################################################################


def get_light_curve(path):

    time = list()
    flux = list()
    with open(path, "r") as file:
        for line in file:
            time.append(float(line.strip().split()[0]))
            flux.append(float(line.strip().split()[1]))

    return np.array(time), np.array(flux)


def make_light_curve(params):
    
    files = glob.glob(params['path'] + params['lcs'])
    files.sort()

    totalTime = []
    totalFlux = []
    clipTime = []
    clipFlux = []
    rawTime = []
    rawFlux = []
    qualityTime = []
    qualityFlux = []

    if params['sectors'] is not None:
        files = files[:int(params['sectors'])]

    stamps = []

    first = fits.open(files[0])
    hdr = first[1].columns

    for each in files:

        temp = fits.open(each)
        lightcurve = temp[1].data
        when = lightcurve.field('TIME')
        sapflux = lightcurve.field(params['flux'])
        if params['inst'] == 'k2':
            when = np.array([(t+2.4e6)%2454833 for t in when])
            sapflux = np.array([f*10.**-6 + 1. for f in sapflux])

        if params['quality_flag']:
            quality = lightcurve.field(params['quality'])
            good = np.ma.getmask(np.ma.masked_where(quality == 0, quality))
            invalid = np.ma.getmask(np.ma.masked_invalid(sapflux))
            valid = ~invalid
            time = when[good*valid]
            flux = sapflux[good*valid]
        else:
            time = when[:]
            flux = sapflux[:]

        Time = time[:]
        Flux = flux[:]

        if params['planet'] and params['clip']:
            for i in range(len(params['planets']['periods'])):
                correctedTime, correctedFlux = transit_clip(Time, Flux, params['planets']['periods'][i], params['planets']['epochs'][i], params['planets']['durations'][i])
                Time = correctedTime[:]
                Flux = correctedFlux[:]
               
        z = np.polyfit(Time,Flux,1)
        p = np.poly1d(z)
        normalFlux = flux/p(time)
        normalizedFlux = Flux/p(Time)
            
        totalTime = np.hstack((totalTime,time))
        totalFlux = np.hstack((totalFlux,normalFlux))
        clipTime = np.hstack((clipTime,Time))
        clipFlux = np.hstack((clipFlux,normalizedFlux))
        rawTime = np.hstack((rawTime,when))
        rawFlux = np.hstack((rawFlux,sapflux))
        qualityTime = np.hstack((qualityTime,time))
        qualityFlux = np.hstack((qualityFlux,flux))

        stamps.append(when[-1])

    stamps.insert(0, rawTime[0])

    plt.figure(figsize = (10,6))
    plt.scatter(rawTime, rawFlux, color = 'r', marker = 'o', s = 5.)
    plt.scatter(qualityTime, qualityFlux, color = 'k', marker = 'o', s = 5.)
    plt.xlim([min(rawTime), max(rawTime)])
    plt.tight_layout()
    if params['save_plots']:
        if params['pdc']:
            plt.savefig(params['path'] + '/raw_light_curve_PDC.png', dpi = 150)
        else:
            plt.savefig(params['path'] + '/raw_light_curve_SAP.png', dpi = 150)
    plt.tick_params(labelsize = 16)
    if params['show']:
        plt.show()
    plt.close()
    
    Time, Flux = sigma_clip(totalTime, totalFlux, params)
    time, flux = sigma_clip(clipTime, clipFlux, params)

    totalTime, totalFlux = sort_LC(Time, Flux)
    clipTime, clipFlux = sort_LC(time, flux)

    if params['planet']:

        plt.figure(figsize = (10,6))
        plt.scatter(totalTime, totalFlux, color = 'r', marker = 'o', s = 5.)
        plt.scatter(clipTime, clipFlux, color = 'k', marker = 'o', s = 5.)
        plt.xlim([min(totalTime), max(totalTime)])
        plt.tight_layout()
        if params['save_plots']:
            if params['pdc']:
                plt.savefig(params['path'] + '/clipped_light_curve_PDC.png', dpi = 150)
            else:
                plt.savefig(params['path'] + '/clipped_light_curve_SAP.png', dpi = 150)
        plt.tick_params(labelsize = 16)
        if params['show']:
            plt.show()
        plt.close()

        fold = max(params['planets']['periods'])

        plt.figure(figsize = (10,6))
        plt.scatter(totalTime%fold, totalFlux, color = 'r', marker = 'o', s = 5.)
        plt.scatter(clipTime%fold, clipFlux, color = 'k', marker = 'o', s = 5.)
        plt.xlim([0., fold])
        plt.tight_layout()
        if params['save_plots']:
            if params['pdc']:
                plt.savefig(params['path'] + '/folded_clipped_light_curve_PDC.png', dpi = 150)
            else:
                plt.savefig(params['path'] + '/folded_clipped_light_curve_SAP.png', dpi = 150)
        plt.tick_params(labelsize = 16)
        if params['show']:
            plt.show()
        plt.close()

    params.update({'stamps':stamps})
      
    return totalTime, totalFlux, clipTime, clipFlux, qualityTime, qualityFlux, params


def transit_clip(time, flux, period, epoch, duration):

    primary = np.ma.getmask(np.ma.masked_inside(time%period, (epoch - duration/2.)%period, (epoch + duration/2.)%period))
    
    if not np.sum(primary):
        totalTime = time[:]
        totalFlux = flux[:]
    else:
        totalTime = time[~primary]
        totalFlux = flux[~primary]

    return totalTime, totalFlux


def sigma_clip(time, flux, params):
    
    sigma_mask = np.zeros_like(time)
    
    begin = 0
    start = time[begin]
    
    for i in range(len(time)):

        if time[i] > start + params['delta_t']:

            temp_time = time[begin:i]
            temp_flux = flux[begin:i]
            avg = np.mean(temp_flux)
            sd = np.std(temp_flux)
    
            temp_mask = np.ma.getmask(np.ma.masked_inside(temp_flux, avg - params['sigma']*sd, avg + params['sigma']*sd))

            sigma_mask[begin:i] = temp_mask
            begin = i
            start = time[begin]
    
    temp_time = time[begin::]
    temp_flux = flux[begin::]
    avg = np.mean(temp_flux)
    sd = np.std(temp_flux)
    
    temp_mask = np.ma.getmask(np.ma.masked_inside(temp_flux, avg - params['sigma']*sd, avg + params['sigma']*sd))
    sigma_mask[begin::] = temp_mask
    sigma_mask = np.array(sigma_mask, dtype = bool)
    
    totalTime = time[sigma_mask]
    totalFlux = flux[sigma_mask]
    
    return totalTime, totalFlux
    
    
def chop_LC(time, flux, params):

    if params['short_cadence']:
        if params['inst'] == 'tess':
            if params['target'] == 136916387:
                indices = [i for i, X in enumerate(time) if (X < 1631.34) or
                           (X > 1631.89 and X < 1650.80) or
                           (X > 1650.98 and X < 1652.70)]
            elif params['target'] == 138017750:
                if params['pdc']:
                    indices = [i for i, X in enumerate(time) if (X < 1791.07) or
                               (X > 1791.46 and X < 1794.78) or
                               (X > 1795.01 and X < 1800.19) or
                               (X > 1803.77 and X < 1809.72) or
                               (X > 1810.)]
                else:
                    indices = [i for i, X in enumerate(time) if (X > 1791.63)]
            elif params['target'] == 141810080 and params['pdc']:
                indices = [i for i, X in enumerate(time) if (X < 1346.73) or
                           (X > 1350.01 and X < 1420.00) or
                           (X > 1421.51 and X < 1465.00) or
                           (X > 1491.87 and X < 1516.47) or
                           (X > 1544.62)]
            elif params['target'] == 141810080 and not params['pdc']:
                indices = [i for i, X in enumerate(time) if 
                           (X > 1354.00 and X < 1407.00) or
                           (X > 1425.46 and X < 1465.00) or
                           (X > 1491.22 and X < 1530.00) or
                           (X > 1544.00)]
            elif params['target'] == 229940491:
                indices = [i for i, X in enumerate(time) if #(X < 1687.63) or
#                           (X > 1697.00 and X < 1701.93) or
                           (X > 1711.00 and X < 1719.54) or
                           (X > 1724.65 and X < 1733.40) or
                           (X > 1738.38 and X < 1748.23) or
                           (X > 1751.00 and X < 1761.00) or
                           (X > 1764.00 and X < 1790.50) or
                           (X > 1792.01)]
            elif params['target'] == 233059608:
                indices = [i for i, X in enumerate(time) if (X < 1696.) or
                           (X > 1698. and X < 1750.53) or
                           (X > 1816.63 and X < 1842.)]
            elif params['target'] == 287776397:
                indices = [i for i, X in enumerate(time) if (X > 1409.)]
            elif params['target'] == 332064670 and params['pdc']:
                indices = [i for i, X in enumerate(time) if (X < 1882.) or
                           (X > 1883.)]
            elif params['target'] == 332064670 and not params['pdc']:
                indices = [i for i, X in enumerate(time) if (X < 1896.33) or
                           (X > 1897.45)]
            elif params['target'] == 349790953:
                indices = [i for i, X in enumerate(time) if (X < 1337.42) or
                           (X > 1340.81 and X < 1346.97) or
                           (X > 1355.00 and X < 1393.18) or
                           (X > 1398.00 and X < 1408.00) or
                           (X > 1340.81 and X < 1346.97) or
                           (X > 1426.00 and X < 1436.36) or
                           (X > 1340.81 and X < 1346.97) or
                           (X > 1439.51 and X < 1449.79) or
                           (X > 1452.92 and X < 1463.53) or
                           (X > 1480.00 and X < 1492.22) or
                           (X > 1494.01 and X < 1502.97) or
                           (X > 1505.45 and X < 1516.00) or
                           (X > 1545.85 and X < 1556.00) or
                           (X > 1559.29 and X < 1584.00) or
                           (X > 1584.90 and X < 1667.00) or
                           (X > 1668.82)]
            else:
                indices = [i for i, X in enumerate(time) if (X < 1338.46) or
                           (X > 1339.73 and X < 1345.92) or
                           (X > 1350.07 and X < 1353.04) or
                           (X > 1354.20 and X < 1361.62) or
                           (X > 1361.66 and X < 1367.08) or
                           (X > 1368.67 and X < 1376.10) or
                           (X > 1376.21 and X < 1381.44) or
                           (X > 1386.09 and X < 1389.80) or
                           (X > 1390.38 and X < 1392.58) or
                           (X > 1392.97 and X < 1395.09) or
                           (X > 1396.74 and X < 1398.65) or
                           (X > 1398.76 and X < 1400.54) or
                           (X > 1400.84 and X < 1402.62) or
                           (X > 1402.87 and X < 1403.91) or
                           (X > 1411.01 and X < 1416.91) or
                           (X > 1417.22 and X < 1418.00) or
#                       (X > 1421.28 and X < 1423.44) or
#                       (X > 1424.61 and X < 1436.75) or
#                       (X > 1438.13 and X < 1440.80) or
                           (X > 1425.00 and X < 1436.00) or
                           (X > 1441.80 and X < 1443.83) or
                           (X > 1444.25 and X < 1449.93) or
                           (X > 1451.65 and X < 1463.88) or
                           (X > 1491.71 and X < 1502.96) or
                           (X > 1504.79 and X < 1516.00) or
                           (X > 1518.00 and X < 1526.96) or
                           (X > 1545.11 and X < 1549.37) or
                           (X > 1549.59 and X < 1555.45) or
                           (X > 1558.34 and X < 1558.49) or
                           (X > 1559.15 and X < 1559.81) or
                           (X > 1560.04 and X < 1566.06) or
                           (X > 1566.28 and X < 1568.36) or
                           (X > 1571.12 and X < 1581.69) or
                           (X > 1584.80 and X < 1595.61) or
                           (X > 1600.00 and X < 1609.63) or
                           (X > 1612.45 and X < 1623.82)]
        else:
            if params['inst'] == 'kepler':
                if params['target'] == 8866102:
                    indices = [i for i, X in enumerate(time) if (X < 157.7) or
                               (X > 157.95 and X < 164.75) or
                               (X > 260.81 and X < 282.88) or
                               (X > 282.97 and X < 349.37) or
                               (X > 352.53 and X < 442.05) or
                               (X > 444.02 and X < 537.47) or
                               (X > 539.61 and X < 629.17) or
                               (X > 630.30 and X < 760.88) or
                               (X > 732.96 and X < 760.87) or
                               (X > 763.95 and X < 782.07) or
                               (X > 803.00 and X < 844.83) or
                               (X > 845.69 and X < 1093.50) or
                               (X > 1099.56 and X < 1121.50) or
                               (X > 1122.36 and X < 1150.43) or
                               (X > 1151.20 and X < 1181.33) or
                               (X > 1182.85 and X < 1185.19) or
                               (X > 1188.07 and X < 1289.34) or
                               (X > 1296.55 and X < 1371.21) or
                               (X > 1375.17 and X < 1412.68) or
                               (X > 1418.59)]
                else:
                    indices = [i for i, X in enumerate(time) if 
                               (X > 170. and X < 181.4) or
                               (X > 188 and X < 230.2) or
                               (X > 233.0 and X < 257.0) or
                               (X > 414.6 and X < 435.75) or
                               (X > 630.5 and X < 660.0) or
                               (X > 661.15 and X < 690.0) or
                               (X > 691.3 and X < 718.9) or
                               (X > 736.2 and X < 752.8) or
                               (X > 753.31 and X < 760.3) or
                               (X > 764.1 and X < 781.9) or
                               (X > 782.5 and X < 1000.00) or
                               (X > 1001.44)]
    else:
        if params['target'] == 10018357:
            indices = [i for i, X in enumerate(time) if (X < 162.) or
                       (X > 171.787 and X < 180.312) or
                       (X > 185.553 and X < 382.007) or
                       (X > 386.007 and X < 393.623) or
                       (X > 402.302 and X < 410.818) or
                       (X > 414.339 and X < 475.000) or
                       (X > 478.353 and X < 536.102) or
                       (X > 542.244 and X < 717.969) or
                       (X > 737.375 and X < 759.402) or
                       (X > 765.789 and X < 1094.00) or
                       (X > 1101.03 and X < 1124.53) or
                       (X > 1128.96 and X < 1151.88) or
                       (X > 1188.00 and X < 1213.54) or
                       (X > 1217.55 and X < 1271.43) or
                       (X > 1277.08 and X < 1335.05) or
                       (X > 1339.22 and X < 1369.57) or
                       (X > 1374.79 and X < 1469.34) or
                       (X > 1490.80 and X < 1522.81) or
                       (X > 1528.38 and X < 1579.81)]
        else:
            indices = [i for i, X in enumerate(time) if (X < 180.9) or
                      (X > 186.6 and X < 229.5) or
                      (X > 233.75 and X < 258.5) or
                      (X > 294.8 and X < 350.0) or
                      (X > 367.8 and X < 372.2) or
                      (X > 378.2 and X < 382.2) or
                      (X > 386.95 and X < 426.4) or
                      (X > 444.15 and X < 530.6) or
                      (X > 630.0 and X < 675.5) or
                      (X > 694.0 and X < 719.75) or
                      (X > 739.5 and X < 758.9) or
                      (X > 764.45 and X < 787.6) or
                      (X > 808.0 and X < 950.0) or
                      (X > 1009.8 and X < 1061.2) or
                      (X > 1065.5 and X < 1069.8) or
                      (X > 1074.25 and X < 1158.6) or
                      (X > 1165.76 and X < 1180.7) or
                      (X > 1195.6 and X < 1215.0) or
                      (X > 1216.3 and X < 1236.6) or
                      (X > 1374.84 and X < 1412.5) or
                      (X > 1447.0 and X < 1584.0)]
                   
    newTime = time[indices]
    newFlux = flux[indices]
    
    return newTime, newFlux


def sort_LC(time, flux):

    s = np.argsort(time)
    time = time[s]
    flux = flux[s]
    
    return time, flux


##########################################################################################
#                                                                                        #
#                                    FREQUENCY TOOLS                                     #
#                                                                                        #
##########################################################################################


def get_spectrum(path, filters):

    freq = list()
    psd = list()
    with open(path, "r") as file:
        for line in file:
            freq.append(float(line.strip().split()[0]))
            psd.append(float(line.strip().split()[1]))

    freq = np.array(freq)
    psd = np.array(psd)

    if filters is not None:
        resolution = freq[1] - freq[0]
        boxsize_1 = np.ceil(float(filters[0])/resolution)
        box_kernel_1 = Box1DKernel(boxsize_1)
        psd_1 = (convolve(psd, box_kernel_1))

        boxsize_2 = np.ceil(float(filters[1])/resolution)
        box_kernel_2 = Box1DKernel(boxsize_2)
        psd_2 = (convolve(psd, box_kernel_2))

        if len(filters) == 2:
            psd_3 = list()
        else:
            boxsize_3 = np.ceil(float(filters[2])/resolution)
            box_kernel_3 = Box1DKernel(boxsize_3)
            psd_3 = (convolve(psd, box_kernel_3))
    else:
        psd_1 = list()
        psd_2 = list()
        psd_3 = list()

    return freq, psd, np.array(psd_1), np.array(psd_2), np.array(psd_3)
  

def seismic_parameters(inst, star_dict):

    # solar parameters
    teff_solar = 5777.0
    teffred_solar = 8907.0
    logg_solar = 4.4
    numax_solar = 3090.0
    dnu_solar = 135.1

    if inst == 'kepler' or inst == 'k2':
        cadence = 60
    if inst == 'tess':
	       cadence = 120

    vnyq = (1.0/(2.0*cadence))*10**6
    teffred = teffred_solar*((star_dict['rstar'])**2*(star_dict['teff']/teff_solar)**4)

    nuMax = numax_solar*(star_dict['rstar']**-1.85)*((star_dict['teff']/teff_solar)**0.92)
    dNu = dnu_solar*(star_dict['rstar']**-1.42)*((star_dict['teff']/teff_solar)**0.71)

    vars = ["vnyq", "nuMax", "dNu", "teff_red", "teff_solar", "teffred_solar", "numax_solar", 
            "dnu_solar", "cadence"]
    vals = [vnyq, nuMax, dNu, teffred, teff_solar, teffred_solar, numax_solar, dnu_solar, 
            cadence]

    params = dict(zip(vars,vals))

    return params
    
    
def delta_nu(nuMax):
    
    return 0.22*(nuMax**0.797)
    

def high_pass_filter(flux, boxsize):

    box_kernel = Box1DKernel(boxsize)
    smoothed_flux = (convolve(flux-1., box_kernel))+1.

    return smoothed_flux


def low_pass_filter(flux, params):

    b, a = butter(params['order'], params['cutoff'], btype = 'low', analog = False)
    smoothed_flux = filtfilt(b, a, flux)

    return smoothed_flux
	
	
def bounds(nuMax):
    
    width = 1300.*(nuMax/3090.)
    w = width/2.
    lower = nuMax - w
    upper = nuMax + w
    
    return lower, upper
    
    
def corr(frequency, power, params):

    resolution = frequency[1] - frequency[0]
    
    frequency = np.array(frequency)
    power = np.array(power)

    mask = np.ma.getmask(np.ma.masked_inside(frequency, params['lower'], params['upper']))
    
    freq = frequency[mask]
    pow = power[mask]
    
    f = freq[:]
    p = pow[:]
        
    n = len(p)
    mean = np.mean(p)
    var = np.var(p)   
    N = np.arange(n)
    
    lag = N*resolution
    
    auto = np.correlate(p - mean, p - mean, "full")    
    auto = auto[int(auto.size/2):]
    
    start = params['star']['dNu']/2. - 10.
    if start < 0:
        start = 0
    end = 2.*params['star']['dNu'] + 10.

    mask = np.ma.getmask(np.ma.masked_inside(lag, start, end))
    
    l = lag[mask]
    a = auto[mask]
    
    lag = l[:]
    auto = a[:]
    
    return lag, auto, start, end


def fast_fourier_transform(time, flux, params):

    nyquist = (1./(2.*params['cadence']))*60.*60.*24.
    if params['verbose']:
        print()
        print("Maximum frequency = %d c/d"%nyquist)

    if params['oversample']:
        frequency, power = lomb(time, flux).autopower(method = 'fast', samples_per_peak = params['peaks'], maximum_frequency = nyquist)
    else:
        frequency, power = lomb(time, flux).autopower(method = 'fast', samples_per_peak = 1, maximum_frequency = nyquist)
	
    conversion = 10.**6/(24.*60.*60.)
    frequency = [each*conversion for each in frequency]
	
    scaling = frequency[1] - frequency[0]
    params.update({'resolution':scaling})

    if params['verbose']:
        print()
        print('Frequency resolution:')
        print(params['resolution'])
        print()

    data = {}
    data['frequency'] = frequency
    data['power'] = power
	
    for filter in params['filters']:
	
	       boxsize = np.ceil(float(filter)/scaling)
	       box_kernel = Box1DKernel(boxsize)
	       data[filter] = (convolve(power, box_kernel))
		
    return data
	
	
def time_to_frequency(time, flux, params):
	
    smoothed_flux = high_pass_filter(flux, params['hp_filter'])

    plt.figure(figsize = (10,6))
    plt.scatter(time, flux, color = 'blue', marker = '.', s = 50.)
    plt.plot(time, smoothed_flux, color = 'orange', linestyle = '-', linewidth = 2.)
    plt.xlim([min(time), max(time)])
    if params['inst'] == 'tess':
        plt.xlabel('BJD - 2457000 (days)', fontsize = 20)
    else:
        plt.xlabel('BJD - 2454833 (days)', fontsize = 20)
    plt.ylabel('Normalized flux', fontsize = 20)
    plt.tick_params(labelsize = 16)
    plt.xlim(min(time),max(time))
    plt.tight_layout()
    if params['save_plots']:
        if params['pdc']:
            plt.savefig(params['path'] + '/filtered_light_curve_PDC.png', dpi = 150)
        else:
            plt.savefig(params['path'] + '/filtered_light_curve_SAP.png', dpi = 150)
    if params['show']:
        plt.show()
    plt.close()

    if params['filter']:
        filter_flux = flux/smoothed_flux
    else:
        filter_flux = flux[:]

    if params['chop']:
        totalTime, finalFlux = chop_LC(time, filter_flux, params)
    else:
        totalTime = time[:]
        finalFlux = filter_flux[:]

    plt.figure(figsize = (10,6))
    plt.scatter(totalTime, finalFlux, color = 'b', marker = '.', s = 2.5)
    plt.xlim([min(totalTime), max(totalTime)])
    if params['inst'] == 'tess':
        plt.xlabel('BJD - 2457000 (days)', fontsize = 20)
    else:
        plt.xlabel('BJD - 2454833 (days)', fontsize = 20)
    plt.ylabel('Normalized flux', fontsize = 20)
    plt.tick_params(labelsize = 16)
    plt.tight_layout()
    if params['save_plots']:
        if params['pdc']:
            plt.savefig(params['path'] + '/pre_fft_light_curve_PDC.png', dpi = 150)
        else:
            plt.savefig(params['path'] + '/pre_fft_light_curve_SAP.png', dpi = 150)
    if params['show']:
        plt.show()
    plt.close()

    if params['simulate']:

        df = {}
        cad = np.nanmedian((np.diff(time)))
        length = int(np.ceil(cad*len(time)))
        baseline = params['baseline']
        baseline.insert(0, length)
        print(str(length) + ' days:')
        print(str(len(time)) + ' data points')
        print()

        for length in baseline:
            print(str(length) + ' days:')
            idx = int(np.ceil(length/cad))
            print(str(idx) + ' data points')
            print()
            data = fast_fourier_transform(time[:idx], flux[:idx], params)
            for key in [k for k in data.keys()]:
                if key == 'frequency':
                    pass
                else:
                    data[key] = 4.*data[key]*np.var(flux[:idx]*1e6)/(np.sum(data[key])*params['resolution'])
            df[length] = data

        return params, df

    else:
	
        data = fast_fourier_transform(totalTime, finalFlux, params)
    
        for key in [k for k in data.keys()]:
            if key == 'frequency':
                pass
            else:
                data[key] = 4.*data[key]*np.var(flux*1e6)/(np.sum(data[key])*params['resolution'])

        params = add_noise(data['frequency'], data['power'], params)
        data['time'] = totalTime
        data['flux'] = finalFlux
		
        return params, data


def add_noise(frequency, power, params):

    if params['inst'] == 'tess':
        mask = np.ma.getmask(np.ma.masked_inside(frequency, 3800., 4000.))
    else:
        mask = np.ma.getmask(np.ma.masked_inside(frequency, 7300., 7500.))

    pow = power[mask]
    noise = np.mean(pow)

    params['noise'] = noise
	
    return params


##########################################################################################
#                                                                                        #
#                                      PLOT TOOLS                                        #
#                                                                                        #
##########################################################################################


def paper_plot(params, time, flux, old_time, old_flux, frequency, psd, psd1, psd2, lag, auto):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    colors = ['darkorange', 'dodgerblue', 'forestgreen', 'red', 'gold', 'lightpink', 'darkorchid', 
              'sienna', 'salmon', 'blue', 'limegreen', 'fuchsia', 'mediumturquoise']
    cmap = matplotlib.colors.ListedColormap(colors)

    bounds = np.arange(0.5, 13.6, 1)
    ticks = np.arange(1, 14, 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize = (7,9))
    og = gridspec.GridSpec(2,2, wspace = 0.0, hspace = 0.0, bottom = 0.1)

    ax = fig.add_subplot(og[0:2])
    ax.scatter(old_time, old_flux, color = '0.75', marker = '.', s = 10., zorder = 0)
    for i in range(len(params['stamps'])-1):
        mask = np.ma.getmask(np.ma.masked_inside(time, params['stamps'][i], params['stamps'][i+1]))
        t = time[mask]
        f = flux[mask]
        if i != 0:
            t = t[1:]
            f = f[1:]
        ax.scatter(t, f, color = colors[i], marker = '.', s = 10., zorder = 1)
#        ax.axvline(params['stamps'][i+1], linestyle = '--', linewidth = 0.75, color = 'k')
    ax.set_ylabel('Normalized Flux', fontsize = 20)
    ax.set_xlabel('BJD - 2457000 (days)', fontsize = 20)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))
    ax.tick_params(axis = 'both', which = 'both', direction = 'inout')
    ax.set_ylim([0.9945, 1.0055])
    ax.set_xlim([time.min(), time.max()])
    ax.tick_params(labelsize = 16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap = cmap, norm = norm, boundaries = bounds, ticks = ticks,
                                          spacing = 'uniform', format = '%1i')
    cb.set_label('Sector', fontsize = 20)
    cb.ax.tick_params(labelsize = 16, size = 0)
        
    ax1 = fig.add_subplot(og[2])
    ax1.loglog(frequency, psd, color = params['filter_colors'][0], linestyle = '-', label = 'No filter')
    ax1.loglog(frequency, psd1, color = params['filter_colors'][1], linestyle = '-', label = params['filter_labels'][0])
    ax1.loglog(frequency, psd2, color = params['filter_colors'][2], linestyle = '-', label = params['filter_labels'][1])
    ax1.set_ylabel(r'PSD (ppm$^{2}$/$\mu$Hz)', fontsize = 20)
    ax1.tick_params(axis = 'both', which = 'both', direction = 'inout')
    ax1.set_xlim([params['x_min'], params['x_max']])
    ax1.set_ylim([0.5, 15.])
    if params['inset']:
        ax1.axvline(params['lower'], linestyle = 'dashed', color = '0.25', linewidth = 2)
        ax1.axvline(params['upper'], linestyle = 'dashed', color = '0.25', linewidth = 2)
    ax1.legend(loc = 'lower left', fontsize = 16, facecolor = 'w', framealpha = 1.0)
    ax1.tick_params(labelsize = 16)
    
    width = params['upper'] - params['lower']
    x_min = params['lower'] - (1./10)*width
    x_max = params['upper'] + (9./10)*width
        
    ax2 = fig.add_subplot(og[3])
    if params['original']:
        ax2.plot(frequency, psd, color = params['filter_colors'][0], linestyle = '-')
    ax2.plot(frequency, psd1, color = params['filter_colors'][1], linestyle = '-')
    ax2.plot(frequency, psd2, color = params['filter_colors'][2], linestyle = '-')
    if params['inset']:
        ax2.axvspan(params['lower'], params['upper'], alpha = 0.25, facecolor = '0.5')
        ax2.axvline(params['lower'], linestyle = 'dashed', color = '0.25', linewidth = 2)
        ax2.axvline(params['upper'], linestyle = 'dashed', color = '0.25', linewidth = 2)
#    ax2.set_ylabel(r'PSD (ppm$^{2}$/$\mu$Hz)', fontsize = 20)
#    ax2.set_xlabel(r'Frequency ($\mu$Hz)', fontsize = 20)
    ax2.xaxis.set_major_locator(MultipleLocator(250))
    ax2.xaxis.set_minor_locator(MultipleLocator(50))
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.tick_params(axis = 'both', which = 'both', direction = 'inout')
    ax2.set_xlim([x_min,4167.])
    ax2.tick_params(labelsize = 16)
    ax2.set_ylim([0.5, 15.])

    if params['inset']:
    
        ig = gridspec.GridSpecFromSubplotSpec(30,50, subplot_spec = og[3])
        ax3 = plt.Subplot(fig, ig[1:14,26:-1])
        ax3.plot(lag, auto, 'k-', linewidth = 1.)
        ax3.set_xlabel(r'$\Delta\nu$', fontsize = 20)
        ax3.set_ylabel(r'ACF', fontsize = 20)
        ax3.axvline(params['star']['dNu']/2., color = 'r', linestyle = 'dashed', linewidth = 2.)
        ax3.axvline(params['star']['dNu'], color = 'b', linestyle = 'dashed', linewidth = 2.)
        ax3.axvline(3*params['star']['dNu']/2., color = 'r', linestyle = 'dashed', linewidth = 2.)
        ax3.axvline(2*params['star']['dNu'], color = 'b', linestyle = 'dashed', linewidth = 2.)
        ax3.set_xlim([params['start'], params['end']])
        ax3.set_ylim([min(auto), 1.25*max(auto)])
        ax3.set_xticks([round(params['star']['dNu']/2.,2), round(params['star']['dNu'],2), round(3*params['star']['dNu']/2.,2), round(2.*params['star']['dNu'],2)])
        ax3.tick_params(axis = 'both', which = 'both', direction = 'inout')
        ax3.tick_params(labelsize = 16)
        ax3.set_yticks([])
        fig.add_subplot(ax3)

    fig.text(0.5, 0.02, r'Frequency ($\mu$Hz)', fontsize = 20, ha = 'center')
    
    og.tight_layout(fig, rect = (0, 0.03, 1., 1.))
    if params['pdc']:
        if params['simulate']:
            plt.savefig(params['path'] + '/paper_plot_PDC_ACF_' + params['ext'] + '_days.png', dpi = 150, format = 'png')
        else:
            plt.savefig(params['path'] + '/paper_plot_PDC_ACF.png', dpi = 150, format = 'png')
            plt.savefig(params['path'] + '/paper_plot_PDC_ACF.pdf', dpi = 150, format = 'pdf')
    else:
        if params['simulate']:
            plt.savefig(params['path'] + '/paper_plot_SAP_ACF_' + params['ext'] + '_days.png', dpi = 150, format = 'png')
        else:
            plt.savefig(params['path'] + '/paper_plot_SAP_ACF.png', dpi = 150, format = 'png')
            plt.savefig(params['path'] + '/paper_plot_SAP_ACF.pdf', dpi = 150, format = 'pdf')
    plt.show()
    plt.close()

    	
def inset_plot_ACF(params, frequency, psd, psd1, psd2, lag, auto):

    fig = plt.figure(figsize = (8,10))
    og = gridspec.GridSpec(2,1, wspace = 0)
        
    ax1 = fig.add_subplot(og[0])
    ax1.loglog(frequency, psd, color = params['filter_colors'][0], linestyle = '-', label = 'No filter')
    ax1.loglog(frequency, psd1, color = params['filter_colors'][1], linestyle = '-', label = params['filter_labels'][0])
    ax1.loglog(frequency, psd2, color = params['filter_colors'][2], linestyle = '-', label = params['filter_labels'][1])
    ax1.set_ylabel(r'PSD (ppm$^{2}$/$\mu$Hz)', fontsize = 20)
    ax1.tick_params(axis = 'both', which = 'both', direction = 'inout')
    ax1.set_xlim([params['x_min'], params['x_max']])
    if params['inset']:
        ax1.axvline(params['lower'], linestyle = 'dashed', color = '0.25', linewidth = 2)
        ax1.axvline(params['upper'], linestyle = 'dashed', color = '0.25', linewidth = 2)
    ax1.legend(loc = 'lower left', fontsize = 16, facecolor = 'w', framealpha = 1.0)
    ax1.tick_params(labelsize = 16)
    
    width = params['upper'] - params['lower']
    x_min = params['lower'] - (1./10)*width
    x_max = params['upper'] + (9./10)*width
        
    ax2 = fig.add_subplot(og[1])
    if params['original']:
        ax2.plot(frequency, psd, color = params['filter_colors'][0], linestyle = '-')
    ax2.plot(frequency, psd1, color = params['filter_colors'][1], linestyle = '-')
    ax2.plot(frequency, psd2, color = params['filter_colors'][2], linestyle = '-')
    if params['inset']:
        ax2.axvspan(params['lower'], params['upper'], alpha = 0.25, facecolor = '0.5')
        ax2.axvline(params['lower'], linestyle = 'dashed', color = '0.25', linewidth = 2)
        ax2.axvline(params['upper'], linestyle = 'dashed', color = '0.25', linewidth = 2)
    ax2.set_ylabel(r'PSD (ppm$^{2}$/$\mu$Hz)', fontsize = 20)
    ax2.set_xlabel(r'Frequency ($\mu$Hz)', fontsize = 20)
    ax2.tick_params(axis = 'both', which = 'both', direction = 'inout')
    ax2.set_xlim([x_min,x_max])
    ax2.tick_params(labelsize = 16)
    ax2.set_ylim([params['y_min'], params['y_max']])

    if params['inset']:
    
        ig = gridspec.GridSpecFromSubplotSpec(30,50, subplot_spec = og[1])
        ax3 = plt.Subplot(fig, ig[1:14,26:-1])
        ax3.plot(lag, auto, 'k-', linewidth = 1.)
        ax3.set_xlabel(r'$\Delta\nu$', fontsize = 20)
        ax3.set_ylabel(r'ACF', fontsize = 20)
        ax3.axvline(params['star']['dNu']/2., color = 'r', linestyle = 'dashed', linewidth = 2.)
        ax3.axvline(params['star']['dNu'], color = 'b', linestyle = 'dashed', linewidth = 2.)
        ax3.axvline(3*params['star']['dNu']/2., color = 'r', linestyle = 'dashed', linewidth = 2.)
        ax3.axvline(2*params['star']['dNu'], color = 'b', linestyle = 'dashed', linewidth = 2.)
        ax3.set_xlim([params['start'], params['end']])
        ax3.set_ylim([min(auto), 1.25*max(auto)])
        ax3.set_xticks([round(params['star']['dNu']/2.,2), round(params['star']['dNu'],2), round(3*params['star']['dNu']/2.,2), round(2.*params['star']['dNu'],2)])
        ax3.tick_params(axis = 'both', which = 'both', direction = 'inout')
        ax3.tick_params(labelsize = 16)
        ax3.set_yticks([])
        fig.add_subplot(ax3)
    
    og.tight_layout(fig)
    if params['pdc']:
        if params['simulate']:
            plt.savefig(params['path'] + '/seismic_analysis_PDC_ACF_' + params['ext'] + '_days.png', dpi = 150, format = 'png')
        else:
            plt.savefig(params['path'] + '/seismic_analysis_PDC_ACF.png', dpi = 150, format = 'png')
    else:
        if params['simulate']:
            plt.savefig(params['path'] + '/seismic_analysis_SAP_ACF_' + params['ext'] + '_days.png', dpi = 150, format = 'png')
        else:
            plt.savefig(params['path'] + '/seismic_analysis_SAP_ACF.png', dpi = 150, format = 'png')
    if params['show']:
        plt.show()
    plt.close()

    if params['presentation']:

        if params['zoom']:

            bounds = params['x_bounds']

            mask = np.ma.getmask(np.ma.masked_inside(frequency, bounds[0], bounds[1]))
            pow = psd[mask]

            plt.figure(figsize = (8,5))

            ax = plt.subplot(1,1,1)
            if params['original']:
                ax.plot(frequency, psd, color = params['filter_colors'][0], linestyle = '-')
            ax.plot(frequency, psd1, color = params['filter_colors'][1], linestyle = '-')
            ax.plot(frequency, psd2, color = params['filter_colors'][2], linestyle = '-')
            ax.set_title(r'Baseline = ' + params['ext'] + ' days', fontsize = 22)
            ax.set_ylabel(r'PSD (ppm$^{2}$/$\mu$Hz)', fontsize = 20)
            ax.set_xlabel(r'Frequency ($\mu$Hz)', fontsize = 20)
            ax.tick_params(axis = 'both', which = 'both', direction = 'inout')
            ax.set_xlim([bounds[0], bounds[1]])
            ax.set_ylim([0.95*pow.min(), 1.15*pow.max()])
            ax.tick_params(labelsize = 16)

            plt.tight_layout()
            plt.savefig(params['path'] + '/Dipole_mode_zoom_1_' + params['ext'] + '_days.png', dpi = 150, format = 'png')
            plt.close()

            mask = np.ma.getmask(np.ma.masked_inside(frequency, bounds[0], bounds[1]))
            pow = psd1[mask]

            plt.figure(figsize = (8,5))

            ax = plt.subplot(1,1,1)
            if params['original']:
                ax.plot(frequency, psd, color = params['filter_colors'][0], linestyle = '-')
            ax.plot(frequency, psd1, color = params['filter_colors'][1], linestyle = '-')
            ax.plot(frequency, psd2, color = params['filter_colors'][2], linestyle = '-')
            ax.set_title(r'Baseline = ' + params['ext'] + ' days', fontsize = 22)
            ax.set_ylabel(r'PSD (ppm$^{2}$/$\mu$Hz)', fontsize = 20)
            ax.set_xlabel(r'Frequency ($\mu$Hz)', fontsize = 20)
            ax.tick_params(axis = 'both', which = 'both', direction = 'inout')
            ax.set_xlim([bounds[0], bounds[1]])
            ax.set_ylim([0.95*pow.min(), 1.15*pow.max()])
            ax.tick_params(labelsize = 16)

            plt.tight_layout()
            plt.savefig(params['path'] + '/Dipole_mode_zoom_2_' + params['ext'] + '_days.png', dpi = 150, format = 'png')
            plt.close()

        fig = plt.figure(figsize = (8,5))
        og = gridspec.GridSpec(1,1, wspace = 0)
        
        ax2 = fig.add_subplot(og[0])
        if params['original']:
            ax2.plot(frequency, psd, color = params['filter_colors'][0], linestyle = '-')
        ax2.plot(frequency, psd1, color = params['filter_colors'][1], linestyle = '-')
        ax2.plot(frequency, psd2, color = params['filter_colors'][2], linestyle = '-')
        if params['inset']:
            ax2.axvspan(params['lower'], params['upper'], alpha = 0.25, facecolor = '0.5')
            ax2.axvline(params['lower'], linestyle = 'dashed', color = '0.25', linewidth = 2)
            ax2.axvline(params['upper'], linestyle = 'dashed', color = '0.25', linewidth = 2)
        ax2.set_ylabel(r'PSD (ppm$^{2}$/$\mu$Hz)', fontsize = 20)
        ax2.set_xlabel(r'Frequency ($\mu$Hz)', fontsize = 20)
        ax2.tick_params(axis = 'both', which = 'both', direction = 'inout')
        ax2.set_xlim([x_min,x_max])
        ax2.set_ylim([params['y_min_1'], params['y_max_1']])
        ax2.tick_params(labelsize = 16)

        if params['star']['nuMax'] >= 270:
    
            ig = gridspec.GridSpecFromSubplotSpec(30,50, subplot_spec = og[0])
            ax3 = plt.Subplot(fig, ig[1:14,26:-1])
            ax3.plot(lag, auto, 'k-', linewidth = 1.)
            ax3.set_xlabel(r'$\Delta\nu$', fontsize = 20)
            ax3.set_ylabel(r'ACF', fontsize = 20)
            ax3.axvline(params['star']['dNu']/2., color = 'r', linestyle = 'dashed', linewidth = 2.)
            ax3.axvline(params['star']['dNu'], color = 'b', linestyle = 'dashed', linewidth = 2.)
            ax3.axvline(3*params['star']['dNu']/2., color = 'r', linestyle = 'dashed', linewidth = 2.)
            ax3.axvline(2*params['star']['dNu'], color = 'b', linestyle = 'dashed', linewidth = 2.)
            ax3.set_xlim([params['start'], params['end']])
            ax3.set_ylim([min(auto), 1.25*max(auto)])
            ax3.set_xticks([round(params['star']['dNu']/2.,2), round(params['star']['dNu'],2), round(3*params['star']['dNu']/2.,2), round(2.*params['star']['dNu'],2)])
            ax3.tick_params(axis = 'both', which = 'both', direction = 'inout')
            ax3.tick_params(labelsize = 16)
            ax3.set_yticks([])
            fig.add_subplot(ax3)
    
        og.tight_layout(fig)
        if params['pdc']:
            if params['simulate']:
                plt.savefig(params['path'] + '/PS_inset_PDC_ACF_' + params['ext'] + '_days.png', dpi = 150, format = 'png')
            else:
                plt.savefig(params['path'] + '/PS_inset_PDC_ACF.png', dpi = 150, format = 'png')
        else:
            if params['simulate']:
                plt.savefig(params['path'] + '/PS_inset_SAP_ACF_' + params['ext'] + '_days.png', dpi = 150, format = 'png')
            else:
                plt.savefig(params['path'] + '/PS_inset_SAP_ACF.png', dpi = 150, format = 'png')
        plt.close()
    
    return


def make_echelle(target, inst, dnu, cut = None, freq_per_res = 100, bounds = None, ask = False,
                 freq_lims = None, pow_lims = None, modes = False, MS = 6.5, mews = 2., pdc = None,
                 annotate = False, filters = [1.0, 2.5, 5.0], show = True, path_freqs = None,
                 oversample = False, smooth_image = False):

    path = '../Targets/' + inst + '/' + str(target) + '/'
    if pdc is not None:
        if pdc:
            if oversample:
                fname = path + str(target) + '_PDC_OS.dat.ts.fft.bgcorr'
            else:
                fname = path + str(target) + '_PDC.dat.ts.fft.bgcorr'
        else:
            if oversample:
                fname = path + str(target) + '_SAP_OS.dat.ts.fft.bgcorr'
            else:
                fname = path + str(target) + '_SAP.dat.ts.fft.bgcorr'
    else:
        if oversample:
            fname = path + str(target) + '_OS.dat.ts.fft.bgcorr'
        else:
            fname = path + str(target) + '.dat.ts.fft.bgcorr'
    freq, psd, psd_1, psd_2, psd_3 = get_spectrum(fname, filters)
    filter_labels = [r'%s $\mu$Hz'%(filters[0]), r'%s $\mu$Hz'%(filters[1]), r'%s $\mu$Hz'%(filters[2])]

    if path_freqs is not None:
        if os.path.exists(path_freqs):
            modes = True
            if path_freqs.split('.')[-1] == 'txt':
                freqs = ascii.read(path_freqs)
                ells = freqs['l'].tolist()
                ells = [int(each) for each in ells]
                errs = freqs['Error'].tolist()
                freqs = freqs['Freq'].tolist()
            if path_freqs.split('.')[-1] == 'csv':
                freqs = pd.read_csv(path_freqs)
                ells = freqs['ell'].tolist()
                ells = [int(each) for each in ells]
                freqs = freqs['50'].tolist()
                if 'upper' in freqs.columns.values.tolist():
                    errs = [freqs['lower'].tolist(), freqs['upper'].tolist()]
            freqs = np.array([float(each) for each in freqs])
        
            if len(set(ells)) == 1:
                if ells[0] == 1:
                    red_patch = Line2D([0], [0], marker = 'D', color = 'red', label = '$\ell$ = 1', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                else:
                    red_patch = Line2D([0], [0], marker = 'o', color = 'green', label = '$\ell$ = 0', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                legend_elements = [red_patch]
            if len(set(ells)) == 2:
                red_patch = Line2D([0], [0], marker = 'D', color = 'red', label = '$\ell$ = 1', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                blue_patch = Line2D([0], [0], marker = 'o', color = 'green', label = '$\ell$ = 0', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                legend_elements = [blue_patch, red_patch]
            if len(set(ells)) == 3:
                red_patch = Line2D([0], [0], marker = 's', color = 'red', label = r'$\ell = 1$', ms = MS, linestyle = 'None', mew = 5.)
                blue_patch = Line2D([0], [0], marker = 's', color = 'green', label = r'$\ell = 0$', ms = MS, linestyle = 'None', mew = 5.)
                green_patch = Line2D([0], [0], marker = 's', color = 'blue', label = r'$\ell = 2$', ms = MS, linestyle = 'None', mew = 5.)
                legend_elements = [blue_patch, red_patch, green_patch]
            if len(set(ells)) == 4:
                yellow_patch = Line2D([0], [0], marker = 'o', color = 'yellow', label = '$\ell$ = 3', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                red_patch = Line2D([0], [0], marker = 'D', color = 'red', label = '$\ell$ = 1', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                blue_patch = Line2D([0], [0], marker = 'o', color = 'green', label = '$\ell$ = 0', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                green_patch = Line2D([0], [0], marker = '^', color = 'blue', label = '$\ell$ = 2', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                legend_elements = [blue_patch, red_patch, green_patch, yellow_patch]

    plt.figure(figsize = (10,8))
    plt.plot(freq, psd, color = '0.8', linestyle = '-', zorder = 0, label = filter_labels[0])
    plt.plot(freq, psd_1, color = 'k', linestyle = '-', zorder = 1, label = filter_labels[1])
    plt.plot(freq, psd_2, color = 'r', linestyle = '-', zorder = 2, label = filter_labels[2])
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
    ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')  
#    ax.legend(loc = 'upper left', fontsize = 16, facecolor = 'w', framealpha = 1.0)
    ax.tick_params(labelsize = 20)
    ax.set_yticks([50., 100., 150., 200., 250., 300.])
    ax.set_yticklabels([r'$50$', r'$100$', r'$150$', r'$200$', r'$250$', r'$300$'])
    ax.set_xticks([80., 100., 120., 140., 160., 180., 200., 220.])
    ax.set_xticklabels([r'$80$', r'$100$', r'$120$', r'$140$', r'$160$', r'$180$', r'$200$', r'$220$'])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plt.xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 24)
    plt.ylabel(r'$\rm PSD \,\, [ppm^{2} \mu Hz^{-1}]$', fontsize = 24)
    plt.xlim([freq_lims[0], freq_lims[1]])
    plt.ylim([0., 330.])

    path += 'Plots/'

    plt.tight_layout()
    if pdc is not None:
        if pdc:
            if oversample:
                fname = path + 'PDC_OS_bg_subtracted_PS.png'
            else:
                fname = path + 'PDC_bg_subtracted_PS.png'
        else:
            if oversample:
                fname = path + 'SAP_OS_bg_subtracted_PS.png'
            else:
                fname = path + 'SAP_bg_subtracted_PS.png'
    else:
        if oversample:
            fname = path + 'OS_bg_subtracted_PS.png'
        else:
            fname = path + 'bg_subtracted_PS.png'
    plt.savefig(fname, dpi = 150)
    plt.close()

    if bounds is not None:
        if bounds[0] is not None:
            if bounds[1] is not None:
                mask = np.ma.getmask(np.ma.masked_inside(freq, bounds[0], bounds[1]))
            else:
                mask = np.ma.getmask(np.ma.masked_greater_equal(freq, bounds[0]))
        else:
            mask = np.ma.getmask(np.ma.masked_less_equal(freq, bounds[1]))
        freq = freq[mask]
        psd = psd[mask]
        psd_1 = psd_1[mask]
        psd_2 = psd_2[mask]
        psd_3 = psd_3[mask]
    """
    # Power spectrum

    plt.figure(figsize = (8,10))
    plt.subplot(2,1,1)

#    plt.plot(freq, psd, color = '0.8', zorder = 0)
    plt.plot(freq, psd_1, color = '0.8', zorder = 1)
    plt.plot(freq, psd_2, color = 'k', zorder = 2)
    plt.xlabel('Frequency ($\mu$Hz)', fontsize = 20) 
    plt.ylabel('PSD (ppm$^2$$\mu$Hz$^{-1}$)', fontsize = 20)
    if freq_lims is not None:
        plt.xlim([freq_lims[0],freq_lims[1]])
    if pow_lims is not None:
        plt.ylim([pow_lims[0],pow_lims[1]])
    ax = plt.gca()
    ax.minorticks_on()
    ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
    ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')  
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(labelsize = 16)
    if modes:
        ax.legend(handles = legend_elements, loc = 'upper left', fontsize = 16, facecolor = 'w', framealpha = 1.0, columnspacing = 0.0, labelspacing = 0.25, borderpad = 0.5, handletextpad = 0.5, handlelength = 0.5)
    if annotate:
        plt.annotate("(a)", xy = (0.025, 0.90), xycoords = "axes fraction", fontsize = 18)
    if modes:
        for i in range(len(ells)):
            if ells[i] == 0:
                plt.axvline(freqs[i], color = 'green', alpha = 1.0, ls = 'dashed', zorder = 1)
            elif ells[i] == 1:
                plt.axvline(freqs[i], color = 'red', alpha = 1.0, ls = 'dashed', zorder = 1)
            elif ells[i] == 2:
                plt.axvline(freqs[i], color = 'blue', alpha = 1.0, ls = 'dashed', zorder = 1)
            else:
                continue

    """

    if ask:
        print()
        answer = ask_int('Which PS would you like to use? (0/1/2/3): ')
    else:
        answer = int(0)

    # Echelle diagram

    if cut is not None:
        psd[psd > cut] = cut
        psd_1[psd_1 > cut] = cut
        psd_2[psd_2 > cut] = cut
        psd_3[psd_3 > cut] = cut

    if answer == 0:
        amp = psd[:]
    elif answer == 1:
        amp = psd_1[:]
    elif answer == 2:
        amp = psd_2[:]
    elif answer == 3:
        amp = psd_3[:]
    else:
        print("Did not understand that input.")
        return

    res = (freq[1] - freq[0])*1.
    rpd = dnu/res
    n = np.ceil(rpd/freq_per_res)
    if n < freq_per_res:
        if dnu >= 100.:
            n = 200.
        else:
            n = 100.
    boxx = dnu/n
    boxy = dnu

    xax = np.arange(0., dnu, boxx)
    yax = np.arange(min(freq), max(freq), dnu)

    arr = np.zeros((len(xax),len(yax)))
    gridx = np.zeros(len(xax))
    gridy = np.zeros(len(yax))

    mod_freq = freq%dnu

    startx = 0.
    starty = min(freq)

    for i in range(len(xax)):

        for j in range(len(yax)):

            use = np.where((mod_freq >= startx) & (mod_freq < startx+boxx) & (freq >= starty) & (freq < starty+boxy))[0]
            if (len(use) == 0):
                continue
            arr[i,j] = np.sum(amp[use])
            gridy[j] = starty + boxy/2.
            starty += boxy
    
        gridx[i] = startx + boxx/2.
        starty = min(freq)
        startx += boxx

    smooth = arr
    dim = smooth.shape

    smooth_2 = np.zeros((2*dim[0]+1,dim[1]))
    smooth_2[0:dim[0],:] = smooth
    smooth_2[dim[0]+1:(2*dim[0])+1,:] = smooth

    plt.figure(figsize = (10,8))
    plt.subplot(1,1,1)
    if smooth_image:
        plt.imshow(np.swapaxes(smooth_2, 0, 1), aspect = 'auto', interpolation = 'bilinear', origin = 'lower', 
                   extent = [min(gridx)-boxx/2., 2*max(gridx)+boxx/2., min(gridy)-boxy/2.,
                   max(gridy)+boxy/2.], cmap = 'gray_r')
    else:
        plt.imshow(np.swapaxes(smooth_2, 0, 1), aspect = 'auto', interpolation = 'none', origin = 'lower', 
                   extent = [min(gridx)-boxx/2., 2*max(gridx)+boxx/2., min(gridy)-boxy/2.,
                   max(gridy)+boxy/2.], cmap = 'gray_r')

    plt.xlabel(r'$\rm Frequency \,\, mod \,\, %.3f \, \mu Hz$'%dnu, fontsize = 24)
    plt.ylabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 24)
#    plt.axvline(dnu, color = 'k', linestyle = '--', linewidth = 3.5)
    ax = plt.gca()
    ax.minorticks_on()
    ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
    ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')  
    ax.set_yticks([80., 100., 120., 140., 160., 180., 200., 220.])
    ax.set_yticklabels([r'$80$', r'$100$', r'$120$', r'$140$', r'$160$', r'$180$', r'$200$', r'$220$'])
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xticks([0., 5., 10., 15., 20., 25.])
    ax.set_xticklabels([r'$0$', r'$5$', r'$10$', r'$15$', r'$20$', r'$25$'])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
#    ax.legend(handles = legend_elements, loc = 'upper right', fontsize = 20, facecolor = 'w', framealpha = 1.0, columnspacing = 0.0, labelspacing = 0.25, borderpad = 0.5, handletextpad = 0.5, handlelength = 0.5)
    ax.tick_params(labelsize = 20)
    if annotate:
        plt.annotate("(b)", xy = (0.025, 0.90), xycoords = "axes fraction", fontsize = 18)
    if modes:
        for i in range(len(ells)):
            if ells[i] == 0:
                plt.errorbar(freqs[i]%dnu,freqs[i], xerr = errs[i], color = 'green', capsize = 4., ecolor = 'green', elinewidth = mews, capthick = 2., ls = 'None')
                plt.errorbar((freqs[i]%dnu)+dnu,freqs[i], xerr = errs[i], color = 'green', capsize = 4., ecolor = 'green', elinewidth = mews, capthick = 2., ls = 'None')
            elif ells[i] == 1:
                plt.errorbar(freqs[i]%dnu,freqs[i], xerr = errs[i], color = 'red', capsize = 4., ecolor = 'red', elinewidth = mews, capthick = 2., ls = 'None')
                plt.errorbar((freqs[i]%dnu)+dnu,freqs[i], xerr = errs[i], color = 'red', capsize = 4., ecolor = 'red', elinewidth = mews, capthick = 2., ls = 'None')
            elif ells[i] == 2:
                plt.errorbar(freqs[i]%dnu,freqs[i], xerr = errs[i], color = 'blue', capsize = 4., ecolor = 'blue', elinewidth = mews, capthick = 2., ls = 'None')
                plt.errorbar((freqs[i]%dnu)+dnu,freqs[i], xerr = errs[i], color = 'blue', capsize = 4., ecolor = 'blue', elinewidth = mews, capthick = 2., ls = 'None')

    if freq_lims is not None:
        plt.ylim([freq_lims[0],freq_lims[1]])
    else:
        plt.ylim([min(freq), max(freq)])
    plt.xlim([0.,2*dnu])
    plt.tight_layout()
    d = str(dnu)

    if pdc is not None:
        if pdc:
            if oversample:
                fname = '%sechelle_PDC_OS_%s'
            else:
                fname = '%sechelle_PDC_%s'
        else:
            if oversample:
                fname = '%sechelle_SAP_OS_%s'
            else:
                fname = '%sechelle_SAP_%s'
    else:
        if oversample:
            fname = '%sechelle_OS_%s'
        else:
            fname = '%sechelle_%s'
    plt.savefig(fname%(path, ''.join(d.split('.')))+'.png', dpi = 200)
    plt.savefig(fname%(path, ''.join(d.split('.')))+'.pdf', dpi = 200)
#    plt.show()
    plt.close()
    return


def make_echelles(target, inst, dnu, paths, shapes, names, cut = None, freq_per_res = 100, bounds = None, 
                  freq_lims = None, pow_lims = None, modes = False, MS = 6.5, mews = 2., oversample = True,
                  annotate = False, filters = [0.5, 1.5, 2.5], show = True, ask = False, peaks = 10):

    path = '../Targets/' + inst + '/' + str(target)

    if oversample:
        fname = path + str(target) + '.dat.ts.fft.bgcorr'
        if not os.path.exists(fname):
            time, flux = get_light_curve(path+str(target)+'_LC.txt')

            vars = ['chop', 'filter', 'simulate', 'oversample', 'peaks', 'verbose', 'filters', 'short_cadence', 'inst', 'save_plots', 'show']
            vals = [False, False, False, True, peaks, False, filters, True, inst, False, False]
            params = dict(zip(vars,vals))

            params, data = time_to_frequency(time, flux, params)
            freq = np.array(data['frequency'])
            psd = np.array(data['power'])
            psd_1 = np.array(data[filters[0]])
            psd_2 = np.array(data[filters[1]])
            psd_3 = np.array(data[filters[2]])
        else:
            freq, psd, psd_1, psd_2, psd_3 = get_spectrum(fname, filters)

        filter_labels = [r'$\rm %.1f \,\,\mu Hz$'%(filters[0]), r'$\rm %.1f \,\,\mu Hz$'%(filters[1]), r'$\rm %.1f \,\,\mu Hz$'%(filters[2])]

        plt.figure(figsize = (10,8))

        plt.plot(freq, psd_1, color = '0.8', linestyle = '-', zorder = 0, label = filter_labels[0])
        plt.plot(freq, psd_2, color = 'k', linestyle = '-', zorder = 1, label = filter_labels[1])
        plt.plot(freq, psd_3, color = 'r', linestyle = '-', zorder = 2, label = filter_labels[2])
        plt.xlabel(r'$\rm Frequency \,\,[\mu Hz]$', fontsize = 26)
        plt.ylabel(r'$\rm PSD \,\, [ppm^2 \, \mu Hz^{-1}]$', fontsize = 26)
        plt.xlim([min(freq), 4167.])
        plt.ylim([0., 5.])
        ax = plt.gca()
        ax.minorticks_on()
        ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
        ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')  
        ax.legend(loc = 'upper left', fontsize = 16, facecolor = 'w', framealpha = 1.0)
        ax.tick_params(labelsize = 16)

        plt.tight_layout()
        plt.show()
        plt.close()

    if bounds is not None:
        if bounds[0] is not None:
            if bounds[1] is not None:
                mask = np.ma.getmask(np.ma.masked_inside(freq, bounds[0], bounds[1]))
            else:
                mask = np.ma.getmask(np.ma.masked_greater_equal(freq, bounds[0]))
        else:
            mask = np.ma.getmask(np.ma.masked_less_equal(freq, bounds[1]))
        freq = freq[mask]
        psd = psd[mask]
        psd_1 = psd_1[mask]
        psd_2 = psd_2[mask]
        psd_3 = psd_3[mask]

    if ask:
        print()
        answer = ask_int('Which PS would you like to use? (0/1/2/3): ')
    else:
        answer = int(2)

    if cut is not None:
        psd[psd > cut] = cut
        psd_1[psd_1 > cut] = cut
        psd_2[psd_2 > cut] = cut
        psd_3[psd_3 > cut] = cut

    if answer == 0:
        amp = psd[:]
    elif answer == 1:
        amp = psd_1[:]
    elif answer == 2:
        amp = psd_2[:]
    elif answer == 3:
        amp = psd_3[:]
    else:
        print("Did not understand that input (i.e. which power spectrum).")
        return

    res = (freq[1] - freq[0])*1.
    rpd = dnu/res
    n = np.ceil(rpd/freq_per_res)
    if n < freq_per_res:
        if dnu >= 100.:
            n = 200.
        else:
            n = 100.
    boxx = dnu/n
    boxy = dnu

    xax = np.arange(0., dnu, boxx)
    yax = np.arange(min(freq), max(freq), dnu)

    arr = np.zeros((len(xax),len(yax)))
    gridx = np.zeros(len(xax))
    gridy = np.zeros(len(yax))

    mod_freq = freq%dnu

    startx = 0.
    starty = min(freq)

    for i in range(len(xax)):

        for j in range(len(yax)):

            use = np.where((mod_freq >= startx) & (mod_freq < startx+boxx) & (freq >= starty) & (freq < starty+boxy))[0]
            if (len(use) == 0):
                continue
            arr[i,j] = np.sum(amp[use])
            gridy[j] = starty + boxy/2.
            starty += boxy
    
        gridx[i] = startx + boxx/2.
        starty = min(freq)
        startx += boxx

    smooth = arr
    dim = smooth.shape

    # Plot
    plt.figure(figsize = (16,6))
    x = 0

    for p, s, n in zip(paths, shapes, names):
        x += 1
        if os.path.exists(path+p):
            modes = True
            path_freqs = path+p
            if path_freqs.split('.')[-1] == 'txt':
                freqs = ascii.read(path_freqs)
                ells = freqs['l'].tolist()
                ells = [int(each) for each in ells]
                freqs = freqs['Freq'].tolist()
            if path_freqs.split('.')[-1] == 'csv':
                freqs = pd.read_csv(path_freqs)
                ells = freqs['ell'].tolist()
                ells = [int(each) for each in ells]
                freqs = freqs['50'].tolist()
            freqs = np.array([float(each) for each in freqs])
        
            if len(set(ells)) == 1:
                if ells[0] == 1:
                    red_patch = Line2D([0], [0], marker = s, color = 'red', label = r'$\ell = 1$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                else:
                    red_patch = Line2D([0], [0], marker = s, color = 'green', label = r'$\ell = 0$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                legend_elements = [red_patch]
            elif len(set(ells)) == 2:
                red_patch = Line2D([0], [0], marker = s, color = 'red', label = r'$\ell = 1$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                blue_patch = Line2D([0], [0], marker = s, color = 'green', label = r'$\ell = 0$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                legend_elements = [blue_patch, red_patch]
            elif len(set(ells)) == 3:
                red_patch = Line2D([0], [0], marker = s, color = 'red', label = r'$\ell = 1$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                blue_patch = Line2D([0], [0], marker = s, color = 'green', label = r'$\ell = 0$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                green_patch = Line2D([0], [0], marker = s, color = 'blue', label = r'$\ell = 2$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                legend_elements = [blue_patch, red_patch, green_patch]
            elif len(set(ells)) == 4:
                yellow_patch = Line2D([0], [0], marker = s, color = 'yellow', label = r'$\ell = 3$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                red_patch = Line2D([0], [0], marker = s, color = 'red', label = r'$\ell = 1$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                blue_patch = Line2D([0], [0], marker = s, color = 'green', label = r'$\ell = 0$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                green_patch = Line2D([0], [0], marker = s, color = 'blue', label = r'$\ell = 2$', ms = MS, fillstyle = 'none', linestyle = 'None', mew = mews)
                legend_elements = [blue_patch, red_patch, green_patch, yellow_patch]
            else:
                print('Too many spherical degrees.')
                return

        else:
            print('Path typed incorrectly.')
            return

        plt.subplot(1,len(paths),x)
        plt.imshow(np.swapaxes(smooth, 0, 1), aspect = 'auto', interpolation = 'none', origin = 'lower', 
                   extent = [min(gridx)-boxx/2., 2*max(gridx)+boxx/2., min(gridy)-boxy/2.,
                   max(gridy)+boxy/2.], cmap = 'gray_r')
        plt.title(n, fontsize = 30)
        ax = plt.gca()
        ax.minorticks_on()
        ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
        ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
        if x == 1:
            ax.set_yticks([2800, 3000, 3200, 3400, 3600])
            ax.set_yticklabels([r'$2800$', r'$3000$', r'$3200$', r'$3400$', r'$3600$'])
            ax.yaxis.set_minor_locator(MultipleLocator(50))
            plt.ylabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize = 26)
        else:
            if x == 2:
                plt.xlabel(r'$\rm Frequency \,\, mod \,\, %.2f \,\, \mu Hz$'%dnu, fontsize = 26)
            plt.yticks([])
        ax.set_xticks([25., 50., 75., 100., 125.])
        ax.set_xticklabels([r'$25$', r'$50$', r'$75$', r'$100$', r'$125$'])
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.legend(handles = legend_elements, loc = 'upper left', fontsize = 16, facecolor = 'w', framealpha = 1.0)
        ax.tick_params(labelsize = 20)

        for i in range(len(ells)):
            if ells[i] == 0:
                plt.plot(freqs[i]%dnu,freqs[i], s, color = 'green', fillstyle = 'none', mew = mews, ms = MS)
                plt.plot((freqs[i]%dnu)+dnu,freqs[i], s, color = 'green', fillstyle = 'none', mew = mews, ms = MS)
            elif ells[i] == 1:
                plt.plot(freqs[i]%dnu,freqs[i], s, color = 'red', fillstyle = 'none', mew = mews, ms = MS)
                plt.plot((freqs[i]%dnu)+dnu,freqs[i], s, color = 'red', fillstyle = 'none', mew = mews, ms = MS)
            elif ells[i] == 2:
                plt.plot(freqs[i]%dnu,freqs[i], s, color = 'blue', fillstyle = 'none', mew = mews, ms = MS)
                plt.plot((freqs[i]%dnu)+dnu,freqs[i], s, color = 'blue', fillstyle = 'none', mew = mews, ms = MS)

        if freq_lims is not None:
            plt.ylim([freq_lims[0],freq_lims[1]])
        else:
            plt.ylim([min(freq), max(freq)])
        plt.xlim([0.,dnu])

    plt.tight_layout()

    plt.savefig(path + 'echelle_multiple.png', dpi = 200)
    plt.savefig(path + 'echelle_multiple.pdf', dpi = 200)
    plt.show()
    plt.close()
    return


def make_spectra(target, inst, paths, colors, names, freq_lims = None, modes = False, MS = 6.5, mews = 2., 
                 annotate = False, filters = [0.25, 1.5, 2.5], show = True):

    path = '../Targets/' + inst + '/' + str(target) + '/'

    fname = path + str(target) + '.dat.ts.fft.bgcorr'
    if not os.path.exists(fname):
        time, flux = get_light_curve(path+str(target)+'_LC.txt')

        vars = ['chop', 'filter', 'simulate', 'oversample', 'peaks', 'verbose', 'filters', 'short_cadence', 'inst', 'save_plots', 'show']
        vals = [False, False, False, True, peaks, False, filters, True, inst, False, False]
        params = dict(zip(vars,vals))

        params, data = time_to_frequency(time, flux, params)
        freq = np.array(data['frequency'])
        psd = np.array(data['power'])
        psd_1 = np.array(data[filters[0]])
        psd_2 = np.array(data[filters[1]])
        psd_3 = np.array(data[filters[2]])
    else:
        freq, psd, psd_1, psd_2, psd_3 = get_spectrum(fname, filters)

    plt.figure(figsize = (20,8))

    plt.plot(freq, psd, color = 'k', linestyle = '-', zorder = 0)
    plt.plot(freq, psd_2, color = 'r', linestyle = '-', zorder = 3)
    plt.xlabel(r'$\rm Frequency \,\,[\mu Hz]$', fontsize = 26)
    plt.ylabel(r'$\rm PSD \,\, [ppm^2 \, \mu Hz^{-1}]$', fontsize = 26)
    if freq_lims is not None:
        mask = np.ma.getmask(np.ma.masked_inside(freq, freq_lims[0], freq_lims[1]))
        plt.xlim([freq_lims[0], freq_lims[1]])
    else:
        mask = np.ma.getmask(np.ma.masked_inside(freq, min(freq), 4167.))
        plt.xlim([min(freq), 4167.])
    pow = psd[mask]
    y_max = 1.25*max(pow)
    plt.ylim([0., y_max])
    ax = plt.gca()
    ax.minorticks_on()
    ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
    ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
    ax.set_xticks([2800, 3000, 3200, 3400, 3600])
    ax.set_xticklabels([r'$2800$', r'$3000$', r'$3200$', r'$3400$', r'$3600$'])
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.set_yticks([0., 5., 10., 15.])
    ax.set_yticklabels([r'$0$', r'$5$', r'$10$', r'$15$'])
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(labelsize = 20)
    legend_elements = []

    for p, c, n in zip(paths, colors, names):

        if os.path.exists(path+p):
            modes = True
            path_freqs = path+p
            if path_freqs.split('.')[-1] == 'txt':
                FR = ascii.read(path_freqs)
                ells = FR['l'].tolist()
                ells = [int(each) for each in ells]
                freqs = FR['Freq'].tolist()
                freq_errs = FR['error'].tolist()
                freq_u = [float(f+e) for f, e in zip(freqs, freq_errs)]
                freq_l = [float(f-e) for f, e in zip(freqs, freq_errs)]
            if path_freqs.split('.')[-1] == 'csv':
                FR = pd.read_csv(path_freqs)
                ells = FR['ell'].tolist()
                ells = [int(each) for each in ells]
                freqs = FR['50'].tolist()
                freq_u = FR['84.13'].tolist()
                freq_l = FR['15.87'].tolist()

        else:
            print('Path typed incorrectly.')
            return

        legend_elements.append(Line2D([0], [0], color = c, lw = 2., ls = '--', label = n))

        for i in range(len(freqs)):
            ax.axvspan(freq_l[i], freq_u[i], color = c, alpha = 0.25, zorder = 1)
            ax.axvline(freqs[i], color = c, ls = '--', lw = 2., zorder = 2)

    ax.legend(handles = legend_elements, loc = 'upper left', fontsize = 24., facecolor = 'w', framealpha = 1.0)
    plt.tight_layout()

    plt.savefig(path + 'PS_multiple.pdf', dpi = 200)
    plt.show()
    plt.close()
    return