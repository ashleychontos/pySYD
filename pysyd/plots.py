import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel

from pysyd import models
from pysyd import utils


def set_plot_params():
    """
    Sets the matplotlib parameters.

    Returns
    -------
    None

    """

    plt.style.use('dark_background')
    plt.rcParams.update({
        'agg.path.chunksize': 10000,
        'mathtext.fontset': 'stix',
        'figure.autolayout': True,
        'lines.linewidth': 1,
        'axes.titlesize': 18.0,
        'axes.labelsize': 16.0,
        'axes.linewidth': 1.25,
        'axes.formatter.useoffset': False,
        'xtick.major.size': 10.0,
        'xtick.minor.size': 5.0,
        'xtick.major.width': 1.25,
        'xtick.minor.width': 1.25,
        'xtick.direction': 'inout',
        'ytick.major.size': 10.0,
        'ytick.minor.size': 5.0,
        'ytick.major.width': 1.25,
        'ytick.minor.width': 1.25,
        'ytick.direction': 'inout',
    })



def plot_estimates(star, block=False):
    """
    Creates a plot summarizing the results of the find excess routine.

    Parameters
    ----------
    star : target.Target
        the pySYD pipeline object

    Returns
    -------
    None
    
    """
    d = utils.get_dict(type='plots')
    npanels = 3+star.excess['n_trials']
    x, y = d[npanels]['x'], d[npanels]['y']
    fig = plt.figure("Estimate numax results for %s"%star.name, figsize=d[npanels]['size'])

    # Time series data
    ax1 = plt.subplot(y, x, 1)
    if star.lc:
        ax1.plot(star.time, star.flux, 'w-')
        ax1.set_xlim([min(star.time), max(star.time)])
    ax1.set_title(r'$\rm Time \,\, series$')
    ax1.set_xlabel(r'$\rm Time \,\, [days]$')
    ax1.set_ylabel(r'$\rm Flux$')

    # log-log power spectrum with crude background fit
    ax2 = plt.subplot(y, x, 2)
    ax2.loglog(star.freq, star.pow, 'w-')
    ax2.set_xlim([min(star.freq), max(star.freq)])
    ax2.set_ylim([min(star.pow), max(star.pow)*1.25])
    ax2.set_title(r'$\rm Crude \,\, background \,\, fit$')
    ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    if star.excess['binning'] is not None:
        ax2.loglog(star.bin_freq, star.bin_pow, 'r-')
    ax2.loglog(star.freq, star.interp_pow, color='lime', linestyle='-', lw=2.0)

    # Crude background-corrected power spectrum
    ax3 = plt.subplot(y, x, 3)
    ax3.plot(star.freq, star.bgcorr_pow, 'w-')
    ax3.set_xlim([min(star.freq), max(star.freq)])
    ax3.set_ylim([0.0, max(star.bgcorr_pow)*1.25])
    ax3.set_title(r'$\rm Background \,\, corrected \,\, PS$')
    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')

    # ACF trials to determine numax
    for i in range(star.excess['n_trials']):
        ax = plt.subplot(y, x, 4+i)
        ax.plot(star.excess['results'][star.name][i+1]['x'], star.excess['results'][star.name][i+1]['y'], 'w-')
        xran = max(star.excess['results'][star.name][i+1]['fitx'])-min(star.excess['results'][star.name][i+1]['fitx'])
        ymax = star.excess['results'][star.name][i+1]['maxy']
        ax.axvline(star.excess['results'][star.name][i+1]['maxx'], linestyle='dotted', color='r', linewidth=0.75)
        ax.set_title(r'$\rm Collapsed \,\, ACF \,\, [trial \,\, %d]$' % (i+1))
        ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax.set_ylabel(r'$\rm Arbitrary \,\, units$')
        if star.excess['results'][star.name][i+1]['good_fit']:
            ax.plot(star.excess['results'][star.name][i+1]['fitx'], star.excess['results'][star.name][i+1]['fity'], color='lime', linestyle='-', linewidth=1.5)
            if max(star.excess['results'][star.name][i+1]['fity']) > star.excess['results'][star.name][i+1]['maxy']:
                ymax = max(star.excess['results'][star.name][i+1]['fity'])
            ax.axvline(star.excess['results'][star.name][i+1]['numax'], color='lime', linestyle='--', linewidth=0.75)
        yran = np.absolute(ymax)
        ax.set_xlim([min(star.excess['results'][star.name][i+1]['x']), max(star.excess['results'][star.name][i+1]['x'])])
        ax.set_ylim([-0.05, ymax+0.15*yran])
        ax.annotate(r'$\rm SNR = %3.2f$' % star.excess['results'][star.name][i+1]['snr'], xy=(min(star.excess['results'][star.name][i+1]['fitx'])+0.05*xran, ymax+0.025*yran), fontsize=18)

    plt.tight_layout()
    if star.params['save']:
        if not star.params['overwrite']:
            plt.savefig(utils.get_next(star,'find_numax.png'), dpi=300)
        else:
            plt.savefig(os.path.join(star.params[star.name]['path'],'find_numax.png'), dpi=300)
    if not star.params['cli']:
        with open('find_numax.pickle','wb') as f:
            pickle.dump(fig, f)
        star.pickles.append('find_numax.pickle')
    if not star.params['show']:
        plt.close()
    if block:
        plt.show(block=False)


def plot_parameters(star, n_peaks=10):
    """
    Creates a plot summarizing the results of the fit background routine.

    Parameters
    ----------
    star : target.Target
        the main pipeline Target class object
    n_peaks : int
        the number of peaks to highlight in the zoomed-in power spectrum

    Results
    -------
    None

    """
    if star.params['background'] and not star.params['global']:
        npanels=3
    else:
        npanels=9
    d = utils.get_dict(type='plots')
    x, y = d[npanels]['x'], d[npanels]['y']
    fig = plt.figure("Global fit for %s"%star.name, figsize=d[npanels]['size'])
    # Time series data
    ax1 = fig.add_subplot(x, y, 1)
    if star.lc:
        ax1.plot(star.time, star.flux, 'w-')
        ax1.set_xlim([min(star.time), max(star.time)])
    ax1.set_title(r'$\rm Time \,\, series$')
    ax1.set_xlabel(r'$\rm Time \,\, [days]$')
    ax1.set_ylabel(r'$\rm Flux$')

    # Initial background guesses
    ax2 = fig.add_subplot(x, y, 2)
    ax2.plot(star.frequency, star.random_pow, c='lightgrey', zorder=0, alpha=0.5)
    ax2.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.random_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'w-', zorder=1)
    ax2.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.random_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'w-', zorder=1)
    ax2.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.smooth_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
    ax2.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.smooth_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
    if star.params['background']:
        total = np.zeros_like(star.frequency)
        for r in range(star.nlaws_orig):
            model = models.background(star.frequency, [star.b_orig[r], star.a_orig[r]])
            ax2.plot(star.frequency, model, color='blue', linestyle=':', linewidth=1.5, zorder=4)
            total += model
        total += star.noise
        ax2.plot(star.frequency, total, color='blue', linewidth=2., zorder=5)
        ax2.errorbar(star.bin_freq, star.bin_pow, yerr=star.bin_err, color='lime', markersize=0., fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
    ax2.axvline(star.params[star.name]['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax2.axvline(star.params[star.name]['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax2.axhline(star.noise, color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5, 5))
    ax2.set_xlim([min(star.frequency), max(star.frequency)])
    ax2.set_ylim([min(star.power), max(star.power)*1.25])
    ax2.set_title(r'$\rm Initial \,\, guesses$')
    ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    n = np.logspace(np.log10(min(star.frequency)),np.log10(max(star.frequency)),10)
    # Fitted background
    ax3 = fig.add_subplot(x, y, 3)
    ax3.plot(star.frequency, star.random_pow, c='lightgrey', zorder=0, alpha=0.5)
    ax3.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.random_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'w-', linewidth=0.75, zorder=1)
    ax3.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.random_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'w-', linewidth=0.75, zorder=1)
    ax3.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.smooth_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
    ax3.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.smooth_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
    if star.params['background']:
        total = np.zeros_like(star.frequency)
        if len(star.pars) > 1:
            for r in range(len(star.pars)//2):
                yobs = models.background(star.frequency, [star.pars[r*2], star.pars[r*2+1]])
                ax3.plot(star.frequency, yobs, color='blue', linestyle=':', linewidth=1.5, zorder=4)
                total += yobs
        if len(star.pars)%2 == 0:
            total += star.noise
            ax3.axhline(star.noise, color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
        else:
            total += star.pars[-1]
            ax3.axhline(star.pars[-1], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
        ax3.plot(star.frequency, total, color='blue', linewidth=2., zorder=5)
        ax3.errorbar(star.bin_freq, star.bin_pow, yerr=star.bin_err, color='lime', markersize=0.0, fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
    else:
        ax3.axhline(star.noise, color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
    ax3.axvline(star.params[star.name]['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax3.axvline(star.params[star.name]['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    if star.params['global']:
        ax3.plot(star.frequency[(star.frequency >= n[2])&(star.frequency <= n[-2])], star.pssm[(star.frequency >= n[2])&(star.frequency <= n[-2])], color='yellow', linewidth=2.0, linestyle='dashed', zorder=6)
    ax3.set_xlim([min(star.frequency), max(star.frequency)])
    ax3.set_ylim([min(star.power), max(star.power)*1.25])
    ax3.set_title(r'$\rm Fitted \,\, model$')
    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    if star.params['background'] and not star.params['global']: 
        plt.tight_layout()
        if star.params['save']:
            plt.savefig(os.path.join(star.params[star.name]['path'],'background_only.png'), dpi=300)
        if not star.params['cli']:
            with open('background_only.pickle','wb') as f:
                pickle.dump(fig, f)
            star.pickles.append('background_only.pickle')
        if not star.params['show']:
            plt.close()
        return

    # Smoothed power excess w/ gaussian
    ax4 = fig.add_subplot(x, y, 4)
    ax4.plot(star.region_freq, star.region_pow, 'w-', zorder=0)
    idx = utils.return_max(star.region_freq, star.region_pow, index=True)
    ax4.plot([star.region_freq[idx]], [star.region_pow[idx]], color='red', marker='s', markersize=7.5, zorder=0)
    ax4.axvline([star.region_freq[idx]], color='white', linestyle='--', linewidth=1.5, zorder=0)
    xx, yy = utils.return_max(star.new_freq, star.numax_fit)
    ax4.plot(star.new_freq, star.numax_fit, 'b-', zorder=3)
    ax4.axvline(xx, color='blue', linestyle=':', linewidth=1.5, zorder=2)
    ax4.plot([xx], [yy], color='b', marker='D', markersize=7.5, zorder=1)
    ax4.axvline(star.exp_numax, color='r', linestyle='--', linewidth=1.5, zorder=1, dashes=(5,5))
    ax4.set_title(r'$\rm Smoothed \,\, bg$-$\rm corrected \,\, PS$')
    ax4.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax4.set_xlim([min(star.region_freq), max(star.region_freq)])

    # Background-corrected power spectrum with n highest peaks
    mask = np.ma.getmask(np.ma.masked_inside(star.frequency, star.params[star.name]['ps_mask'][0], star.params[star.name]['ps_mask'][1]))
    star.freq = star.frequency[mask]
    star.psd = star.bg_corr_smooth[mask]
    peaks_f, peaks_p = utils.max_elements(star.freq, star.psd, n_peaks)
    ax5 = fig.add_subplot(x, y, 5)
    ax5.plot(star.freq, star.psd, 'w-', zorder=0, linewidth=1.0)
    ax5.scatter(peaks_f, peaks_p, s=25.0, edgecolor='r', marker='s', facecolor='none', linewidths=1.0)
    ax5.set_title(r'$\rm Bg$-$\rm corrected \,\, PS$')
    ax5.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax5.set_ylabel(r'$\rm Power$')
    ax5.set_xlim([star.params[star.name]['ps_mask'][0], star.params[star.name]['ps_mask'][1]])
    ax5.set_ylim([min(star.psd)-0.025*(max(star.psd)-min(star.psd)), max(star.psd)+0.1*(max(star.psd)-min(star.psd))])

    sig = 0.35*star.exp_dnu/2.35482 
    weights = 1./(sig*np.sqrt(2.*np.pi))*np.exp(-(star.lag-star.exp_dnu)**2./(2.*sig**2))
    new_weights = weights/max(weights)
    diff = list(np.absolute(star.lag-star.obs_dnu))
    idx = diff.index(min(diff))
    # ACF for determining dnu
    ax6 = fig.add_subplot(x, y, 6)
    ax6.plot(star.lag, star.auto, 'w-', zorder=0, linewidth=1.)
    ax6.scatter(star.peaks_l, star.peaks_a, s=30.0, edgecolor='r', marker='^', facecolor='none', linewidths=1.0)
    ax6.axvline(star.exp_dnu, color='red', linestyle=':', linewidth=1.5, zorder=5)
    ax6.axvline(star.obs_dnu, color='lime', linestyle='--', linewidth=1.5, zorder=2)
    ax6.scatter(star.lag[idx], star.auto[idx], s=45.0, edgecolor='lime', marker='s', facecolor='none', linewidths=1.0)
    ax6.plot(star.zoom_lag, star.zoom_auto, 'r-', zorder=5, linewidth=1.0)
    ax6.plot(star.lag, new_weights, c='yellow', linestyle=':', zorder = 0, linewidth = 1.0)
    ax6.set_title(r'$\rm ACF \,\, for \,\, determining \,\, \Delta\nu$')
    ax6.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
    ax6.set_xlim([min(star.lag), max(star.lag)])
    ax6.set_ylim([min(star.auto)-0.05*(max(star.auto)-min(star.auto)), max(star.auto)+0.1*(max(star.auto)-min(star.auto))])

    # dnu fit
    ax7 = fig.add_subplot(x, y, 7)
    ax7.plot(star.zoom_lag, star.zoom_auto, 'w-', zorder=0, linewidth=1.0)
    ax7.axvline(star.obs_dnu, color='lime', linestyle='--', linewidth=1.5, zorder=2)
    ax7.plot(star.new_lag, star.dnu_fit, color='lime', linewidth=1.5)
    ax7.axvline(star.exp_dnu, color='red', linestyle=':', linewidth=1.5, zorder=5)
    ax7.set_title(r'$\rm \Delta\nu \,\, fit$')
    ax7.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
    ax7.annotate(r'$\Delta\nu = %.2f$'%star.obs_dnu, xy=(0.025, 0.85), xycoords="axes fraction", fontsize=18, color='lime')
    ax7.set_xlim([min(star.zoom_lag), max(star.zoom_lag)])

    if star.globe['interp_ech']:
        interpolation='bilinear'
    else:
        interpolation='nearest'
    # echelle diagram
    ax8 = fig.add_subplot(x, y, 8)
    ax8.imshow(star.z, extent=star.extent, interpolation=interpolation, aspect='auto', origin='lower', cmap=plt.get_cmap(star.globe['cmap']))
    ax8.axvline([star.obs_dnu], color='white', linestyle='--', linewidth=1.5, dashes=(5, 5))
    ax8.set_title(r'$\rm \grave{E}chelle \,\, diagram$')
    ax8.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$'%star.obs_dnu)
    ax8.set_ylabel(r'$\rm \nu \,\, [\mu Hz]$')
    ax8.set_xlim([star.extent[0], star.extent[1]])
    ax8.set_ylim([star.extent[2], star.extent[3]])

    yrange = max(star.yax)-min(star.yax)
    ax9 = fig.add_subplot(x, y, 9)
    ax9.plot(star.xax, star.yax, color='white', linestyle='-', linewidth=0.75)
    ax9.set_title(r'$\rm Collapsed \,\, \grave{e}chelle \,\, diagram$')
    ax9.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$'%star.obs_dnu)
    ax9.set_ylabel(r'$\rm Collapsed \,\, power$')
    ax9.set_xlim([0.0, 2.0*star.obs_dnu])
    ax9.set_ylim([min(star.yax)-0.025*(yrange), max(star.yax)+0.05*(yrange)])

    plt.tight_layout()
    if star.params['save']:
        if not star.params['overwrite']:
            plt.savefig(utils.get_next(star,'global_fit.png'), dpi=300)
        else:
            plt.savefig(os.path.join(star.params[star.name]['path'],'global_fit.png'), dpi=300)
    if not star.params['cli']:
        with open('global_fit.pickle','wb') as f:
            pickle.dump(fig, f)
        star.pickles.append('global_fit.pickle')
    if not star.params['show']:
        plt.close()
    if star.params['testing']:
        dnu_comparison(star)


def plot_samples(star):
    """
    Plot results of the Monte-Carlo sampling.

    Parameters
    ----------
    star : target.Target
        the pySYD pipeline object

    Returns
    -------
    None
    
    """
    npanels = len(star.df.columns.values.tolist())
    d = utils.get_dict(type='plots')
    params = utils.get_dict()
    x, y = d[npanels]['x'], d[npanels]['y']
    fig = plt.figure("Posteriors for %s"%star.name, figsize=d[npanels]['size'])
    for i, col in enumerate(star.df.columns.values.tolist()):
        ax = plt.subplot(x, y, i+1)
        ax.hist(star.df[col], bins=20, color='cyan', histtype='step', lw=2.5, facecolor='0.75')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_title(params[col]['label'], fontsize=16)
    plt.tight_layout()
    if star.params['save']:
        if not star.params['overwrite']:
            plt.savefig(utils.get_next(star,'samples.png'), dpi=300)
        else:
            plt.savefig(os.path.join(star.params[star.name]['path'],'samples.png'), dpi=300)
    if not star.params['cli']:
        with open('samples.pickle','wb') as f:
            pickle.dump(fig, f)
        star.pickles.append('samples.pickle')
    if not star.params['show']:
        plt.close()


def _plot_fits(star, color='lime'):

    npanels=len(star.models)+1
    d = utils.get_dict(type='plots')
    x, y = d[npanels]['x'], d[npanels]['y']
    fig = plt.figure("Different fits for %s"%star.name, figsize=d[npanels]['size'])
    ax = fig.add_subplot(x, y, 1)
    ax.plot(star.frequency, star.random_pow, c='lightgrey', zorder=0, alpha=0.5)
    ax.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.random_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'w-', zorder=1)
    ax.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.random_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'w-', zorder=1)
    ax.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.smooth_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
    ax.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.smooth_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
    total = np.zeros_like(star.frequency)
    for r in range(star.nlaws_orig):
        model = models.background(star.frequency, [star.b_orig[r], star.a_orig[r]])
        ax.plot(star.frequency, model, color='blue', linestyle=':', linewidth=1.5, zorder=4)
        total += model
    total += star.noise
    ax.plot(star.frequency, total, color='blue', linewidth=2., zorder=5)
    ax.errorbar(star.bin_freq, star.bin_pow, yerr=star.bin_err, color='lime', markersize=0., fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
    ax.axvline(star.params[star.name]['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax.axvline(star.params[star.name]['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax.axhline(star.noise, color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5, 5))
    ax.set_xlim([min(star.frequency), max(star.frequency)])
    ax.set_ylim([min(star.power), max(star.power)*1.25])
    ax.set_title(r'$\rm Initial \,\, guesses$')
    ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    for n, mm in enumerate(star.models):
        ax = fig.add_subplot(x, y, n+2)
        ax.plot(star.frequency, star.random_pow, c='lightgrey', zorder=0, alpha=0.5)
        ax.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.random_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'w-', zorder=1)
        ax.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.random_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'w-', zorder=1)
        ax.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.smooth_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
        ax.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.smooth_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
        pars = star.paras[n]
        total = np.zeros_like(star.frequency)
        if len(pars) > 1:
            for r in range(mm//2):
                yobs = models.background(star.frequency, [pars[r*2], pars[r*2+1]])
                ax.plot(star.frequency, yobs, color='blue', linestyle=':', linewidth=1.5, zorder=4)
                total += yobs
        if len(pars)%2 == 0:
            total += star.noise
            ax.axhline(star.noise, color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
        else:
            total += pars[-1]
            ax.axhline(pars[-1], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
        ax.plot(star.frequency, total, color='blue', linewidth=2., zorder=5)
        ax.errorbar(star.bin_freq, star.bin_pow, yerr=star.bin_err, color='lime', markersize=0., fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
        ax.axvline(star.params[star.name]['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
        ax.axvline(star.params[star.name]['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
        ax.set_xlim([min(star.frequency), max(star.frequency)])
        ax.set_ylim([min(star.power), max(star.power)*1.25])
        if mm%2 == 0:
            wn='fixed'
        else:
            wn='free'
        ax.set_title(r'$\rm nlaws=%s \,\, | \,\, wn=%s $'%(str(int(mm//2)),wn))
        ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Highlight selected model
        if mm == star.model:
            ax.spines['bottom'].set_color(color)
            ax.spines['top'].set_color(color) 
            ax.spines['right'].set_color(color)
            ax.spines['left'].set_color(color)
            ax.tick_params(axis='both', which='both', colors=color)
            ax.yaxis.label.set_color(color)
            ax.xaxis.label.set_color(color)
            ax.title.set_color(color)
    plt.tight_layout()
    if star.params['save']:
        if not star.params['overwrite']:
            plt.savefig(utils.get_next(star,'model_fits.png'), dpi=300)
        else:
            plt.savefig(os.path.join(star.params[star.name]['path'],'model_fits.png'), dpi=300)
    if not star.params['cli']:
        with open('model_fits.pickle','wb') as f:
            pickle.dump(fig, f)
        star.pickles.append('model_fits.pickle')
    if not star.params['show']:
        plt.close()


def _time_series(star, npanels=1):

    d = utils.get_dict(type='plots')
    x, y = d[npanels]['x'], d[npanels]['y']
    fig = plt.figure("%s time series"%star.name, figsize=d[npanels]['size'])
    ax = plt.subplot(x,y,1)
    ax.plot(star.time, star.flux, 'w-')
    ax.set_xlim([min(star.time), max(star.time)])
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')  
    ax.tick_params(labelsize=22)
    plt.xlabel(r'$\rm Time \,\, [days]$', fontsize=28)
    plt.ylabel(r'$\rm Normalized \,\, flux$', fontsize=28)
    plt.tight_layout()
    if not star.params['cli']:
        with open('lc.pickle','wb') as f:
            pickle.dump(fig, f)
    if not star.params['show']:
        plt.close()


def _frequency_series(star, npanels=1):

    d = utils.get_dict(type='plots')
    x, y = d[npanels]['x'], d[npanels]['y']
    fig = plt.figure("%s power spectrum"%star.name, figsize=d[npanels]['size'])
    ax = plt.subplot(1,1,1)
    ax.plot(star.frequency, star.power, 'w-')
    ax.set_xlim([min(star.frequency), max(star.frequency)])
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')  
    ax.tick_params(labelsize=22)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize=28)
    plt.ylabel(r'$\rm Power \,\, [ppm^2 \, \mu Hz^{-1}]$', fontsize=28)
    plt.tight_layout()
    if not star.params['cli']:
        with open('ps.pickle','wb') as f:
            pickle.dump(fig, f)
    if not star.params['show']:
        plt.close()


def _dnu_comparison(star, methods=['M','A','D'], markers=['o','D','^'], styles=['--','-.',':'],
                   colors=['#FF9408','#00A9E0','g'], names=['Maryum','Ashley','Dennis'], npanels=2):

    sig = 0.35*star.exp_dnu/2.35482 
    weights = 1./(sig*np.sqrt(2.*np.pi))*np.exp(-(star.lag-star.exp_dnu)**2./(2.*sig**2))
    new_weights = weights/max(weights)
    weighted_acf = star.auto*weights
    obs_dnu=star.globe['results'][star.name]['dnu'][0]

    d = utils.get_dict(type='plots')
    y, x = d[npanels]['x'], d[npanels]['y']
    fig = plt.figure("Dnu trials for %s"%star.name, figsize=(12,12))
    ax1 = fig.add_subplot(x, y, 1)
    ax1.plot(star.lag, star.auto, 'w-', zorder=0, linewidth=1.)
    ax1.plot(star.lag, new_weights, c='yellow', linestyle=':', zorder = 0, linewidth = 1.0)
    ax1.axvline(star.exp_dnu, color='red', linestyle=':', linewidth=1.5, zorder=5)
    ax1.set_title(r'$\rm ACF$', fontsize=28)
    ax1.set_xlim([min(star.lag), max(star.lag)])
    ax1.set_ylim([min(star.auto)-0.05*(max(star.auto)-min(star.auto)), max(star.auto)+0.1*(max(star.auto)-min(star.auto))])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    ax2 = fig.add_subplot(x, y, 2)
    ax2.plot(star.lag, weighted_acf, 'w-', zorder=0, linewidth=1.)
    ax2.axvline(star.exp_dnu, color='red', linestyle=':', linewidth=1.5, zorder=5)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$', fontsize=28)
    ax2.set_xlim([min(star.lag), max(star.lag)])
    ax2.set_ylim([min(weighted_acf)-0.05*(max(weighted_acf)-min(weighted_acf)), max(weighted_acf)+0.1*(max(weighted_acf)-min(weighted_acf))])

    method_0 = star.globe['method']
    for m, method in enumerate(methods):
        star.globe['method'] = method
        star.initial_dnu()
        star.get_acf_cutout()
        star.globe['results'][star.name]['dnu'] = star.globe['results'][star.name]['dnu'][:-1]
        ax1.scatter(star.peaks_l, star.peaks_a, s=30.0, edgecolor=colors[m], marker=markers[m], facecolor='none', linewidths=1.0, label=r'$\rm %s$'%method)
        ax1.axvline(star.obs_dnu, linestyle=styles[m], color=colors[m], linewidth=2.5, zorder=2)
        for lag in star.peaks_l:
            ax2.axvline(lag, linestyle=styles[m], color=colors[m], linewidth=0.75, zorder=2)
        ax2.axvline(star.obs_dnu, linestyle=styles[m], color=colors[m], linewidth=2.5, zorder=2, label=r'$\rm \Delta\nu \,\, %s$'%names[m])
        ax2.axvspan(min(star.zoom_lag), max(star.zoom_lag), color=colors[m], alpha=0.25, zorder=0)
    star.globe['method'] = method_0
    ax1.axvline(star.globe['results'][star.name]['dnu'][0], linestyle='-', color='lime', linewidth=2.5, zorder=99, label=r'$\rm Observed \,\, \Delta\nu$')
    ax2.axvline(star.globe['results'][star.name]['dnu'][0], linestyle='-', color='lime', linewidth=2.5, zorder=99, label=r'$\rm Observed \,\, \Delta\nu$')
    ax1.legend(fontsize=24, loc='upper right', scatteryoffsets=[0.5], handletextpad=0.25, markerscale=1.5, handlelength=0.75, labelspacing=0.3, columnspacing=0.1)
    ax2.legend(fontsize=24)
    plt.tight_layout()
    if star.params['save']:
        if not star.params['overwrite']:
            plt.savefig(utils.get_next(star,'dnu_comparisons.png'), dpi=300)
        else:
            plt.savefig(os.path.join(star.params[star.name]['path'],'dnu_comparisons.png'), dpi=300)
    if not star.params['show']:
        plt.close()
