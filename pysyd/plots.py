import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel

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



def plot_excess(star):
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

    plt.figure(figsize=(12,8))

    # Time series data
    ax1 = plt.subplot(2, 3, 1)
    if star.lc:
        ax1.plot(star.time, star.flux, 'w-')
        ax1.set_xlim([min(star.time), max(star.time)])
    ax1.set_title(r'$\rm Time \,\, series$')
    ax1.set_xlabel(r'$\rm Time \,\, [days]$')
    ax1.set_ylabel(r'$\rm Flux$')

    # log-log power spectrum with crude background fit
    ax2 = plt.subplot(2, 3, 2)
    ax2.loglog(star.freq, star.pow, 'w-')
    ax2.set_xlim([min(star.freq), max(star.freq)])
    ax2.set_ylim([min(star.pow), max(star.pow)*1.25])
    ax2.set_title(r'$\rm Crude \,\, background \,\, fit$')
    ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    if star.findex['binning'] is not None:
        ax2.loglog(star.bin_freq, star.bin_pow, 'r-')
    ax2.loglog(star.freq, star.interp_pow, color='lime', linestyle='-', lw=2.0)

    # Crude background-corrected power spectrum
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(star.freq, star.bgcorr_pow, 'w-')
    ax3.set_xlim([min(star.freq), max(star.freq)])
    ax3.set_ylim([0.0, max(star.bgcorr_pow)*1.25])
    ax3.set_title(r'$\rm Background \,\, corrected \,\, PS$')
    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')

    # ACF trials to determine numax
    for i in range(star.findex['n_trials']):
        ax = plt.subplot(2, 3, 4+i)
        ax.plot(star.findex['results'][star.name][i+1]['x'], star.findex['results'][star.name][i+1]['y'], 'w-')
        xran = max(star.findex['results'][star.name][i+1]['fitx'])-min(star.findex['results'][star.name][i+1]['fitx'])
        ymax = star.findex['results'][star.name][i+1]['maxy']
        ax.axvline(star.findex['results'][star.name][i+1]['maxx'], linestyle='dotted', color='r', linewidth=0.75)
        ax.set_title(r'$\rm Collapsed \,\, ACF \,\, [trial \,\, %d]$' % (i+1))
        ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax.set_ylabel(r'$\rm Arbitrary \,\, units$')
        if star.findex['results'][star.name][i+1]['good_fit']:
            ax.plot(star.findex['results'][star.name][i+1]['fitx'], star.findex['results'][star.name][i+1]['fity'], color='lime', linestyle='-', linewidth=1.5)
            if max(star.findex['results'][star.name][i+1]['fity']) > star.findex['results'][star.name][i+1]['maxy']:
                ymax = max(star.findex['results'][star.name][i+1]['fity'])
            ax.axvline(star.findex['results'][star.name][i+1]['numax'], color='lime', linestyle='--', linewidth=0.75)
        yran = np.absolute(ymax)
        ax.set_xlim([min(star.findex['results'][star.name][i+1]['x']), max(star.findex['results'][star.name][i+1]['x'])])
        ax.set_ylim([-0.05, ymax+0.15*yran])
        ax.annotate(r'$\rm SNR = %3.2f$' % star.findex['results'][star.name][i+1]['snr'], xy=(min(star.findex['results'][star.name][i+1]['fitx'])+0.05*xran, ymax+0.025*yran), fontsize=18)

    plt.tight_layout()
    if star.params['save']:
        plt.savefig(os.path.join(star.params[star.name]['path'],'excess.png'), dpi=300)
    if star.params['show']:
        plt.show(block=False)
    if star.verbose:
        print('----------------------------------------------------')
        input('------------------- press RETURN -------------------')
    plt.close()


def plot_background(star, n_peaks=10):
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

    from pysyd.functions import return_max, max_elements
    from pysyd import models

    exp_numax=star.fitbg['results'][star.name]['numax_gauss'][0]
    exp_dnu=0.22*(exp_numax**0.797)
    obs_dnu=star.fitbg['results'][star.name]['dnu'][0]

    fig = plt.figure(figsize=(12, 12))
    # Time series data
    ax1 = fig.add_subplot(3, 3, 1)
    if star.lc:
        ax1.plot(star.time, star.flux, 'w-')
        ax1.set_xlim([min(star.time), max(star.time)])
    ax1.set_title(r'$\rm Time \,\, series$')
    ax1.set_xlabel(r'$\rm Time \,\, [days]$')
    ax1.set_ylabel(r'$\rm Flux$')

    # Initial background guesses
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(star.frequency, star.random_pow, c='lightgrey', zorder=0, alpha=0.5)
    ax2.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.random_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'w-', zorder=1)
    ax2.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.random_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'w-', zorder=1)
    ax2.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.smooth_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
    ax2.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.smooth_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
    total = np.zeros_like(star.frequency)
    for r in range(star.nlaws_orig):
        model = models.harvey(star.frequency, [star.b_orig[r], star.a_orig[r], star.noise])
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

    lower = (min(star.frequency)+exp_numax)/2.
    upper = (max(star.frequency)+exp_numax)/2.
    mask = (star.frequency >= lower)&(star.frequency <= upper)
    # Fitted background
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(star.frequency, star.random_pow, c='lightgrey', zorder=0, alpha=0.5)
    ax3.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.random_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'w-', linewidth=0.75, zorder=1)
    ax3.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.random_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'w-', linewidth=0.75, zorder=1)
    ax3.plot(star.frequency[star.frequency < star.params[star.name]['ps_mask'][0]], star.smooth_pow[star.frequency < star.params[star.name]['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
    ax3.plot(star.frequency[star.frequency > star.params[star.name]['ps_mask'][1]], star.smooth_pow[star.frequency > star.params[star.name]['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
    if star.nlaws != 0:
        for r in range(star.nlaws):
            ax3.plot(star.frequency, models.harvey(star.frequency, [star.pars[2*r], star.pars[2*r+1], star.pars[-1]]), color='blue', linestyle=':', linewidth=1.5, zorder=4)
        ax3.plot(star.frequency, models.harvey(star.frequency, star.pars, total=True), color='blue', linewidth=2.0, zorder=5)
    ax3.errorbar(star.bin_freq, star.bin_pow, yerr=star.bin_err, color='lime', markersize=0.0, fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
    ax3.axvline(star.params[star.name]['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax3.axvline(star.params[star.name]['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax3.axhline(star.pars[-1], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5, 5))
    ax3.plot(star.frequency[mask], star.pssm[mask], color='yellow', linewidth=2.0, linestyle='dashed', zorder=6)
    ax3.set_xlim([min(star.frequency), max(star.frequency)])
    ax3.set_ylim([min(star.power), max(star.power)*1.25])
    ax3.set_title(r'$\rm Fitted \,\, model$')
    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # Smoothed power excess w/ gaussian
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(star.region_freq, star.region_pow, 'w-', zorder=0)
    idx = return_max(star.region_freq, star.region_pow, index=True)
    ax4.plot([star.region_freq[idx]], [star.region_pow[idx]], color='red', marker='s', markersize=7.5, zorder=0)
    ax4.axvline([star.region_freq[idx]], color='white', linestyle='--', linewidth=1.5, zorder=0)
    ax4.plot(star.new_freq, star.numax_fit, 'b-', zorder=3)
    ax4.axvline(exp_numax, color='blue', linestyle=':', linewidth=1.5, zorder=2)
    ax4.plot([exp_numax], [max(star.numax_fit)], color='b', marker='D', markersize=7.5, zorder=1)
    ax4.set_title(r'$\rm Smoothed \,\, bg$-$\rm corrected \,\, PS$')
    ax4.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax4.set_xlim([min(star.region_freq), max(star.region_freq)])

    # Background-corrected power spectrum with n highest peaks
    mask = np.ma.getmask(np.ma.masked_inside(star.frequency, star.params[star.name]['ps_mask'][0], star.params[star.name]['ps_mask'][1]))
    star.freq = star.frequency[mask]
    star.psd = star.bg_corr_smooth[mask]
    peaks_f, peaks_p = max_elements(star.freq, star.psd, n_peaks)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(star.freq, star.psd, 'w-', zorder=0, linewidth=1.0)
    ax5.scatter(peaks_f, peaks_p, s=25.0, edgecolor='r', marker='s', facecolor='none', linewidths=1.0)
    ax5.set_title(r'$\rm Bg$-$\rm corrected \,\, PS$')
    ax5.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax5.set_ylabel(r'$\rm Power$')
    ax5.set_xlim([star.params[star.name]['ps_mask'][0], star.params[star.name]['ps_mask'][1]])
    ax5.set_ylim([min(star.psd)-0.025*(max(star.psd)-min(star.psd)), max(star.psd)+0.1*(max(star.psd)-min(star.psd))])

    sig = 0.35*exp_dnu/2.35482 
    weights = 1./(sig*np.sqrt(2.*np.pi))*np.exp(-(star.lag-exp_dnu)**2./(2.*sig**2))
    new_weights = weights/max(weights)
    # ACF for determining dnu
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(star.lag, star.auto, 'w-', zorder=0, linewidth=1.)
    ax6.scatter(star.peaks_l, star.peaks_a, s=30.0, edgecolor='r', marker='^', facecolor='none', linewidths=1.0)
    ax6.axvline(exp_dnu, color='red', linestyle=':', linewidth=1.5, zorder=5)
    ax6.axvline(obs_dnu, color='lime', linestyle='--', linewidth=1.5, zorder=2)
    ax6.scatter(star.best_lag, star.best_auto, s=45.0, edgecolor='lime', marker='s', facecolor='none', linewidths=1.0)
    ax6.plot(star.zoom_lag, star.zoom_auto, 'r-', zorder=5, linewidth=1.0)
    ax6.plot(star.lag, new_weights, c='yellow', linestyle=':', zorder = 0, linewidth = 1.0)
    ax6.set_title(r'$\rm ACF \,\, for \,\, determining \,\, \Delta\nu$')
    ax6.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
    ax6.set_xlim([min(star.lag), max(star.lag)])
    ax6.set_ylim([min(star.auto)-0.05*(max(star.auto)-min(star.auto)), max(star.auto)+0.1*(max(star.auto)-min(star.auto))])

    # dnu fit
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(star.zoom_lag, star.zoom_auto, 'w-', zorder=0, linewidth=1.0)
    ax7.axvline(obs_dnu, color='lime', linestyle='--', linewidth=1.5, zorder=2)
    ax7.plot(star.new_lag, star.dnu_fit, color='lime', linewidth=1.5)
    ax7.axvline(exp_dnu, color='red', linestyle=':', linewidth=1.5, zorder=5)
    ax7.set_title(r'$\rm \Delta\nu \,\, fit$')
    ax7.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
    ax7.annotate(r'$\Delta\nu = %.2f$'%obs_dnu, xy=(0.025, 0.85), xycoords="axes fraction", fontsize=18, color='lime')
    ax7.set_xlim([min(star.zoom_lag), max(star.zoom_lag)])

    if star.globe['interp_ech']:
        interpolation='bilinear'
    else:
        interpolation='none'
    # echelle diagram
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(star.ech, extent=star.extent, interpolation=interpolation, aspect='auto', origin='lower', cmap=plt.get_cmap('viridis'))
    ax8.axvline([obs_dnu], color='white', linestyle='--', linewidth=1.0, dashes=(5, 5))
    ax8.set_title(r'$\rm \grave{E}chelle \,\, diagram$')
    ax8.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$'%obs_dnu)
    ax8.set_ylabel(r'$\rm \nu \,\, [\mu Hz]$')
    ax8.set_xlim([0.0, 2.0*obs_dnu])
    ax8.set_ylim([star.params[star.name]['ps_mask'][0], star.params[star.name]['ps_mask'][1]])

    yrange = max(star.yax)-min(star.yax)
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(star.xax, star.yax, color='white', linestyle='-', linewidth=0.75)
    ax9.set_title(r'$\rm Collapsed \,\, \grave{e}chelle \,\, diagram$')
    ax9.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$'%obs_dnu)
    ax9.set_ylabel(r'$\rm Collapsed \,\, power$')
    ax9.set_xlim([0.0, 2.0*obs_dnu])
    ax9.set_ylim([min(star.yax)-0.025*(yrange), max(star.yax)+0.05*(yrange)])

    plt.tight_layout()
    if star.params['save']:
        if star.globe['interp_ech']:
            plt.savefig(os.path.join(star.params[star.name]['path'],'background_sm_ech.png'), dpi=300)
        else:
            plt.savefig(os.path.join(star.params[star.name]['path'],'background.png'), dpi=300)
    if star.params['show']:
        plt.show(block=False)
    if star.verbose:
        print('----------------------------------------------------')
        input('------------------- press RETURN -------------------')
    plt.close()


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

    n = len(star.df.columns.values.tolist())
    plt.figure(figsize=(12, 8))
    params = utils.get_params_dict()
    if n <= 3:
        x, y = 3, 1
    elif n > 3 and n <= 6:
        x, y = 3, 2
    elif n > 6 and n <= 9:
        x, y = 3, 3
    elif  n > 9 and n <= 12:
        x, y = 4, 3
    elif n > 12 and n <= 16:
        x, y = 4, 4
    else:
        x, y = 5, 4
    for i, col in enumerate(star.df.columns.values.tolist()):
        ax = plt.subplot(y, x, i+1)
        ax.hist(star.df[col], bins=20, color='cyan', histtype='step', lw=2.5, facecolor='0.75')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_title(params[col]['label'], fontsize=16)
    plt.tight_layout()
    if star.params['save']:
        plt.savefig(os.path.join(star.params[star.name]['path'],'samples.png'), dpi=300)
    if star.params['show']:
        plt.show(block=False)
    plt.close()
