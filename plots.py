import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator, ScalarFormatter

from functions import *
from models import *


def set_plot_params():
    """Sets the matplotlib parameters."""

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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """TODO: Write description."""

    import matplotlib.colors as mcolors
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n))
    )

    return new_cmap

def plot_excess(target):
    """Creates a plot summarising the results of the find excess routine."""

    plt.figure(figsize=(12,8))

    # Time series data
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(target.time, target.flux, 'w-')
    ax1.set_xlim([min(target.time), max(target.time)])
    ax1.set_title(r'$\rm Time \,\, series$')
    ax1.set_xlabel(r'$\rm Time \,\, [days]$')
    ax1.set_ylabel(r'$\rm Flux$')

    # log-log power spectrum with crude background fit
    ax2 = plt.subplot(2, 3, 2)
    ax2.loglog(target.freq, target.pow, 'w-')
    ax2.set_xlim([min(target.freq), max(target.freq)])
    ax2.set_ylim([min(target.pow), max(target.pow)*1.25])
    ax2.set_title(r'$\rm Crude \,\, background \,\, fit$')
    ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    if target.findex['binning'] is not None:
        ax2.loglog(target.bin_freq, target.bin_pow, 'r-')
    ax2.loglog(target.freq, target.interp_pow, color='lime', linestyle='-', lw=2.0)

    # Crude background-corrected power spectrum
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(target.freq, target.bgcorr_pow, 'w-')
    ax3.set_xlim([min(target.freq), max(target.freq)])
    ax3.set_ylim([0.0, max(target.bgcorr_pow)*1.25])
    ax3.set_title(r'$\rm Background \,\, corrected \,\, PS$')
    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')

    # ACF trials to determine numax
    for i in range(target.findex['n_trials']):
        ax = plt.subplot(2, 3, 4+i)
        ax.plot(target.findex['results'][i+1]['x'], target.findex['results'][i+1]['y'], 'w-')
        xran = max(target.findex['results'][i+1]['fitx'])-min(target.findex['results'][i+1]['fitx'])
        ymax = target.findex['results'][i+1]['maxy']
        ax.axvline(target.findex['results'][i+1]['maxx'], linestyle='dotted', color='r', linewidth=0.75)
        ax.set_title(r'$\rm Collapsed \,\, ACF \,\, [trial \,\, %d]$' % (i+1))
        ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax.set_ylabel(r'$\rm Arbitrary \,\, units$')
        if target.findex['results'][i+1]['good_fit']:
            ax.plot(target.findex['results'][i+1]['fitx'], target.findex['results'][i+1]['fity'], color='lime', linestyle='-', linewidth=1.5)
            if max(target.findex['results'][i+1]['fity']) > target.findex['results'][i+1]['maxy']:
                ymax = max(target.findex['results'][i+1]['fity'])
            ax.axvline(target.findex['results'][i+1]['numax'], color='lime', linestyle='--', linewidth=0.75)
        yran = np.absolute(ymax)
        ax.set_xlim([min(target.findex['results'][i+1]['x']), max(target.findex['results'][i+1]['x'])])
        ax.set_ylim([-0.05, ymax+0.15*yran])
        ax.annotate(r'$\rm SNR = %3.2f$' % target.findex['results'][i+1]['snr'], xy=(min(target.findex['results'][i+1]['fitx'])+0.05*xran, ymax+0.025*yran), fontsize=18)

    plt.tight_layout()
    # Save
    if target.params['save']:
        plt.savefig('%sexcess.png' % target.params[target.target]['path'], dpi=300)
    # Show plots
    if target.params['show']:
        plt.show()
    plt.close()

def plot_background(target):
    """Creates a plot summarising the results of the fit background routine."""

    fig = plt.figure(figsize=(12, 12))

    # Time series data
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(target.time, target.flux, 'w-')
    ax1.set_xlim([min(target.time), max(target.time)])
    ax1.set_title(r'$\rm Time \,\, series$')
    ax1.set_xlabel(r'$\rm Time \,\, [days]$')
    ax1.set_ylabel(r'$\rm Flux$')

    # Initial background guesses
    target.smooth_power = convolve(target.power, Box1DKernel(int(np.ceil(target.fitbg['box_filter']/target.resolution))))
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(target.frequency[target.frequency < target.maxpower[0]], target.power[target.frequency < target.maxpower[0]], 'w-', zorder=0)
    ax2.plot(target.frequency[target.frequency > target.maxpower[1]], target.power[target.frequency > target.maxpower[1]], 'w-', zorder=0)
    ax2.plot(target.frequency[target.frequency < target.maxpower[0]], target.smooth_power[target.frequency < target.maxpower[0]], 'r-', linewidth=0.75, zorder=1)
    ax2.plot(target.frequency[target.frequency > target.maxpower[1]], target.smooth_power[target.frequency > target.maxpower[1]], 'r-', linewidth=0.75, zorder=1)
    for r in range(target.nlaws):
        ax2.plot(target.frequency, harvey(target.frequency, [target.a_orig[r], target.b_orig[r], target.noise]), color='blue', linestyle=':', linewidth=1.5, zorder=3)
    ax2.plot(target.frequency, harvey(target.frequency, target.pars, total=True), color='blue', linewidth=2., zorder=4)
    ax2.errorbar(target.bin_freq, target.bin_pow, yerr=target.bin_err, color='lime', markersize=0., fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=2)
    for m, n in zip(target.mnu_orig, target.a_orig):
        ax2.plot(m, n, color='blue', fillstyle='none', mew=3.0, marker='s', markersize=5.0)
    ax2.axvline(target.maxpower[0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5, 5))
    ax2.axvline(target.maxpower[1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5, 5))
    ax2.axhline(target.noise, color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5, 5))
    ax2.set_xlim([min(target.frequency), max(target.frequency)])
    ax2.set_ylim([min(target.power), max(target.power)*1.25])
    ax2.set_title(r'$\rm Initial \,\, guesses$')
    ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Fitted background
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(target.frequency[target.frequency < target.maxpower[0]], target.power[target.frequency < target.maxpower[0]], 'w-', linewidth=0.75, zorder=0)
    ax3.plot(target.frequency[target.frequency > target.maxpower[1]], target.power[target.frequency > target.maxpower[1]], 'w-', linewidth=0.75, zorder=0)
    ax3.plot(target.frequency[target.frequency < target.maxpower[0]], target.smooth_power[target.frequency < target.maxpower[0]], 'r-', linewidth=0.75, zorder=1)
    ax3.plot(target.frequency[target.frequency > target.maxpower[1]], target.smooth_power[target.frequency > target.maxpower[1]], 'r-', linewidth=0.75, zorder=1)
    for r in range(target.nlaws):
        ax3.plot(target.frequency, harvey(target.frequency, [target.pars[2*r], target.pars[2*r+1], target.pars[-1]]), color='blue', linestyle=':', linewidth=1.5, zorder=3)
    ax3.plot(target.frequency, harvey(target.frequency, target.pars, total=True), color='blue', linewidth=2.0, zorder=4)
    ax3.errorbar(target.bin_freq, target.bin_pow, yerr=target.bin_err, color='lime', markersize=0.0, fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=2)
    ax3.axvline(target.maxpower[0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5, 5))
    ax3.axvline(target.maxpower[1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5, 5))
    ax3.axhline(target.pars[-1], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5, 5))
    ax3.plot(target.frequency, target.pssm, color='yellow', linewidth=2.0, linestyle='dashed', zorder=5)
    ax3.set_xlim([min(target.frequency), max(target.frequency)])
    ax3.set_ylim([min(target.power), max(target.power)*1.25])
    ax3.set_title(r'$\rm Fitted \,\, model$')
    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # Smoothed power excess w/ gaussian
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(target.region_freq, target.region_pow, 'w-', zorder=0)
    idx = return_max(target.region_pow, index=True)
    ax4.plot([target.region_freq[idx]], [target.region_pow[idx]], color='red', marker='s', markersize=7.5, zorder=0)
    ax4.axvline([target.region_freq[idx]], color='white', linestyle='--', linewidth=1.5, zorder=0)
    ax4.plot(target.new_freq, target.numax_fit, 'b-', zorder=3)
    ax4.axvline(target.exp_numax, color='blue', linestyle=':', linewidth=1.5, zorder=2)
    ax4.plot([target.exp_numax], [max(target.numax_fit)], color='b', marker='D', markersize=7.5, zorder=1)
    ax4.set_title(r'$\rm Smoothed \,\, bg$-$\rm corrected \,\, PS$')
    ax4.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax4.set_xlim([min(target.region_freq), max(target.region_freq)])

    # Background-corrected power spectrum with n highest peaks
    target.freq = target.frequency[target.params[target.target]['mask']]
    target.psd = target.bg_corr_smooth[target.params[target.target]['mask']]
    peaks_f, peaks_p = max_elements(target.freq, target.psd, target.fitbg['n_peaks'])
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(target.freq, target.psd, 'w-', zorder=0, linewidth=1.0)
    ax5.scatter(peaks_f, peaks_p, s=25.0, edgecolor='r', marker='s', facecolor='none', linewidths=1.0)
    ax5.set_title(r'$\rm Bg$-$\rm corrected \,\, PS$')
    ax5.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax5.set_ylabel(r'$\rm Power$')
    ax5.set_xlim([min(target.region_freq), max(target.region_freq)])
    ax5.set_ylim([min(target.psd)-0.025*(max(target.psd)-min(target.psd)), max(target.psd)+0.1*(max(target.psd)-min(target.psd))])

    # ACF for determining dnu
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(target.lag, target.auto, 'w-', zorder=0, linewidth=1.)
    ax6.scatter(target.peaks_l, target.peaks_a, s=30.0, edgecolor='r', marker='^', facecolor='none', linewidths=1.0)
    ax6.axvline(target.exp_dnu, color='lime', linestyle='--', linewidth=1.5, zorder=5)
    # ax6.axvline(target.best_lag, color='red', linestyle='--', linewidth=1.5, zorder=2)
    ax6.scatter(target.best_lag, target.best_auto, s=45.0, edgecolor='lime', marker='s', facecolor='none', linewidths=1.0)
    ax6.plot(target.zoom_lag, target.zoom_auto, 'r-', zorder=5, linewidth=1.0)
    ax6.set_title(r'$\rm ACF \,\, for \,\, determining \,\, \Delta\nu$')
    ax6.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
    ax6.set_xlim([min(target.lag), max(target.lag)])
    ax6.set_ylim([min(target.auto)-0.05*(max(target.auto)-min(target.auto)), max(target.auto)+0.1*(max(target.auto)-min(target.auto))])

    # dnu fit
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(target.zoom_lag, target.zoom_auto, 'w-', zorder=0, linewidth=1.0)
    ax7.axvline(target.obs_dnu, color='red', linestyle='--', linewidth=1.5, zorder=2)
    ax7.plot(target.new_lag, target.dnu_fit, color='red', linewidth=1.5)
    ax7.axvline(target.exp_dnu, color='blue', linestyle=':', linewidth=1.5, zorder=5)
    ax7.set_title(r'$\rm \Delta\nu \,\, fit$')
    ax7.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
    ax7.annotate(r'$\Delta\nu = %.2f$' % target.obs_dnu, xy=(0.025, 0.85), xycoords="axes fraction", fontsize=18, color='lime')
    ax7.set_xlim([min(target.zoom_lag), max(target.zoom_lag)])

    #cmap = plt.get_cmap('jet')
    # new_cmap = cmap(np.linspace(0.1, 0.9, 100))
    #colors = truncate_colormap(cmap, 0.1, 0.8, 100)
    # echelle diagram
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(target.ech, extent=target.extent, interpolation='none', aspect='auto', origin='lower', cmap=plt.get_cmap('jet'))
    ax8.axvline([target.obs_dnu], color='white', linestyle='--', linewidth=1.0, dashes=(5, 5))
    ax8.set_title(r'$\rm \grave{E}chelle \,\, diagram$')
    ax8.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$' % target.obs_dnu)
    ax8.set_ylabel(r'$\rm \nu \,\, [\mu Hz]$')
    ax8.set_xlim([0.0, 2.0*target.obs_dnu])
    ax8.set_ylim([target.maxpower[0], target.maxpower[1]])

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(target.xax, target.yax, color='white', linestyle='-', linewidth=0.75)
    ax9.set_title(r'$\rm Collapsed \,\, \grave{e}chelle \,\, diagram$')
    ax9.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$' % target.obs_dnu)
    ax9.set_ylabel(r'$\rm Collapsed \,\, power$')
    ax9.set_xlim([0.0, 2.0*target.obs_dnu])
    ax9.set_ylim([
        min(target.yax) - 0.025*(max(target.yax) - min(target.yax)),
        max(target.yax) + 0.05*(max(target.yax) - min(target.yax))
    ])

    plt.tight_layout()
    # Save
    if target.params['save']:
        plt.savefig('%sbackground.png' % target.params[target.target]['path'], dpi=300)
    # Show plots
    if target.params['show']:
        plt.show()
    plt.close()

def plot_samples(target):
    """Plot results of the Monte-Carlo sampling."""

    plt.figure(figsize=(12, 8))
    panels = ['numax_smooth', 'amp_smooth', 'numax_gaussian', 'amp_gaussian', 'fwhm_gaussian', 'dnu']
    titles = [r'$\rm Smoothed \,\, \nu_{max} \,\, [\mu Hz]$', r'$\rm Smoothed \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$', r'$\rm Gaussian \,\, \nu_{max} \,\, [\mu Hz]$', r'$\rm Gaussian \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$', r'$\rm Gaussian \,\, FWHM \,\, [\mu Hz]$', r'$\rm \Delta\nu \,\, [\mu Hz]$']
    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        ax.hist(target.df[panels[i]], bins=20, color='cyan', histtype='step', lw=2.5, facecolor='0.75')
        ax.set_title(titles[i])

    plt.tight_layout()
    # Save
    if target.params['save']:
        plt.savefig('%ssamples.png' % target.params[target.target]['path'], dpi=300)
    # Show plots
    if target.params['show']:
        plt.show()
    plt.close()
