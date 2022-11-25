import os
import numpy as np
import matplotlib.pyplot as plt





# Package mode
from . import utils
from . import models
from . import PACKAGEDIR
from .utils import Question





MPLSTYLE = os.path.join(PACKAGEDIR,'data','pysyd.mplstyle')
plt.style.use(MPLSTYLE)

d = utils.get_dict(type='plots')



def make_plots(star, show_all=False,):
    """Make plots

    Function that establishes the default plotting parameters and then calls each
    of the relevant plotting routines

    Parameters
        star : pysyd.target.Target
            the pySYD pipeline object
        showall : bool, optional
            option to plot, save and show the different background models (default=`False`)

    Calls
        :mod:`pysyd.plots.plot_estimates`
        :mod:`pysyd.plots.plot_parameters`
        :mod:`pysyd.plots.plot_bgfits` [optional]
        :mod:`pysyd.plots.plot_samples`
    
    """
    if 'estimates' in star.params['plotting']:
        plot_estimates(star)
    if 'parameters' in star.params['plotting']:
        plot_parameters(star)
        if star.params['show_all']:
            plot_bgfits(star)
    if 'samples' in star.params['plotting']:
        plot_samples(star)
    if star.params['show']:
        plt.show(block=False)


def select_trial(star):
    r"""Select trial

    This is called when ``--ask`` is `True` (i.e. select which trial to use for :math:`\\rm \\nu_{max}`)
    This feature used to be called as part of a method in the :mod:`pysyd.target.Target` class but left a
    stale figure open -- this way it can be closed after the value is selected

    Parameters
        star : pysyd.target.Target
            the pySYD pipeline object

    Returns
        value : int or float
            depending on which `trial` was selected, this can be of integer or float type

    """
    x, y = d[star.params['n_trials']]['x'], d[star.params['n_trials']]['y']
    params = star.params['plotting']['estimates']

    fig = plt.figure("Numax guesses for %s"%star.name, figsize=d[star.params['n_trials']]['size'])
    # ACF trials to determine numax
    for i in range(star.params['n_trials']):
        ax = plt.subplot(x, y, 1+i)
        ax.plot(params[i]['x'], params[i]['y'], 'w-')
        xran = max(params[i]['fitx'])-min(params[i]['fitx'])
        ymax = params[i]['maxy']
        ax.axvline(params[i]['maxx'], linestyle='dotted', color='r', linewidth=0.75)
        ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        if params[i]['good_fit']:
            ax.plot(params[i]['fitx'], params[i]['fity'], color='lime', linestyle='-', linewidth=1.5)
            if max(params[i]['fity']) > params[i]['maxy']:
                ymax = max(params[i]['fity'])
            ax.axvline(params[i]['value'], color='lime', linestyle='--', linewidth=0.75)
            ax.set_title(r'$\rm Trial \,\, %d \,\, | \,\, \nu_{max} \sim %.0f \, \mu Hz $'%(i+1, params[i]['value']))
        yran = np.absolute(ymax)
        ax.set_xlim([min(params[i]['x']), max(params[i]['x'])])
        ax.set_ylim([-0.05, ymax+0.15*yran])
        ax.annotate(r'$\rm SNR = %3.2f$' % params[i]['snr'], xy=(min(params[i]['fitx'])+0.05*xran, ymax+0.025*yran), fontsize=18)
    plt.tight_layout()
    plt.show()
    value = Question().ask_integer('Which estimate would you like to use? ', special=True, n_trials=star.params['n_trials'])
    if isinstance(value, int):
        star.params['best'] = value
        print('Selecting model %d' % value)
    else:
        star.params['numax'] = value
        star.params['dnu'] = utils.delta_nu(value)
        print('Using numax of %.2f muHz as an initial guess' % value)
    plt.close()
    return star


def _select_trial2(star):
    """Select trial

    This is called when ``--ask`` is `True` (i.e. select which trial to use for :math:`\rm \nu_{max}`)
    This feature used to be called as part of a method in the `pysyd.target.Target` class but left a
    stale figure open -- this way it can be closed after the value is selected

    Parameters
        star : pysyd.target.Target
            the pySYD pipeline object

    Returns
        value : int or float
            depending on which `trial` was selected, this can be of integer or float type

    
    """
    x, y = d[star.params['n_trials2']]['x'], d[star.params['n_trials2']]['y']
    params = star.params['plotting']['trial']

    fig = plt.figure("Dnu guesses for %s"%star.name, figsize=d[star.params['n_trials2']]['size'])
    # ACF trials to determine numax
    for i in range(star.params['n_trials2']):
        ax = plt.subplot(x, y, 1+i)
        ax.plot(params[i]['x'], params[i]['y'], 'w-')
        xran = max(params[i]['fitx'])-min(params[i]['fitx'])
        ymax = params[i]['maxy']
        ax.axvline(params[i]['maxx'], linestyle='dotted', color='r', linewidth=0.75)
        ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        yran = np.absolute(ymax)
        ax.set_xlim([min(params[i]['x']), max(params[i]['x'])])
        ax.set_ylim([-0.05, ymax+0.15*yran])
        if params[i]['good_fit']:
            ax.plot(params[i]['fitx'], params[i]['fity'], color='lime', linestyle='-', linewidth=1.5)
            if max(params[i]['fity']) > params[i]['maxy']:
                ymax = max(params[i]['fity'])
            ax.axvline(params[i]['value'], color='lime', linestyle='--', linewidth=0.75)
            ax.set_title(r'$\rm Trial \,\, %d \,\, | \,\, \Delta\nu \sim %.0f \, \mu Hz $'%(i+1, params[i]['value']))
            ax.annotate(r'$\rm SNR = %3.2f$' % params[i]['snr'], xy=(min(params[i]['fitx'])+0.05*xran, ymax+0.025*yran), fontsize=18)
    plt.tight_layout()
    plt.show()
    value = utils._ask_int('Which estimate would you like to use? ', star.params['n_trials2'])
    if isinstance(value, int):
        star.params['best'] = value
        print('Selecting model %d' % value)
    else:
        star.params['dnu'] = value
        print('Using dnu of %.2f muHz as an initial guess' % value)
    plt.close()
    return star


def plot_estimates(star, filename='search_&_estimate.png', highlight=True, n=0):
    """Plot estimates

    Creates a plot summarizing the results of the find excess routine.

    Parameters
        star : pysyd.target.Target
            the pySYD pipeline object
        filename : str
            the path or extension to save the figure to
        highlight : bool, default=True
            option to highlight the selected estimate

    
    """
    n_panels = 3+star.params['n_trials']
    if not star.lc:
        n_panels -= 1
    x, y = d[n_panels]['x'], d[n_panels]['y']
    params = star.params['plotting']['estimates']

    fig = plt.figure("Estimates for %s"%star.name, figsize=d[n_panels]['size'])
    n += 1
    if 'time' in params:
        # PANEL 1: time series data
        ax1 = plt.subplot(y, x, n)
        ax1.plot(params['time'], params['flux'], 'w-')
        ax1.set_xlim([min(params['time']), max(params['time'])])
        ax1.set_title(r'$\rm Time \,\, series$')
        ax1.set_xlabel(r'$\rm Time \,\, [days]$')
        ax1.set_ylabel(r'$\rm Flux$')
    else:
        n -= 1

    n += 1
    # PANEL 2: log-log PS with crude background fit
    ax2 = plt.subplot(y, x, n)
    ax2.loglog(params['freq'], params['pow'], 'w-')
    ax2.set_xlim([min(params['freq']), max(params['freq'])])
    ax2.set_ylim([min(params['pow']), max(params['pow'])*1.25])
    ax2.set_title(r'$\rm Crude \,\, background \,\, fit$')
    ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax2.loglog(params['bin_freq'], params['bin_pow'], 'r-')
    ax2.loglog(params['freq'], params['interp_pow'], color='lime', linestyle='-', lw=2.0)

    n += 1
    # PANEL 3: background-corrected power spectrum
    ax3 = plt.subplot(y, x, n)
    ax3.plot(params['freq'], params['bgcorr_pow'], 'w-')
    ax3.set_xlim([min(params['freq']), max(params['freq'])])
    ax3.set_ylim([0.0, max(params['bgcorr_pow'])*1.25])
    ax3.set_title(r'$\rm Background \,\, corrected \,\, PS$')
    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')

    n += 1
    # PANEL 4-6: ACF trials 
    for i in range(star.params['n_trials']):
        ax = plt.subplot(y, x, n+i)
        ax.plot(params[i]['x'], params[i]['y'], 'w-')
        ymax = params[i]['maxy']
        ax.axvline(params[i]['maxx'], linestyle='dotted', color='r', linewidth=0.75)
        ax.set_ylabel(r'$\rm Collapsed \,\, ACF \,\, [trial \,\, %d]$' % (i+1))
        ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
#        ax.set_ylabel(r'$\rm Arbitrary \,\, units$')
        if params[i]['good_fit']:
            xran = max(params[i]['fitx'])-min(params[i]['fitx'])
            ax.plot(params[i]['fitx'], params[i]['fity'], color='lime', linestyle='-', linewidth=1.5)
            if max(params[i]['fity']) > params[i]['maxy']:
                ymax = max(params[i]['fity'])
            ax.axvline(params[i]['value'], color='lime', linestyle='--', linewidth=0.75)
        yran = np.absolute(ymax)
        ax.set_xlim([min(params[i]['x']), max(params[i]['x'])])
        ax.set_ylim([-0.05, ymax+0.15*yran])
        if 'best' in star.params and (i+1) == star.params['best'] and highlight:
            for spine in ['bottom','left','top','right']:
                ax.spines[spine].set_linewidth(2.)
                ax.spines[spine].set_color('lime')
            ax.tick_params(axis='both', which='both', colors='lime')
            ax.yaxis.label.set_color('lime')
            ax.xaxis.label.set_color('lime')
            ax.title.set_color('lime')
            ax.annotate(r'$\rm SNR = %3.2f$' % params[i]['snr'], xy=(min(params[i]['fitx'])+0.05*xran, ymax+0.025*yran), fontsize=18, color='lime')
        else:
            if params[i]['good_fit']:
                ax.annotate(r'$\rm SNR = %3.2f$' % params[i]['snr'], xy=(min(params[i]['fitx'])+0.05*xran, ymax+0.025*yran), fontsize=18)
    plt.tight_layout()
    if star.params['save']:
        path = os.path.join(star.params['path'],filename)
        if not star.params['overwrite']:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300)
    if not star.params['show']:
        plt.close()


def plot_parameters(star, subfilename='background_only.png', filename='global_fit.png', n=0):
    """Plot parameters

    Creates a plot summarizing all derived parameters

    Parameters
        star : pysyd.target.Target
            the main pipeline Target class object
        subfilename : str
            separate filename in the event that only the background is being fit
        filename : str
            the path or extension to save the figure to

    """
    if star.params['background'] and not star.params['globe']:
        n_panels=3
    else:
        n_panels=9
    if not star.lc:
        n_panels -= 1
    x, y = d[n_panels]['x'], d[n_panels]['y']
    params = star.params['plotting']['parameters']

    fig = plt.figure("Global fit for %s"%star.name, figsize=d[n_panels]['size'])
    n += 1
    # PANEL 1: time series data
    if 'time' in params:
        ax1 = fig.add_subplot(x, y, n)
        ax1.plot(params['time'], params['flux'], 'w-')
        ax1.set_xlim([min(params['time']), max(params['time'])])
        ax1.set_title(r'$\rm Time \,\, series$')
        ax1.set_xlabel(r'$\rm Time \,\, [days]$')
        ax1.set_ylabel(r'$\rm Flux$')
    else:
        n -= 1

    n += 1
    # PANEL 2: initial background guesses
    ax2 = fig.add_subplot(x, y, n)
    ax2.plot(params['frequency'], params['random_pow'], c='lightgrey', zorder=0, alpha=0.5)
    ax2.plot(params['frequency'][params['frequency'] < star.params['ps_mask'][0]], params['random_pow'][params['frequency'] < star.params['ps_mask'][0]], 'w-', zorder=1)
    ax2.plot(params['frequency'][params['frequency'] > star.params['ps_mask'][1]], params['random_pow'][params['frequency'] > star.params['ps_mask'][1]], 'w-', zorder=1)
    ax2.plot(params['frequency'][params['frequency'] < star.params['ps_mask'][0]], params['smooth_pow'][params['frequency'] < star.params['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
    ax2.plot(params['frequency'][params['frequency'] > star.params['ps_mask'][1]], params['smooth_pow'][params['frequency'] > star.params['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
    if star.params['background']:
        total = np.zeros_like(params['frequency'])
        for r in range(params['nlaws_orig']):
            model = models.background(params['frequency'], [params['b_orig'][r], params['a_orig'][r]])
            ax2.plot(params['frequency'], model, color='blue', linestyle=':', linewidth=1.5, zorder=4)
            total += model
        total += params['noise']
        ax2.plot(params['frequency'], total, color='blue', linewidth=2., zorder=5)
        ax2.errorbar(params['bin_freq'], params['bin_pow'], yerr=params['bin_err'], color='lime', markersize=0., fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
    ax2.axvline(star.params['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax2.axvline(star.params['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax2.axhline(params['noise'], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5, 5))
    ax2.set_xlim([min(params['frequency']), max(params['frequency'])])
    ax2.set_ylim([np.percentile(params['random_pow'], 1), max(params['random_pow'])*1.25])
    ax2.set_title(r'$\rm Initial \,\, guesses$')
    ax2.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax2.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    n += 1
    # PANEL 3: fitted background
    ax3 = fig.add_subplot(x, y, n)
    ax3.plot(params['frequency'], params['random_pow'], c='lightgrey', zorder=0, alpha=0.5)
    ax3.plot(params['frequency'][params['frequency'] < star.params['ps_mask'][0]], params['random_pow'][params['frequency'] < star.params['ps_mask'][0]], 'w-', zorder=1)
    ax3.plot(params['frequency'][params['frequency'] > star.params['ps_mask'][1]], params['random_pow'][params['frequency'] > star.params['ps_mask'][1]], 'w-', zorder=1)
    ax3.plot(params['frequency'][params['frequency'] < star.params['ps_mask'][0]], params['smooth_pow'][params['frequency'] < star.params['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
    ax3.plot(params['frequency'][params['frequency'] > star.params['ps_mask'][1]], params['smooth_pow'][params['frequency'] > star.params['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
    if star.params['background']:
        total = np.zeros_like(params['frequency'])
        if len(params['pars']) > 1:
            for r in range(len(params['pars'])//2):
                yobs = models.background(params['frequency'], [params['pars'][r*2], params['pars'][r*2+1]])
                ax3.plot(params['frequency'], yobs, color='blue', linestyle=':', linewidth=1.5, zorder=4)
                total += yobs
        if len(params['pars'])%2 == 0:
            total += params['noise']
            ax3.axhline(params['noise'], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
        else:
            total += params['pars'][-1]
            ax3.axhline(params['pars'][-1], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
        ax3.plot(params['frequency'], total, color='blue', linewidth=2., zorder=5)
        ax3.errorbar(params['bin_freq'], params['bin_pow'], yerr=params['bin_err'], color='lime', markersize=0.0, fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
    else:
        ax3.axhline(params['noise'], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
    ax3.axvline(star.params['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax3.axvline(star.params['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    if star.params['globe']:
        mask = np.ma.getmask(np.ma.masked_inside(params['frequency'], star.params['ps_mask'][0], star.params['ps_mask'][1]))
        ax3.plot(params['frequency'][mask], params['pssm'][mask], color='yellow', linewidth=2.0, linestyle='dashed', zorder=6)
    ax3.set_xlim([min(params['frequency']), max(params['frequency'])])
    ax3.set_ylim([np.percentile(params['random_pow'], 1.), max(params['random_pow'])*1.25])
    ax3.set_title(r'$\rm Fitted \,\, model$')
    ax3.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax3.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    if star.params['background'] and not star.params['globe'] and star.params['save']:
        plt.tight_layout()
        path = os.path.join(star.params['path'],subfilename)
        if not star.params['overwrite']:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300)
        if not star.params['show']:
            plt.close()
        return

    n += 1
    # PANEL 4: smoothed power excess 
    ax4 = fig.add_subplot(x, y, n)
    ax4.plot(params['region_freq'], params['region_pow'], 'w-', zorder=3)
    _, f, p = utils._return_max(params['region_freq'], params['region_pow'])
    ax4.axvline([f], color='orange', linestyle='--', linewidth=1.5, zorder=4)
    ax4.scatter([f], [p], s=35.0, edgecolor='orange', marker='X', facecolor='none', linewidths=1.0, zorder=5)
    _, ff, pp = utils._return_max(params['new_freq'], params['numax_fit'])
    ax4.plot(params['new_freq'], params['numax_fit'], 'b-', zorder=0)
    ax4.axvline(ff, color='cyan', linestyle=':', linewidth=1.5, zorder=1)
    ax4.scatter([ff], [pp], s=25.0, edgecolor='cyan', marker='s', facecolor='none', linewidths=1.0, zorder=2)
    ax4.axvline(params['exp_numax'], color='r', linestyle='--', linewidth=1.5, zorder=-1, dashes=(5,5),)
    ax4.set_title(r'$\rm Smoothed \,\, PS$')
    ax4.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax4.set_xlim([min(params['region_freq']), max(params['region_freq'])])

    yrange = max(params['zoom_pow'])-min(params['zoom_pow'])
    peaks_f, peaks_p, _ = utils._max_elements(params['zoom_freq'], params['zoom_pow'], npeaks=10, distance=params['obs_dnu']/4.)
    n += 1
    # PANEL 5: background-corrected power spectrum 
    ax5 = fig.add_subplot(x, y, n)
    ax5.plot(params['zoom_freq'], params['zoom_pow'], 'w-', zorder=0, linewidth=1.0)
    ax5.scatter(peaks_f, peaks_p, s=30.0, edgecolor='orange', marker='o', facecolor='none', linewidths=1.0)
    ax5.set_title(r'$\rm Bg$-$\rm corrected \,\, PS$')
    ax5.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax5.set_ylabel(r'$\rm Power$')
    ax5.set_xlim([star.params['ps_mask'][0], star.params['ps_mask'][1]])
    ax5.set_ylim([min(params['zoom_pow'])-0.025*yrange, max(params['zoom_pow'])+0.1*yrange])

    yrange = max(params['auto'])-min(params['auto'])
    n += 1
    # PANEL 6: ACF 
    ax6 = fig.add_subplot(x, y, n)
    ax6.plot(params['lag'], params['auto'], 'w-', zorder=0, linewidth=1.)
    ax6.scatter(params['peaks_l'], params['peaks_a'], s=35.0, edgecolor='red', marker='^', facecolor='none', linewidths=1.0,)
    ax6.axvline(params['obs_dnu'], color='lime', linestyle='--', linewidth=1.5, zorder=2)
    ax6.scatter(params['best_lag'], params['best_auto'], s=45.0, edgecolor='lime', marker='s', facecolor='none', linewidths=1.0)
    ax6.plot(params['zoom_lag'], params['zoom_auto'], 'r-', zorder=5, linewidth=1.0)
    ax6.plot(params['lag'], params['weights'], c='yellow', linestyle=':', zorder=-1, linewidth=0.75)
    ax6.set_title(r'$\rm Autocorrelation \,\, function$')
    ax6.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
    ax6.set_xlim([min(params['lag']), max(params['lag'])])
    ax6.set_ylim([min(params['auto'])-0.05*yrange, max(params['auto'])+0.1*yrange])

    n += 1
    # PANEL 7: dnu fit
    ax7 = fig.add_subplot(x, y, n)
    ax7.plot(params['zoom_lag'], params['zoom_auto'], 'w-', zorder=0, linewidth=1.0)
    ax7.plot(params['new_lag'], params['dnu_fit'], color='lime', linewidth=1.5)
    ax7.set_title(r'$\rm \Delta\nu \,\, fit$')
    ax7.set_xlabel(r'$\rm Frequency \,\, separation \,\, [\mu Hz]$')
    ax7.annotate(r'$\Delta\nu = %.2f$'%params['obs_dnu'], xy=(0.025, 0.85), xycoords="axes fraction", fontsize=18, color='lime')
    ax7.set_xlim([min(params['zoom_lag']), max(params['zoom_lag'])])

    if star.params['interp_ech']:
        interpolation='bilinear'
    else:
        interpolation='nearest'
    n += 1
    # PANEL 8: echelle diagram
    use_dnu = params['use_dnu']
    ax8 = fig.add_subplot(x, y, n)
    ax8.imshow(params['ed'], extent=params['extent'], interpolation=interpolation, aspect='auto', origin='lower', cmap=plt.get_cmap(star.params['cmap']))
    ax8.axvline([use_dnu], color='white', linestyle='--', linewidth=1.5, dashes=(5, 5))
    ax8.set_title(r'$\rm \grave{E}chelle \,\, diagram$')
    ax8.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$' % use_dnu)
    ax8.set_ylabel(r'$\rm \nu \,\, [\mu Hz]$')
    ax8.set_xlim([params['extent'][0], params['extent'][1]])
    ax8.set_ylim([params['extent'][2], params['extent'][3]])

    yrange = max(params['y'])-min(params['y'])
    n += 1
    # PANEL 9: collapsed ED
    ax9 = fig.add_subplot(x, y, n)
    ax9.plot(params['x'], params['y'], color='white', linestyle='-', linewidth=0.75)
    ax9.set_title(r'$\rm Collapsed \,\, ED$')
    ax9.set_xlabel(r'$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$' % use_dnu)
    ax9.set_ylabel(r'$\rm Collapsed \,\, power$')
    ax9.set_xlim([0.0, 2.0*use_dnu])
    ax9.set_ylim([min(params['y'])-0.025*(yrange), max(params['y'])+0.05*(yrange)])

    plt.tight_layout()
    if star.params['save']:
        path = os.path.join(star.params['path'],filename)
        if not star.params['overwrite']:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300)
    if not star.params['show']:
        plt.close()


def plot_samples(star, filename='samples.png'):
    """
    Plot results of the Monte-Carlo sampling

    Parameters
        star : target.Target
            the pySYD pipeline object
        filename : str
            the path or extension to save the figure to
    
    """
    x, y = d[len(star.df.columns.values.tolist())]['x'], d[len(star.df.columns.values.tolist())]['y']
    params = utils.get_dict()
    sample = star.params['plotting']['samples']

    fig = plt.figure("Posteriors for %s"%star.name, figsize=d[len(star.df.columns.values.tolist())]['size'])
    for i, col in enumerate(sample['df'].columns.values.tolist()):
        ax = plt.subplot(x, y, i+1)
        ax.hist(sample['df'][col], bins=20, color='cyan', histtype='step', lw=2.5, facecolor='0.75')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_title(params[col]['label'], fontsize=16)
    plt.tight_layout()
    if star.params['save']:
        path = os.path.join(star.params['path'],filename)
        if not star.params['overwrite']:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300)
    if not star.params['show']:
        plt.close()


def plot_bgfits(star, filename='bgmodel_fits.png', highlight=True):
    """
    Comparison of the background model fits 

    Parameters
        star : target.Target
            the pySYD pipeline object
        filename : str
            the path or extension to save the figure to
        highlight : bool, optional
            if `True`, highlights the selected model
    
    """
    params = star.params['plotting']['parameters']
    x, y = d[len(params['models'])+1]['x'], d[len(params['models'])+1]['y']

    fig = plt.figure("Background model comparison for %s"%star.name, figsize=d[len(params['models'])+1]['size'])
    # Initial background guesses
    ax1 = fig.add_subplot(x, y, 1)
    ax1.plot(params['frequency'], params['random_pow'], c='lightgrey', zorder=0, alpha=0.5)
    ax1.plot(params['frequency'][params['frequency'] < star.params['ps_mask'][0]], params['random_pow'][params['frequency'] < star.params['ps_mask'][0]], 'w-', zorder=1)
    ax1.plot(params['frequency'][params['frequency'] > star.params['ps_mask'][1]], params['random_pow'][params['frequency'] > star.params['ps_mask'][1]], 'w-', zorder=1)
    ax1.plot(params['frequency'][params['frequency'] < star.params['ps_mask'][0]], params['smooth_pow'][params['frequency'] < star.params['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
    ax1.plot(params['frequency'][params['frequency'] > star.params['ps_mask'][1]], params['smooth_pow'][params['frequency'] > star.params['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
    if star.params['background']:
        total = np.zeros_like(params['frequency'])
        for r in range(params['nlaws_orig']):
            model = models.background(params['frequency'], [params['b_orig'][r], params['a_orig'][r]])
            ax1.plot(params['frequency'], model, color='blue', linestyle=':', linewidth=1.5, zorder=4)
            total += model
        total += params['noise']
        ax1.plot(params['frequency'], total, color='blue', linewidth=2., zorder=5)
        ax1.errorbar(params['bin_freq'], params['bin_pow'], yerr=params['bin_err'], color='lime', markersize=0., fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
    ax1.axvline(star.params['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax1.axvline(star.params['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
    ax1.axhline(params['noise'], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5, 5))
    ax1.set_xlim([min(params['frequency']), max(params['frequency'])])
    ax1.set_ylim([min(params['random_pow']), max(params['random_pow'])*1.25])
    ax1.set_title(r'$\rm Initial \,\, guesses$')
    ax1.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
    ax1.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot background models
    for n, mm in enumerate(params['models']):
        ax = fig.add_subplot(x, y, n+2)
        ax.plot(params['frequency'], params['random_pow'], c='lightgrey', zorder=0, alpha=0.5)
        ax.plot(params['frequency'][params['frequency'] < star.params['ps_mask'][0]], params['random_pow'][params['frequency'] < star.params['ps_mask'][0]], 'w-', zorder=1)
        ax.plot(params['frequency'][params['frequency'] > star.params['ps_mask'][1]], params['random_pow'][params['frequency'] > star.params['ps_mask'][1]], 'w-', zorder=1)
        ax.plot(params['frequency'][params['frequency'] < star.params['ps_mask'][0]], params['smooth_pow'][params['frequency'] < star.params['ps_mask'][0]], 'r-', linewidth=0.75, zorder=2)
        ax.plot(params['frequency'][params['frequency'] > star.params['ps_mask'][1]], params['smooth_pow'][params['frequency'] > star.params['ps_mask'][1]], 'r-', linewidth=0.75, zorder=2)
        pars = params['paras'][n]
        total = np.zeros_like(params['frequency'])
        if len(pars) > 1:
            for r in range(len(pars)//2):
                yobs = models.background(params['frequency'], [pars[r*2], pars[r*2+1]])
                ax.plot(params['frequency'], yobs, color='blue', linestyle=':', linewidth=1.5, zorder=4)
                total += yobs
        if len(pars)%2 == 0:
            total += params['noise']
            ax.axhline(params['noise'], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
        else:
            total += pars[-1]
            ax.axhline(pars[-1], color='blue', linestyle='dashed', linewidth=1.5, zorder=3, dashes=(5,5))
        ax.plot(params['frequency'], total, color='blue', linewidth=2., zorder=5)
        ax.errorbar(params['bin_freq'], params['bin_pow'], yerr=params['bin_err'], color='lime', markersize=0.0, fillstyle='none', ls='None', marker='D', capsize=3, ecolor='lime', elinewidth=1, capthick=2, zorder=3)
        ax.axvline(star.params['ps_mask'][0], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
        ax.axvline(star.params['ps_mask'][1], color='darkorange', linestyle='dashed', linewidth=2.0, zorder=1, dashes=(5,5))
        if star.params['globe']:
            mask = np.ma.getmask(np.ma.masked_inside(params['frequency'], star.params['ps_mask'][0], star.params['ps_mask'][1]))
            ax.plot(params['frequency'][mask], params['pssm'][mask], color='yellow', linewidth=2.0, linestyle='dashed', zorder=6)
        ax.set_xlim([min(params['frequency']), max(params['frequency'])])
        ax.set_ylim([min(params['random_pow']), max(params['random_pow'])*1.25])
        if mm%2 == 0:
            wn='fixed'
        else:
            wn='free'
#        ax.set_title(r'$\rm nlaws=%s \,\, | \,\, wn=%s \,\, | \,\, AIC = %.2f \,\, | \,\, BIC = %.2f$'%(str(int(mm//2)),wn,params['aic'][n],params['bic'][n]))
        ax.set_title(r'$\rm nlaws=%s \,\, | \,\, wn=%s $'%(str(int(mm//2)),wn))
        ax.set_xlabel(r'$\rm Frequency \,\, [\mu Hz]$')
        ax.set_ylabel(r'$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Highlight selected model
        if highlight and mm == params['model']:
            ax.spines['bottom'].set_color('lime')
            ax.spines['top'].set_color('lime') 
            ax.spines['right'].set_color('lime')
            ax.spines['left'].set_color('lime')
            ax.tick_params(axis='both', which='both', colors='lime')
            ax.yaxis.label.set_color('lime')
            ax.xaxis.label.set_color('lime')
            ax.title.set_color('lime')

    plt.tight_layout()
    if star.params['save']:
        path = os.path.join(star.params['path'],filename)
        if not star.params['overwrite']:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300)
    if not star.params['show']:
        plt.close()


def create_benchmark_plot(filename='comparison.png', variables=['numax','dnu'], show=False, save=True, overwrite=False, npanels=2,):
    """
    Compare ensemble results between the ``pySYD`` and ``SYD`` pipelines
    for the *Kepler* legacy sample
    
    """
    # optional modules
    from astropy.stats import mad_std
    import matplotlib.gridspec as gridspec
    # load data and formatting for plots
    params = utils.get_dict('params')
    df = utils.get_results()

    fig = plt.figure(figsize=(6,12), facecolor='white', edgecolor='white')
    og = gridspec.GridSpec(npanels, 1, wspace=0.0, hspace=0.0, bottom=0.1)
    for i, variable in enumerate(variables):
        # inner grid for including residuals
        ig = gridspec.GridSpecFromSubplotSpec(8, 6, subplot_spec=og[i])
        # plot parameters
        ax1 = plt.Subplot(fig, ig[0:6, 0:6])
        ax1.plot(df[params['c%s'%variable]['syd']].values, df[params['c%s'%variable]['syd']].values, c='k',ls='--')
        ax1.errorbar(df[params['c%s'%variable]['syd']].values, df[params['c%s'%variable]['pysyd']].values, fmt='o', mec='k', mfc='none', color='lightgrey')
        ax1.set_ylabel(params['c%s'%variable]['ylabel'])
        ax1.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
        ax1.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')
        ax1.tick_params(labelsize=22)

        # plot residuals
        residuals = (df[params['c%s'%variable]['syd']].values-df[params['c%s'%variable]['pysyd']].values)/df[params['c%s'%variable]['syd']].values
        ax2 = plt.Subplot(fig, ig[6:8, 0:6])
        ax2.errorbar(df[params['c%s'%variable]['syd']].values, residuals, fmt='o' ,mec='k', mfc='none', color='lightgrey')
        ax2.set_ylabel(r'$\frac{\textrm{SYD -- pySYD}}{\textrm{SYD}}$')
        ax2.set_xlabel(params['c%s'%variable]['xlabel']) 
        ax2.set_ylim(params['c%s'%variable]['ylimit'][0],params['c%s'%variable]['ylimit'][1])
        ax2.axhline(0.01, c='lightgrey', ls='--', dashes=(5,5), zorder=-1)
        ax2.axhline(0, c='k', ls='--', dashes=(5,5), zorder=-1)
        ax2.axhline(-0.01, c='lightgrey', ls='--', dashes=(5,5), zorder=-1)
        ax2.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
        ax2.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')
        ax2.tick_params(labelsize=22)
        # add stats
        STR1='Offset: {0:.5f} +/- {0:.5f}'.format(np.median(residuals), 1.25*np.std(residuals)/np.sqrt(len(residuals)))
        STR2='Scatter (MAD): {0:.5f}'.format(mad_std(residuals))
        STR3='Scatter (SD): {0:.5f}'.format(np.std(residuals))
        t = ax1.text(0.03,0.9, s=STR1+STR2+STR3, color='k', ha='left', va='center', transform=ax1.transAxes)
        t.set_bbox(dict(facecolor='white',edgecolor='k'))

    plt.tight_layout()
    if save:
        path = os.path.join(os.path.abspath(os.getcwd()), filename)
        if not overwrite:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300, facecolor='white', edgecolor='white')
    if show:
        plt.show()
    plt.close()


def check_data(star, args, show=True):
    """Plot input data for a target

    """
    if star.params['verbose']:
        print(' - displaying figures')
    if star.lc:
        plot_light_curve(star, args)
    if star.ps:
        plot_power_spectrum(star, args)
    if star.params['show']:
        plt.show(block=False)
        input(' - press any key to exit')
        print('\n\n')
    plt.close()


def plot_light_curve(star, args, filename='time_series.png', npanels=1):
    """Plot light curve data

    Parameters
        star : target.Target
            the pySYD pipeline object
        filename : str
            the path or extension to save the figure to
        npanels : int
            number of panels in this figure (default=`1`)

    
    """
    x, y = d[npanels]['x'], d[npanels]['y']

    fig = plt.figure("%s time series"%star.name, figsize=d[npanels]['size'])
    ax = plt.subplot(x,y,1)
    ax.plot(star.time, star.flux, 'w-')
    if args.lower_ts is not None:
        lower = args.lower_ts
    else:
        lower = min(star.time)
    if args.upper_ts is not None:
        upper = args.upper_ts
    else:
        upper = max(star.time)
    ax.set_xlim([lower, upper])
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')  
    ax.tick_params(labelsize=22)
    plt.xlabel(r'$\rm Time \,\, [days]$', fontsize=28)
    plt.ylabel(r'$\rm Normalized \,\, flux$', fontsize=28)

    plt.tight_layout()
    if star.params['save']:
        path = os.path.join(star.params['path'],filename)
        if not star.params['overwrite']:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300)
    if not star.params['show']:
        plt.close()


def plot_power_spectrum(star, args, filename='power_spectrum.png', npanels=1):
    """Plot power spectrum

    Parameters
        star : target.Target
            the pySYD pipeline object
        filename : str
            the path or extension to save the figure to
        npanels : int
            number of panels in this figure (default=`1`)

    
    """
    x, y = d[npanels]['x'], d[npanels]['y']

    fig = plt.figure("%s power spectrum"%star.name, figsize=d[npanels]['size'])
    ax = plt.subplot(1,1,1)
    ax.plot(star.frequency, star.power, 'w-')
    if args.lower_ps is not None:
        lower = args.lower_ps
    else:
        lower = min(star.frequency)
    if args.upper_ps is not None:
        upper = args.upper_ps
    else:
        upper = max(star.frequency)
    ax.set_xlim([lower, upper])
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')  
    ax.tick_params(labelsize=22)
    if args.log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    plt.xlabel(r'$\rm Frequency \,\, [\mu Hz]$', fontsize=28)
    plt.ylabel(r'$\rm Power \,\, [ppm^2 \, \mu Hz^{-1}]$', fontsize=28)

    plt.tight_layout()
    if star.params['save']:
        path = os.path.join(star.params['path'],filename)
        if not star.params['overwrite']:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300)
    if not star.params['show']:
        plt.close()


def plot_1d_ed(star, filename='1d_ed.png', npanels=1):
    """Plot collapsed ED

    Parameters
        star : target.Target
            the pySYD pipeline object
        filename : str
            the path or extension to save the figure to
        npanels : int
            number of panels in this figure (default=`1`)

    
    """
    x, y = d[npanels]['x'], d[npanels]['y']

    fig = plt.figure("%s 1d ED"%star.name, figsize=d[npanels]['size'])
    ax = plt.subplot(x,y,1)
    ax.plot(star.x, star.y, 'w-')
    ax.set_xlim([min(star.x), max(star.x)])
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')  
    ax.tick_params(labelsize=22)
    plt.xlabel(r'$\rm Folded \,\, Frequency \,\, [\mu Hz]$', fontsize=28)
    plt.ylabel(r'$\rm Collapsed \,\, ED \,\, [power]$', fontsize=28)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()
    if star.params['save']:
        path = os.path.join(star.params['path'],filename)
        if not star.params['overwrite']:
            path = utils._get_next(path)
        plt.savefig(path, dpi=300)
    if not star.params['show']:
        plt.close()


def get_colors(n, cmap='cividis', min_value=0.0, max_value=1.0):
    color_by = np.copy(n)
    new_cmap = truncate_colormap(plt.get_cmap(cmap), minval=min_value, maxval=max_value)
    colors = [new_cmap(i) for i in np.linspace(0, 1, color_by.shape[0])]
    return colors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    import matplotlib.colors as colors
    if n == -1:
        n = cmap.N
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

