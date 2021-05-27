import os
import glob
import numpy as np
import pandas as pd
from itertools import chain
from astropy.io import ascii
import multiprocessing as mp
from astropy.stats import mad_std
from astropy.timeseries import LombScargle as lomb

from pysyd.plots import set_plot_params
from pysyd.functions import *
from pysyd.models import *



def get_info(args, parallel=False, stars=None, verbose=False, show=False, save=True, G=6.67428e-8, 
             teff_sun=5777.0, mass_sun=1.9891e33, radius_sun=6.95508e10, dnu_sun=135.1, 
             numax_sun=3090.0, width_sun=1300.0, tau_sun=[5.2e6,1.8e5,1.7e4,2.5e3,280.0,80.0], 
             tau_sun_single=[3.8e6, 2.5e5, 1.5e5, 1.0e5, 230., 70.], kepcorr=False, groups=[],
             oversample=True, of_actual=0, of_new=5):
    """
    Loads todo.txt, sets up file paths, loads in any available star information,
    and sets up matplotlib params.

    Parameters
    ----------
    args : argparse.Namespace
        command line arguments
    star_info : str
        the file path to the star_info.csv file containing star information. Default value is `'info/star_info.csv'`
    params : dict
        the pipeline parameters, which is saved to args.params

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    """

    params={}
    # Open star list
    if args.stars is None or args.stars == []:
        with open(args.todo, "r") as f:
            args.stars = [int(float(line.strip().split()[0])) for line in f.readlines()]
    check_inputs(args)
    params['inpdir'] = args.inpdir
    params['outdir'] = args.outdir
    if parallel:
        todo = np.array(args.stars)
        args.verbose = False
        args.show = False
        if args.n_threads == 0:
            args.n_threads = mp.cpu_count()
        if len(todo) < args.n_threads:
            args.n_threads = len(todo)
        # divide stars into groups set by number of cpus/nthreads available
        digitized = np.digitize(np.arange(len(todo))%args.n_threads,np.arange(args.n_threads))
        groups = np.array([todo[digitized == i] for i in range(1, args.n_threads+1)], dtype=object)
    else:
        groups = np.array([])

    # Adding constants and the star list
    params.update({
        'numax_sun':3090.0, 'width_sun':1300.0, 'stars':args.stars, 'G':6.67428e-8, 
        'show':args.show, 'oversample':args.oversample, 'tau_sun':[5.2e6,1.8e5,1.7e4,2.5e3,280.0,80.0], 
        'tau_sun_single': [3.8e6, 2.5e5, 1.5e5, 1.0e5, 230., 70.], 'radius_sun': 6.95508e10, 'save':args.save, 
        'teff_sun':5777.0, 'mass_sun':1.9891e33, 'dnu_sun':135.1, 'kepcorr': args.kepcorr, 'groups':groups, 
        'of_actual':args.of_actual, 'of_new':args.of_new,
    })
    # Set file paths
    for star in args.stars:
        params[star] = {}
        params[star]['path'] = '%s/%d/' % (args.outdir,star)
    args.params = params

    # Initialise parameters for the find excess routine
    args = get_excess_params(args)
    # Initialise parameters for the fit background routine
    args = get_background_params(args)
    # Get star info
    args = get_star_info(args)
    set_plot_params()

    return args


def check_inputs(args):
    """ 
    Make sure the command line inputs are the proper lengths and/or types

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments

    """

    checks={'lower_b':args.lower_b,'upper_b':args.upper_b,'lower_x':args.lower_x,
            'upper_x':args.upper_x,'dnu':args.dnu,'numax':args.numax}
    for check in checks:
        if checks[check] is not None:
            assert len(args.stars) == len(checks[check]), "The number of values provided for %s does not equal the number of stars"%check
    if args.of_actual is not None:
        assert isinstance(args.of_actual,int), "The (actual) oversampling factor provided must be an integer"
    if args.of_new is not None:
        assert isinstance(args.of_new,int), "The (new) oversampling factor provided must be an integer"


def get_excess_params(args, n_trials=3, step=0.25, binning=0.005, smooth_width=1.5):
    """
    Get the parameters for the find excess routine.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    findex : dict
        the parameters of the find excess routine
    step : float
        TODO: Write description. Default value is `0.25`.
    binning : float
        logarithmic binning width. Default value is `0.005`.
    n_trials : int
        the number of trials. Default value is `3`.
    lower : float
        the lower frequency bound. Default value is `10.0`.
    upper : float
        the upper frequency bound. Default value is `4000.0`.

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments

    """

    findex = {
        'step': args.step,
        'binning': args.binning,
        'smooth_width': args.smooth_width,
        'n_trials': args.n_trials,
    }

    # Initialise save folders
    if args.save:
        for star in args.params['stars']:
            if not os.path.exists(args.params[star]['path']):
                os.makedirs(args.params[star]['path'])
    args.findex = findex
    args.findex['results'] = {}

    return args


def get_background_params(args, box_filter=2.5, mc_iter=1, ind_width=50, n_rms=20, n_peaks=5, 
                          smooth_ps=1.0, slope=False, samples=False, clip_ech=True, clip_value=None, 
                          smooth_ech=None, interp_ech=False, convert=True, drop=True):
    """
    Get the parameters for the background-fitting routine.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments

    Uses
    ----
    fitbg : dict
        the parameters relevant for the fit background routine, which is saved to args.fitbg
    box_filter : args.box_filter
        the size of the 1D box smoothing filter (in $\mu$Hz). Default value is `2.5`.
    ind_width : int
        the independent average smoothing width. Default value is `50`.
    n_rms : int
        number of data points to estimate red noise contributions. Default value is `20`.
    n_peaks : int
        the number of peaks to select. Default value is `10`.
    force : float
        if not false (i.e. non-zero) will force dnu to be the equal to this value. 
    clip : bool
        if true will set the minimum frequency value of the echelle plot to `clip_value`. Default value is `True`.
    clip_value : float
        the minimum frequency of the echelle plot. Default value is `0.0`.
    smooth_ech : float
        option to smooth the output of the echelle plot
    smooth_ps : float
        TODO: Write description. Default value is `1.0`.
    slope : bool
        if true will correct for edge effects and residual slope in Gaussian fit. Default value is `False`.
    samples : bool
        if true, will save the monte carlo samples to a csv. Default value is `True`.
    convert : args.convert
        converts to Harvey parametrization {a_n,b_n} -> {tau_n,sigma_n}. Default value is `True`.
    drop : args.drop
        drops the extra columns after converting the samples. Default value is `True`.

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    """

    fitbg = {
        'box_filter': args.box_filter,
        'mc_iter': args.mc_iter,
        'ind_width': args.ind_width,
        'n_rms': args.n_rms,
        'n_peaks': args.n_peaks,
        'smooth_ps': args.smooth_ps,
        'slope': args.slope,
        'samples': args.samples,
        'clip_ech': args.clip_ech,
        'clip_value': args.clip_value,
        'smooth_ech': args.smooth_ech,
        'interp_ech': args.interp_ech,
        'convert': args.convert,
        'drop': args.drop,
    }

    # Harvey components
    fitbg['functions'] = {1: harvey_one, 2: harvey_two, 3: harvey_three, 4: harvey_four, 5: harvey_five, 6: harvey_six}

    # Initialise save folders
    if args.save:
        for star in args.params['stars']:
            if not os.path.exists(args.params[star]['path']):
                os.makedirs(args.params[star]['path'])
    args.fitbg = fitbg
    args.fitbg['results'] = {}
    args.fitbg['acf_mask'] = {}

    return args


def get_star_info(args):
    """
    Get star information stored in args.info. Please note: this is not required for pySYD to run
    successfully. Default value is `info/star_info.csv`.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    cols : list
        the list of columns to provide stellar information for

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    """

    columns = get_data_columns(type='required')
    # Open file if it exists
    if os.path.exists(args.info):
        df = pd.read_csv(args.info)
        stars = df.stars.values.tolist()
        for i, star in enumerate(args.params['stars']):
            args.params[star]['excess'] = args.excess
            args.params[star]['background'] = args.background
            args.params[star]['force'] = False
            if star in stars:
                idx = stars.index(star)
                # Update information from columns
                for column in columns:
                    if not np.isnan(float(df.loc[idx,column])):
                        args.params[star][column] = float(df.loc[idx, column])
                    else:
                        args.params[star][column] = None
                # Add estimate of numax if the column exists
                if args.params[star]['numax'] is not None:
                    args.params[star]['dnu'] = 0.22*(args.params[star]['numax']**0.797)
                # Otherwise estimate using other stellar parameters
                else:
                    if args.params[star]['radius'] is not None and args.params[star]['logg'] is not None:
                        args.params[star]['mass'] = ((((args.params[star]['radius']*args.params['radius_sun'])**(2.0))*10**(args.params[star]['logg'])/args.params['G'])/args.params['mass_sun'])
                        args.params[star]['numax'] = args.params['numax_sun']*args.params[star]['mass']*(args.params[star]['radius']**(-2.0))*((args.params[star]['teff']/args.params['teff_sun'])**(-0.5))
                        args.params[star]['dnu'] = args.params['dnu_sun']*(args.params[star]['mass']**(0.5))*(args.params[star]['radius']**(-1.5))
            override={'lower_b':args.lower_b,'upper_b':args.upper_b,'lower_x':args.lower_x,
                      'upper_x':args.upper_x,'dnu':args.dnu,'numax':args.numax}
            for each in override:
                if override[each] is not None:
                    # if numax is provided via CLI, findex is skipped
                    if each == 'numax':
                        args.params[star]['excess'] = False
                        args.params[star]['numax'] = override[each][i]
                        args.params[star]['dnu'] = 0.22*(args.params[star]['numax']**0.797)
                    # if dnu is provided via CLI, this value is used instead of the derived dnu
                    elif each == 'dnu':
                        args.params[star]['force'] = True
                        args.params[star]['guess'] = override[each][i]
                    else:
                        args.params[star][each] = override[each][i]
                
    return args


def load_data(star, args):
    """
    Loads both the light curve and power spectrum data in for a given star,
    which will return `False` if unsuccessful and therefore, not run the rest
    of the pipeline.

    Parameters
    ----------
    star : target.Target
        the pySYD pipeline object

    Returns
    -------
    star : target.Target
        the pySYD pipeline object
    lc_data : bool
        will return `True` if the light curve data was loaded in properly otherwise `False`
    ps_data : bool
        will return `True` if the power spectrum file was successfully loaded otherwise `False`
    """

    star.lc = False
    star.ps = False
    # Now done at beginning to make sure it only does this one per star
    if glob.glob('%s/%d_*'%(args.inpdir,star.name)) != []:
        if star.verbose:
            print('\n\n-------------------------------------------------')
            print('Target: %d'%star.name)
            print('-------------------------------------------------')
        # Load light curve
        if not os.path.exists('%s/%d_LC.txt'%(args.inpdir, star.name)):
            if args.cadence != 0 and args.nyquist is not None:
                star.cadence = args.cadence
                star.nyquist = args.nyquist
                nyquist = 10**6./(2.0*args.cadence)
                if int(args.nyquist) != int(nyquist):
                    if self.verbose:
                        print('# WARNING: LC CADENCE AND PS NYQUIST ARE INCONSISTENT')
            elif args.cadence != 0 and args.nyquist is None:
                star.cadence = args.cadence
                star.nyquist = 10**6./(2.0*args.cadence)
            elif args.cadence == 0 and args.nyquist is not None:
                star.cadence = 10**6./(2.0*args.nyquist)
                star.nyquist = args.nyquist
            else:
                if star.verbose:
                    print('# WARNING: NO TIME SERIES DATA PROVIDED')
                    print('#          Please specify either:')
                    print('#          1) the time series cadence or')
                    print('#          2) the nyquist frequency of the power spectrum')
                    print('#          for pySYD to run properly.')
        else:
            star.lc = True
            star.time, star.flux = get_file('%s/%d_LC.txt'%(args.inpdir, star.name))
            star.cadence = int(np.nanmedian(np.diff(star.time)*24.0*60.0*60.0))
            star.nyquist = 10**6./(2.0*star.cadence)
            if star.verbose:
                print('# LIGHT CURVE: %d lines of data read'%len(star.time))
                print('# Time series cadence: %d seconds'%star.cadence)

        # Load power spectrum
        if not os.path.exists('%s/%d_PS.txt'%(args.inpdir, star.name)):
            if star.verbose:
                print('# ERROR: %s/%d_PS.txt not found'%(args.inpdir, star.name))
        else:
            star.ps = True
            star.frequency, star.power = get_file('%s/%d_PS.txt'%(args.inpdir, star.name))
            if star.lc:
                of_actual = int(round((1./((max(star.time)-min(star.time))*0.0864))/(star.frequency[1]-star.frequency[0])))
                if args.of_actual == 0:
                    args.of_actual = of_actual
                    star.params['of_actual'] = of_actual
                if args.of_actual != of_actual:
                    if star.verbose:
                        print('# WARNING: THE OVERSAMPLING FACTORS ARE INCONSISTENT')
                        print('#       -> i.e. the oversampling factor provided via CLI (i.e. -ofa) ')
                        print('#          was %d but an oversampling factor of %d was calculated '%(args.of_actual,of_actual))
                        print('#          from the time series data. This will likely cause fatal errors!')
                        print('#          ** 12/10 recommend fixing this first **')
                        print('#          -ofa == %d'%of_actual)
            if star.verbose:
                print('# POWER SPECTRUM: %d lines of data read'%len(star.frequency))
                if args.of_actual == 1:
                    print('# Critically sampled')
                else:
                    print('# Oversampled by a factor of %d'%args.of_actual)
            if args.oversample:
                star.freq_cs = np.array(star.frequency[args.of_actual-1::args.of_actual])
                star.pow_cs = np.array(star.power[args.of_actual-1::args.of_actual])
                if args.of_actual != args.of_new:
                    if not star.lc:
                        if star.verbose:
                            print('-------------------------------------------------')
                            print('#   ERROR: need time series data to compute an  #')
                            print('#  oversampled power spectrum... exiting script #')
                            print('-------------------------------------------------')
                        star.ps = False
                        return star
                    else:
                        star.freq_os, star.pow_os = lomb(star.time, star.flux).autopower(method='fast', samples_per_peak=args.of_new, maximum_frequency=star.nyquist)
                else:
                    star.freq_os, star.pow_os = np.copy(star.frequency), np.copy(star.power)
            else:
                if args.of_actual != 1:
                    star.freq_cs = np.array(star.frequency[args.of_actual-1::args.of_actual])
                    star.pow_cs = np.array(star.power[args.of_actual-1::args.of_actual])
                else:
                    star.freq_cs, star.pow_cs = np.copy(star.frequency), np.copy(star.power)
                star.freq_os, star.pow_os = np.copy(star.freq_cs), np.copy(star.pow_cs)

            star.resolution = star.freq_cs[1]-star.freq_cs[0]
            if star.verbose:
                print('# Power spectrum resolution: %.6f muHz'%star.resolution)
            if args.kepcorr:
                if star.verbose:
                    print(' - using Kepler artefact correction - ')
                star = remove_artefact(star)

            # If running the first module, mask out any unwanted frequency regions
            star.frequency, star.power = np.copy(star.freq_cs), np.copy(star.pow_cs)
            if star.params[star.name]['excess']:
                # Make a mask using the given frequency bounds for the find excess routine
                mask = np.ones_like(star.frequency, dtype=bool)
                if star.params[star.name]['lower_x'] is not None:
                    mask *= np.ma.getmask(np.ma.masked_greater_equal(star.frequency, star.params[star.name]['lower_x']))
                else:
                    mask *= np.ma.getmask(np.ma.masked_greater_equal(star.frequency, min(star.freq_cs)))
                if star.params[star.name]['upper_x'] is not None:
                    mask *= np.ma.getmask(np.ma.masked_less_equal(star.frequency, star.params[star.name]['upper_x']))
                else:
                    mask *= np.ma.getmask(np.ma.masked_less_equal(star.frequency, star.nyquist))
                star.params[star.name]['ex_mask'] = mask
                star.frequency = star.frequency[mask]
                star.power = star.power[mask]
                if star.params[star.name]['numax'] is not None and star.params[star.name]['numax'] <= 500.:
                    star.boxes = np.logspace(np.log10(0.5), np.log10(25.), star.findex['n_trials'])*1.
                else:
                    star.boxes = np.logspace(np.log10(50.), np.log10(500.), star.findex['n_trials'])*1.
    return star


def get_file(path):
    """Load either a light curve or a power spectrum data file and saves the data into `x` and `y`.

    Parameters
    ----------
    path : str
        the file path of the data file
    """
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    # Set values
    x = np.array([float(line.strip().split()[0]) for line in lines])
    y = np.array([float(line.strip().split()[1]) for line in lines])
    return x, y


def check_fitbg(star):
    """Check if there is prior knowledge about numax as SYD needs this information to work well
    (either from findex module or from star info csv).

    Returns
    -------
    result : bool
        will return `True` if there is prior value for numax otherwise `False`.
    """
    if star.params[star.name]['excess']:
        # Check whether output from findex module exists; 
        # if yes, let that override star info guesses
        if glob.glob('%sexcess.csv' % star.params[star.name]['path']) != []:
            df = pd.read_csv('%sexcess.csv' % star.params[star.name]['path'])
            for col in ['numax', 'dnu', 'snr']:
                star.params[star.name][col] = df.loc[0, col]
        # Break if no numax is provided in any scenario
        if star.params[star.name]['numax'] is None:
            print(
                '# ERROR: pySYD cannot run without any value for numax'
            )
            return False
        else:
            return True
    else:
        if star.verbose:
            print(
                '# WARNING: you are not running findex. \n# A value of %.2f muHz was provided for numax.'%star.params[star.name]['numax']
            )
        return True


def get_initial_guesses(star):
    """
    Get initial guesses for the granulation background.

    """
    from pysyd.functions import mean_smooth_ind

    if star.params[star.name]['lower_b'] is not None:
        lower = star.params[star.name]['lower_b']
    else:
        lower = min(star.frequency)
    if star.params[star.name]['upper_b'] is not None:
        upper = star.params[star.name]['upper_b']
    else:
        if star.nyquist is not None:
            upper = star.nyquist
        else:
            upper = 5000.
    star.params[star.name]['fb_mask']=[lower,upper]
    star.frequency, star.power = np.copy(star.freq_os), np.copy(star.pow_os)
    star.resolution = (star.freq_cs[1]-star.freq_cs[0])/star.params['of_actual']

    # Mask power spectrum for fitbg module
    mask = np.ma.getmask(np.ma.masked_inside(star.frequency, star.params[star.name]['fb_mask'][0], star.params[star.name]['fb_mask'][1]))
    star.frequency, star.random_pow = np.copy(star.frequency[mask]), np.copy(star.power[mask])
    
    # Use scaling relations from sun to get starting points
    star = solar_scaling(star)

    # if lower numax adjust default smoothing filter from 2.5->0.5muHz
    # this needs to be fixed - it doesn't change for the rest
    if star.params[star.name]['numax'] <= 500.:
        star.fitbg['smooth_ps'] = 0.5
    else:
        star.fitbg['smooth_ps'] = 2.5

    # Bin power spectrum to model stellar background/correlated red noise components
    bin_freq, bin_pow, bin_err = mean_smooth_ind(star.frequency, star.random_pow, star.fitbg['ind_width'])
    # Mask out region with power excess
    star.bin_freq = bin_freq[~((bin_freq > star.maxpower[0])&(bin_freq < star.maxpower[1]))]
    star.bin_pow = bin_pow[~((bin_freq > star.maxpower[0])&(bin_freq < star.maxpower[1]))]
    star.bin_err = bin_err[~((bin_freq > star.maxpower[0])&(bin_freq < star.maxpower[1]))]

    return star


def solar_scaling(star, scaling='tau_sun_single'):
    """
    Uses scaling relations from the Sun to:
        1) estimate the width of the region of oscillations using numax
        2) guess starting values for granulation timescales

    """

    # Use scaling relations to estimate width of oscillation region to mask out of the background fit
    star.width = star.params['width_sun']*(star.params[star.name]['numax']/star.params['numax_sun'])
    star.times = star.width/star.params[star.name]['dnu']
    star.maxpower = [star.params[star.name]['numax'] - star.times*star.params[star.name]['dnu'],
                     star.params[star.name]['numax'] + star.times*star.params[star.name]['dnu']]

    # Use scaling relation for granulation timescales from the sun to get starting points
    scale = star.params['numax_sun']/((star.maxpower[1]+star.maxpower[0])/2.0)
    taus = np.array(star.params[scaling])*scale
    b = 2.0*np.pi*(taus*1e-6)
    mnu = (1.0/taus)*1e5
    star.b = b[mnu >= min(star.frequency)]
    star.mnu = mnu[mnu >= min(star.frequency)]
    if len(star.mnu)==0:
        star.b = b[mnu >= 10] 
        star.mnu = mnu[mnu >= 10]
    star.nlaws = len(star.mnu)
    star.mnu_orig = np.copy(star.mnu)
    star.b_orig = np.copy(star.b)

    return star


def save_findex(star):
    """Save the results of the find excess routine into the save folder of the current star.

    Parameters
    ----------
    results : list
        the results of the find excess routine
    """
    best = star.findex['results'][star.name]['best']
    variables = ['star', 'numax', 'dnu', 'snr']
    results = [star.name, star.findex['results'][star.name][best]['numax'], star.findex['results'][star.name][best]['dnu'], star.findex['results'][star.name][best]['snr']]
    save_path = '%sexcess.csv' % star.params[star.name]['path']
    ascii.write(np.array(results), save_path, names=variables, delimiter=',', overwrite=True)


def convert_samples(df, drop=True):
    """
    Converts {a_n, b_n} Harvey parametrization to {tau_n, sigma_n} before
    saving any information from fit_background. This makes it easier to
    compute uncertainties for the latter without having to do error 
    propagation.

    Parameters
    ----------
    df : pandas.DataFrame
        the results of the fits background routine
    drop : bool
        drops the additional columns after the samples are converted
    cols_to_drop : list
        the list of columns to drop

    Returns
    -------
    df : pandas.DataFrame
        the converted results of the fits background routine

    """
    cols_to_drop=[]
    for column in df.columns.values.tolist():
        if 'a_' in column:
            n = int(column.split('_')[-1])
            b = ((df['b_%d'%n].values)**2.)**(0.5)
            sigma = ((df[column].values*np.pi)/(2.*b))**(0.5)
            df['sigma_%d'%n] = sigma
            cols_to_drop.append(column)
        elif 'b_' in column:
            n = int(column.split('_')[-1])
            b = ((df['b_%d'%n].values)**2.)**(0.5)
            tau = (b*10**6.)/(2.*np.pi)
            df['tau_%d'%n] = tau
            cols_to_drop.append(column)
        else:
            pass
    if drop and cols_to_drop != []:
        df.drop(columns=cols_to_drop, inplace=True)

    return df


def save_fitbg(star):
    """
    Saves the results of the `fit_background` module.

    """
    df = pd.DataFrame(star.fitbg['results'][star.name])
    if star.fitbg['convert']:
        df = convert_samples(df, drop=star.fitbg['drop'])
    star.df = df.copy()
    new_df = pd.DataFrame(columns=['parameter', 'value', 'uncertainty'])
    for c, col in enumerate(df.columns.values.tolist()):
        new_df.loc[c, 'parameter'] = col
        new_df.loc[c, 'value'] = df.loc[0,col]
        if star.fitbg['mc_iter'] > 1:
            new_df.loc[c, 'uncertainty'] = mad_std(df[col].values)
        else:
            new_df.loc[c, 'uncertainty'] = '--'
    new_df.to_csv('%sbackground.csv' % star.params[star.name]['path'], index=False)
    if star.fitbg['samples']:
        df.to_csv('%ssamples.csv' % star.params[star.name]['path'], index=False)


def verbose_output(star, sampling=False):
    """
    Print results of the `fit_background` routine if verbose is `True`.

    """
    note=''
    df = pd.read_csv('%sbackground.csv' % star.params[star.name]['path'])
    params = get_params_dict()
    if sampling:
        note+='\nOutput parameters:'
        line='\n%s: %.2f +/- %.2f %s'
        for idx in df.index.values.tolist():
            note+=line%(df.loc[idx,'parameter'],df.loc[idx,'value'],df.loc[idx,'uncertainty'],params[df.loc[idx,'parameter']]['unit'])
    else:
        note+='-------------------------------------------------\nOutput parameters:'
        line='\n%s: %.2f %s'
        for idx in df.index.values.tolist():
            note+=line%(df.loc[idx,'parameter'],df.loc[idx,'value'],params[df.loc[idx,'parameter']]['unit'])
    note+='\n-------------------------------------------------'
    print(note)


def scrape_output(args):
    """
    Grabs each individual star's results and concatenates results into a single csv in info/ for each submodule
    (i.e. excess.csv and background.csv). This is automatically called if the pySYD is successfully executed for 
    at least one star.

    """

    path = '%s/**/'%args.params['outdir']
    # Findex outputs
    output = '%s*excess.csv'%path
    files = glob.glob(output)
    df = pd.read_csv(files[0])
    for i in range(1,len(files)):
        df_new = pd.read_csv(files[i])
        df = pd.concat([df, df_new])
    df.to_csv('%s/excess.csv'%args.params['outdir'], index=False)

    # Fitbg outputs
    output = '%s*background.csv'%path
    files = glob.glob(output)
    df = pd.DataFrame(columns=['star'])

    for i, file in enumerate(files):
	       df_new = pd.read_csv(file)
	       df_new.set_index('parameter',inplace=True,drop=False)
	       df.loc[i,'star']=file.strip().split('/')[-2]
	       new_header_names=[[i,i+'_err'] for i in df_new.index.values.tolist()]
	       new_header_names=list(chain.from_iterable(new_header_names))          
	       for col in new_header_names:
		          if '_err' in col:
			             df.loc[i,col]=df_new.loc[col[:-4],'uncertainty']
		          else:
			             df.loc[i,col]=df_new.loc[col,'value']

    df.fillna('--', inplace=True)
    df.to_csv('%s/background.csv'%args.params['outdir'], index=False)


def make_latex_table(path, lines='', header=True, columns=None, labels=None, formats=None, units=None, 
                     include_units=True, include_uncertainty=True, save=True, verbose=True):

    # Get data
    df = pd.read_csv(path)
    if columns is None:
        columns = get_data_columns(type='table')
    # Get parameter information
    params = get_params_dict()

    # Make header row
    if header:
        line='Star & '
        for column in columns:
            if include_units:
                line+='%s [%s] & '%(params[column]['latex']['label'],params[column]['latex']['unit'])
            else:
                line+='%s & '%params[column]['latex']['label']
        line = line[:-3] + ' \\\ '
        if verbose:
            print(line)
        lines+=line+'\n'

    # Iterate through all targets
    for i in df.index.values.tolist():
        line='%d & '%df.loc[i,'star']
        for column in columns:
            if df.loc[i,column] != '--':
                if include_uncertainty and '%s_err'%column in df.columns.values.tolist():
                    par = '%s $\pm$ %s & '%(params[column]['latex']['format'], params[column]['latex']['format'])
                    line += par%(df.loc[i,column],df.loc[i,'%s_err'%column])
                else:
                    par = '%s & '%params[column]['latex']['format']
                    line += par%df.loc[i,column]
            else:
                line += '-- & '
        line = line[:-3] + ' \\\ '
        if verbose:
            print(line)
        lines+=line+'\n'

    if save:
        if len(path.split('/')) > 1:
            new_fn='%s/%s.txt'%('/'.join(path.split('/')[:-1]),(path.split('/')[-1]).split('.')[0])
        else:
            new_fn='%s.txt'%(path.split('/')[-1]).split('.')[0]
        f = open(new_fn, "w")
        f.write(lines)
        f.close()


def get_params_dict():

    params={
            'numax_smooth':{'unit':'muHz','label':r'$\rm Smoothed \,\, \nu_{max} \,\, [\mu Hz]$','latex':{'label':'$\\nu_{\\mathrm{max}}$', 'format':'%.2f', 'unit':'$\\rm \\mu Hz$'}}, 
            'A_smooth':{'unit':'ppm^2/muHz','label':r'$\rm Smoothed \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\\rm A_{osc}$', 'format':'%.2f', 'unit':'$\\rm ppm^{2} \\mu Hz^{-1}$'}}, 
            'numax_gauss':{'unit':'muHz','label':r'$\rm Gaussian \,\, \nu_{max} \,\, [\mu Hz]$','latex':{'label':'$\\nu_{\\mathrm{max}}$', 'format':'%.2f', 'unit':'$\\rm \\mu Hz$'}}, 
            'A_gauss':{'unit':'ppm^2/muHz','label':r'$\rm Gaussian \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\\rm A_{osc}$', 'format':'%.2f', 'unit':'$\\rm ppm^{2} \\mu Hz^{-1}$'}}, 
            'FWHM':{'unit':'muHz','label':r'$\rm Gaussian \,\, FWHM \,\, [\mu Hz]$','latex':{'label':'FWHM', 'format':'%.2f', 'unit':'$\\rm \\mu Hz$'}}, 
            'dnu':{'unit':'muHz','label':r'$\rm \Delta\nu \,\, [\mu Hz]$','latex':{'label':'$\\Delta\\nu$', 'format':'%.2f', 'unit':'$\\rm \\mu Hz$'}},
            'white':{'unit':'ppm^2/muHz','label':r'$\rm White \,\, noise \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'White noise', 'format':'%.2f', 'unit':'$\\rm ppm^{2} \\mu Hz^{-1}$'}}, 
            'a_1':{'unit':'ppm^2/muHz','label':r'$\rm a_{1} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\\rm a_{1}$', 'format':'%.2f', 'unit':'$\\rm ppm^{2} \\mu Hz^{-1}$'}},
            'b_1':{'unit':'muHz^-1','label':r'$\rm b_{1} \,\, [\mu Hz^{-1}]$','latex':{'label':'$\rm b_{1}$', 'format':'%.2f', 'unit':'$\rm \mu Hz^{-1}$'}}, 
            'tau_1':{'unit':'s','label':r'$\rm \tau_{1} \,\, [s]$','latex':{'label':'$\tau_{1}$', 'format':'%.2f', 'unit':'s'}},
            'sigma_1':{'unit':'ppm','label':r'$\rm \sigma_{1} \,\, [ppm]$','latex':{'label':'$\sigma_{1}$', 'format':'%.2f', 'unit':'ppm'}}, 
            'a_2':{'unit':'ppm^2/muHz','label':r'$\rm a_{2} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\rm a_{2}$', 'format':'%.2f', 'unit':'$\rm ppm^{2} \mu Hz^{-1}$'}},
            'b_2':{'unit':'muHz^-1','label':r'$\rm b_{2} \,\, [\mu Hz^{-1}]$','latex':{'label':'$\rm b_{2}$', 'format':'%.2f', 'unit':'$\rm \mu Hz^{-1}$'}}, 
            'tau_2':{'unit':'s','label':r'$\rm \tau_{2} \,\, [s]$','latex':{'label':'$\tau_{2}$', 'format':'%.2f', 'unit':'s'}}, 
            'sigma_2':{'unit':'ppm','label':r'$\rm \sigma_{2} \,\, [ppm]$','latex':{'label':'$\sigma_{2}$', 'format':'%.2f', 'unit':'ppm'}},
            'a_3':{'unit':'ppm^2/muHz','label':r'$\rm a_{3} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\rm a_{3}$', 'format':'%.2f', 'unit':'$\rm ppm^{2} \mu Hz^{-1}$'}},
            'b_3':{'unit':'muHz^-1','label':r'$\rm b_{3} \,\, [\mu Hz^{-1}]$','latex':{'label':'$\rm b_{3}$', 'format':'%.2f', 'unit':'$\rm \mu Hz^{-1}$'}}, 
            'tau_3':{'unit':'s','label':r'$\rm \tau_{3} \,\, [s]$','latex':{'label':'$\tau_{3}$', 'format':'%.2f', 'unit':'s'}}, 
            'sigma_3':{'unit':'ppm','label':r'$\rm \sigma_{3} \,\, [ppm]$','latex':{'label':'$\sigma_{3}$', 'format':'%.2f', 'unit':'ppm'}}, 
            'a_4':{'unit':'ppm^2/muHz','label':r'$\rm a_{4} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\rm a_{4}$', 'format':'%.2f', 'unit':'$\rm ppm^{2} \mu Hz^{-1}$'}},
            'b_4':{'unit':'muHz^-1','label':r'$\rm b_{4} \,\, [\mu Hz^{-1}]$','latex':{'label':'$\rm b_{4}$', 'format':'%.2f', 'unit':'$\rm \mu Hz^{-1}$'}}, 
            'tau_4':{'unit':'s','label':r'$\rm \tau_{4} \,\, [s]$','latex':{'label':'$\tau_{4}$', 'format':'%.2f', 'unit':'s'}},
            'sigma_4':{'unit':'ppm','label':r'$\rm \sigma_{4} \,\, [ppm]$','latex':{'label':'$\sigma_{4}$', 'format':'%.2f', 'unit':'ppm'}}, 
            'a_5':{'unit':'ppm^2/muHz','label':r'$\rm a_{5} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\rm a_{5}$', 'format':'%.2f', 'unit':'$\rm ppm^{2} \mu Hz^{-1}$'}},
            'b_5':{'unit':'muHz^-1','label':r'$\rm b_{5} \,\, [\mu Hz^{-1}]$','latex':{'label':'$\rm b_{5}$', 'format':'%.2f', 'unit':'$\rm \mu Hz^{-1}$'}}, 
            'tau_5':{'unit':'s','label':r'$\rm \tau_{5} \,\, [s]$','latex':{'label':'$\tau_{5}$', 'format':'%.2f', 'unit':'s'}}, 
            'sigma_5':{'unit':'ppm','label':r'$\rm \sigma_{5} \,\, [ppm]$','latex':{'label':'$\sigma_{5}$', 'format':'%.2f', 'unit':'ppm'}}, 
            'a_6':{'unit':'ppm^2/muHz','label':r'$\rm a_{6} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\rm a_{6}$', 'format':'%.2f', 'unit':'$\rm ppm^{2} \mu Hz^{-1}$'}},
            'b_6':{'unit':'muHz^-1','label':r'$\rm b_{6} \,\, [\mu Hz^{-1}]$','latex':{'label':'$\rm b_{6}$', 'format':'%.2f', 'unit':'$\rm \mu Hz^{-1}$'}}, 
            'tau_6':{'unit':'s','label':r'$\rm \tau_{6} \,\, [s]$','latex':{'label':'$\tau_{6}$', 'format':'%.2f', 'unit':'s'}}, 
            'sigma_6':{'unit':'ppm','label':r'$\rm \sigma_{6} \,\, [ppm]$','latex':{'label':'$\sigma_{6}$', 'format':'%.2f', 'unit':'ppm'}},
            }

    return params


def get_data_columns(type=['full','required','table']):

    if type == 'full':
        columns = ['stars','radius','radius_err','teff','teff_err','logg','logg_err',
                   'numax','lower_x','upper_x','lower_b','upper_b','seed']
    elif type == 'required':
        columns = ['radius','logg','teff','numax','lower_x','upper_x','lower_b','upper_b','seed']
    elif type == 'table':
        columns = ['numax_smooth', 'A_smooth', 'FWHM', 'dnu', 'white']
    else:
        print("Did not understand that input, please specify type. Choose from: ['full','required','table']")
        columns = []

    return columns