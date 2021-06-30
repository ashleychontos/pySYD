import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from astropy.io import ascii
import multiprocessing as mp
from astropy.stats import mad_std
from astropy.timeseries import LombScargle as lomb

from pysyd.plots import set_plot_params
from pysyd.functions import *
from pysyd.models import *



def get_info_all(args, parallel, CLI=True):
    """
    Loads todo.txt, sets up file paths, loads in any available star information, saves the 
    relevant parameters for each of the two main routines and sets the plotting parameters.

    Parameters
    ----------
    args : argparse.Namespace
        command-line arguments
    parallel : bool
        if pysyd will be running in parallel mode
    CLI : bool, optional
        if CLI is not being used (i.e. `False`), the modules draw default values from a different location

    Returns
    -------
    args : argparse.Namespace
        the updated command-line arguments

    """
    # Get parameters for all modules
    args = get_pysyd_parameters(args, parallel, CLI)
    # Get invidual/specific star info from csv file (if it exists)
    args = get_csv_info(args)
    if CLI:
        # Check the input variables
        check_input_args(args)
        args = get_command_line(args)
    set_plot_params()

    return args


def get_pysyd_parameters(args, parallel, CLI):
    """
    Basic function to call the individual functions that load and
    save parameters for different modules.

    Parameters
    ----------
    args : argparse.Namespace
        command-line arguments
    CLI : bool
        `True` if running pysyd via command line

    Returns
    -------
    args : argparse.Namespace
        the updated command-line arguments

    """

    # Initialize main 'params' dictionary
    args = get_main_params(args, CLI)
    args = get_groups(args, parallel)
    # Initialize parameters for the find excess routine
    args = get_excess_params(args, CLI)
    # Initialize parameters for the fit background routine
    args = get_background_params(args, CLI)
    # Initialize parameters relevant for estimating global parameters
    args = get_global_params(args, CLI)

    return args


def get_main_params(args, CLI, stars=None, verbose=False, show=False, save=True, 
                    kepcorr=False, of_actual=None, of_new=None,):
    """
    Get the parameters for the find excess routine.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    stars : List[str], optional
        list of targets to process. If `None`, will read in from `info/todo.txt` (default).
    verbose : bool, optional
        turn on verbose output. Default is `False`.
    show : bool, optional
        show output figures. Default is `False`.
    save : bool, optional
        save all data products. Default is `True`.
    kepcorr : bool, optional
        use the module that corrects for known kepler artefacts. Default is `False`.
    of_actual : int, optional
        oversampling factor of input PS. Default value is `None`.
    of_new : int, optional
        oversampling factor of newly-computed PS. Default value is `None`.

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    args.params : Dict[str,object]
        the parameters of higher-level functionality

    """
    if CLI:
        params = {
            'stars': args.stars,
            'inpdir': args.inpdir,
            'outdir': args.outdir,
            'info': args.info,
            'show': args.show,
            'save': args.save,
            'of_actual': args.of_actual,
            'of_new': args.of_new,
            'kepcorr': args.kepcorr,
        }
    else:
        args.todo = os.path.join(os.path.abspath(os.getcwd()), 'info', 'todo.txt')
        args.info = os.path.join(os.path.abspath(os.getcwd()), 'info', 'star_info.csv')
        params = {
            'stars': stars,
            'inpdir': os.path.join(os.path.abspath(os.getcwd()), 'data'),
            'outdir': os.path.join(os.path.abspath(os.getcwd()), 'results'),
            'info': args.info,
            'show': show,
            'save': save,
            'of_actual': of_actual,
            'of_new': of_new,
            'kepcorr': kepcorr,
        }

    # Open star list
    if params['stars'] is None or params['stars'] == []:
        with open(args.todo, "r") as f:
            params['stars'] = [line.strip().split()[0] for line in f.readlines()]

    # Set file paths and make directories if they don't yet exist
    for star in params['stars']:
        params[star] = {}
        params[star]['path'] = os.path.join(params['outdir'], star)
        if params['save'] and not os.path.exists(params[star]['path']):
            os.makedirs(params[star]['path'])
    args.params = params

    return args


def get_groups(args, parallel=False):
    """
    Sets up star groups to run in parallel based on the number of threads.

    Parameters
    ----------
    args : argparse.Namespace
        command line arguments
    parallel : bool
        run pySYD in parallel

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    args.params['groups'] : ndarray
        star groups to process (groups == number of threads)

    Returns
    ----------
    None

    """
    if parallel:
        todo = np.array(args.params['stars'])
        args.params['verbose'] = False
        args.params['show'] = False
        if args.n_threads == 0:
            args.n_threads = mp.cpu_count()
        if len(todo) < args.n_threads:
            args.n_threads = len(todo)
        # divide stars into groups set by number of cpus/nthreads available
        digitized = np.digitize(np.arange(len(todo))%args.n_threads,np.arange(args.n_threads))
        args.params['groups'] = np.array([todo[digitized == i] for i in range(1, args.n_threads+1)], dtype=object)
    else:
        args.params['groups'] = np.array([])

    return args


def get_excess_params(args, CLI, n_trials=3, step=0.25, binning=0.005, smooth_width=50.0, 
                      lower_ex=10.0, upper_ex=4000.0):
    """
    Get the parameters for the find excess routine.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    lower_ex : float, optional
        the lower frequency bound (in muHz). Default value is `10.0` muHz.
    upper_ex : float, optional
        the upper frequency bound (in muHz). Default value is `4000.0` muHz.
    step : float, optional
        TODO: Write description. Default value is `0.25`.
    binning : float, optional
        logarithmic binning width. Default value is `0.005`.
    mode : {'mean', 'median', 'gaussian'}
        mode to use when binning
    n_trials : int, optional
        the number of trials. Default value is `3`.

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    args.findex : Dict[str,object]
        the parameters of the find excess routine

    """
    if CLI:
        findex = {
            'lower_ex': lower_ex,
            'upper_ex': upper_ex,
            'step': args.step,
            'binning': args.binning,
            'mode': args.mode,
            'smooth_width': args.smooth_width,
            'n_trials': args.n_trials,
            'results': {},
        }
    else:
        findex = {
            'lower_ex': lower_ex,
            'upper_ex': upper_ex,
            'step': step,
            'binning': binning,
            'mode': mode,
            'smooth_width': smooth_width,
            'n_trials': n_trials,
            'results': {},
        }
    args.findex = findex

    return args


def get_background_params(args, CLI, lower_bg=10.0, upper_bg=4000.0, ind_width=20.0,   
                          box_filter=1.0, n_rms=20, mc_iter=1, samples=False, n_laws=None,
                          use='bic',):
    """
    Get the parameters for the background-fitting routine.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    lower_bg : float, optional
        the lower frequency bound (in muHz). Default value is `10.0` muHz.
    upper_bg : float, optional
        the upper frequency bound (in muHz). Default value is `4000.0` muHz.
    box_filter : float
        the size of the 1D box smoothing filter (in muHz). Default value is `1.0`.
    ind_width : float
        the independent average smoothing width (in muHz). Default value is `20.0`.
    n_rms : int
        number of data points to estimate red noise contributions. Default value is `20`.
    use : str
        which metric to use (i.e. bic or aic) for model selection. Default is `'bic'`.
    n_laws : int
        force number of Harvey-like components in background fit. Default value is `None`.
    mc_iter : int
        number of samples used to estimate uncertainty. Default value is `1`.
    samples : bool
        if true, will save the monte carlo samples to a csv. Default value is `False`.
    args.n_peaks : int
        the number of peaks to select. Default value is `10`.
    args.force : float
        if not false (i.e. non-zero) will force dnu to be the equal to this value. 
    args.clip : bool
        if true will set the minimum frequency value of the echelle plot to `clip_value`. Default value is `True`.
    args.clip_value : float
        the minimum frequency of the echelle plot. Default value is `0.0`.
    args.smooth_ech : float
        option to smooth the output of the echelle plot
    args.smooth_ps : float
        frequency with which to smooth power spectrum. Default value is `1.0`.
    args.slope : bool
        if true will correct for edge effects and residual slope in Gaussian fit. Default value is `False`.
    args.samples : bool
        if true, will save the monte carlo samples to a csv. Default value is `True`.
    args.convert : bool
        converts Harvey parametrization to physical quantities {a_n,b_n} -> {tau_n,sigma_n}. Default value is `True`.
    args.drop : bool
        drops the extra columns after converting the samples. Default value is `True`.

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    args.fitbg : Dict[str,object]
        the parameters relevant for the fit background routine

    """
    if CLI:
        fitbg = {
            'lower_bg': lower_bg,
            'upper_bg': upper_bg,
            'ind_width': args.ind_width,
            'box_filter': args.box_filter,
            'n_rms': args.n_rms,
            'n_laws': args.n_laws,
            'metric': args.use,
            'functions': {0: harvey_none, 1: harvey_one, 2: harvey_two, 3: harvey_three},
            'mc_iter': args.mc_iter,
            'samples': args.samples,
            'results': {},
        }
    else:
        fitbg = {
            'lower_bg': lower_bg,
            'upper_bg': upper_bg,
            'ind_width': ind_width,
            'box_filter': box_filter,
            'n_rms': n_rms,
            'n_laws': n_laws,
            'metric': use,
            'functions': {0: harvey_none, 1: harvey_one, 2: harvey_two, 3: harvey_three},
            'mc_iter': mc_iter,
            'samples': samples,
            'results': {},
        }
    args.fitbg = fitbg

    return args


def get_global_params(args, CLI, sm_par=None, numax=None, lower_ps=None, upper_ps=None,
                      width=1.0, dnu=None, smooth_ps=2.5, threshold=1.0, n_peaks=5, 
                      clip_ech=True, clip_value=None, smooth_ech=None, interp_ech=False, 
                      lower_ech=None, upper_ech=None, n_across=50, n_down=5,):
    """
    Get the parameters relevant for finding global asteroseismic parameters numax and dnu.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    sm_par : float
        Gaussian filter width for determining smoothed numax (values are typically between 1-4)
    numax : float
        guess for numax
    lower_ps : float
        lower bound of power excess (in muHz). Default value is `None`.
    upper_ps : float
        upper bound of power excess (in muHz). Default value is `None`.
    width : float
        fractional width to use for power excess centerd on numax. Default value is `1.0`.
    smooth_ps : float
        box filter [in muHz] for PS smoothing before calculating ACF. Default value is `1.5`.
    threshold : float
        fractional width of FWHM to use in ACF for later iterations. Default value is `1.0`.
    n_peaks : int
        the number of peaks to select. Default value is `10`.
    lower_ech : float
        lower bound of folded PS (in muHz) to 'whiten' mixed modes. Default value is `None`.
    upper_ech : float
        upper bound of folded PS (in muHz) to 'whiten' mixed modes. Default value is `None`.
    clip : bool
        if true will set the minimum frequency value of the echelle plot to `clip_value`. Default value is `True`.
    clip_value : float
        the minimum frequency of the echelle plot. Default value is `0.0`.
    smooth_ech : float
        option to smooth the output of the echelle plot
    interp_ech : bool
        turns on the bilinear smoothing in echelle plot
    n_across : int
        x-axis resolution on the echelle diagram. Default value is `50`. (NOT CURRENTLY IMPLEMENTED YET)
    n_down : int
        how many radial orders to plot on the echelle diagram. Default value is `5`. (NOT CURRENTLY IMPLEMENTED YET)

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    args.globe : Dict[str,object]
        the parameters relevant for determining the global parameters routine

    """
    if CLI:
        globe = {
            'sm_par': args.sm_par,
            'width': args.width,
            'smooth_ps': args.smooth_ps,
            'threshold': args.threshold,
            'n_peaks': args.n_peaks,
            'clip_ech': args.clip_ech,
            'clip_value': args.clip_value,
            'smooth_ech': args.smooth_ech,
            'interp_ech': args.interp_ech,
            'n_across': args.n_across,
            'n_down': args.n_down,
        }
    else:
        globe = {
            'sm_par': sm_par,
            'width': width,
            'smooth_ps': smooth_ps,
            'threshold': threshold,
            'n_peaks': n_peaks,
            'clip_ech': clip_ech,
            'clip_value': clip_value,
            'smooth_ech': smooth_ech,
            'interp_ech': interp_ech,
            'n_across': n_across,
            'n_down': n_down,
        }
    args.globe = globe

    return args


def get_csv_info(args):
    """
    Reads in any star information provided via args.info and is 'info/star_info.csv' by default. 
    ** Please note that this is NOT required for pySYD to run successfully **

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    columns : list
        the list of columns to provide stellar information for

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments

    """
    constants = Constants()
    columns = get_data_columns(type='required')
    # Open file if it exists
    if os.path.exists(args.info):
        df = pd.read_csv(args.info)
        stars = [str(each) for each in df.stars.values.tolist()]
        for i, star in enumerate(args.stars):
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
                        args.params[star]['mass'] = ((((args.params[star]['radius']*constants.r_sun)**(2.0))*10**(args.params[star]['logg'])/constants.G)/constants.m_sun)
                        args.params[star]['numax'] = constants.numax_sun*args.params[star]['mass']*(args.params[star]['radius']**(-2.0))*((args.params[star]['teff']/constants.teff_sun)**(-0.5))
                        args.params[star]['dnu'] = constants.dnu_sun*(args.params[star]['mass']**(0.5))*(args.params[star]['radius']**(-1.5))  
            # if target isn't in csv, still save basic parameters called througout the code
            else:
                for column in columns:
                    args.params[star][column] = None
    # same if the file does not exist
    else:
        for star in args.stars:
            args.params[star]['excess'] = args.excess
            args.params[star]['background'] = args.background
            args.params[star]['force'] = False
            for column in columns:
                args.params[star][column] = None
    return args


def check_input_args(args, max_laws=3):
    """ 
    Make sure that any command-line inputs are the proper lengths, types, etc.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    max_laws : int
        maximum number of resolvable Harvey components

    Yields
    ------
    ???

    """

    checks={'lower_bg':args.lower_bg,'upper_bg':args.upper_bg,'lower_ex':args.lower_ex,
            'upper_ex':args.upper_ex,'lower_ps':args.lower_ps,'upper_ps':args.upper_ps,
            'lower_ech':args.lower_ech,'upper_ech':args.upper_ech,'dnu':args.dnu,'numax':args.numax}
    for check in checks:
        if checks[check] is not None:
            assert len(args.stars) == len(checks[check]), "The number of values provided for %s does not equal the number of stars"%check
    if args.of_actual is not None:
        assert isinstance(args.of_actual,int), "The oversampling factor for the input PS must be an integer"
    if args.of_new is not None:
        assert isinstance(args.of_new,int), "The new oversampling factor must be an integer"
    if args.n_laws is not None:
        assert args.n_laws <= max_laws, "We likely cannot resolve %d Harvey-like components for point sources. Please select a smaller number."%args.n_laws


def get_command_line(args, numax=None, dnu=None, lower_ps=None, upper_ps=None, 
                     lower_ech=None, upper_ech=None):
    """
    If certain CLI options are provided, it saves it to the appropriate star. This
    is called after the csv is checked and therefore, this will override any duplicated
    information provided there (if applicable).

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    args.lower_ps : float, optional
        the lower frequency bound for numax (in muHz). Default is `None`.
    args.upper_ps : float, optional
        the upper frequency bound for numax (in muHz). Default is `None`.
    args.numax : List[float], optional
        the estimated numax (in muHz). Default is `None`.
    args.dnu : List[float], optional
        the estimated frequency spacing or dnu (in muHz). Default is `None`.
    args.lower_ech : List[float], optional
        the lower frequency for whitening the folded PS (in muHz). Default is `None`.
    args.upper_ech : List[float], optional
        the upper frequency for whitening the folded PS (in muHz). Default is `None`.

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments

    """

    override = {
        'lower_bg': args.lower_bg,
        'upper_bg': args.upper_bg,
        'lower_ex': args.lower_ex,
        'upper_ex': args.upper_ex,
        'lower_ps': args.lower_ps,
        'upper_ps': args.upper_ps,
        'numax': args.numax,
        'dnu': args.dnu,
        'lower_ech': args.lower_ech,
        'upper_ech': args.upper_ech,
    }

    for i, star in enumerate(args.stars):
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
        if args.params[star]['force']:
            args.params[star]['dnu'] = args.params[star]['guess']
        if args.params[star]['lower_ech'] is not None and args.params[star]['upper_ech'] is not None:
            args.params[star]['ech_mask'] = [args.params[star]['lower_ech'],args.params[star]['upper_ech']]
        else:
            args.params[star]['ech_mask'] = None

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
    args : argparse.Namespace
        command line arguments

    Returns
    -------
    star : target.Target
        the pySYD pipeline object
    star.lc : bool
        will return `True` if the light curve data was loaded in properly otherwise `False`
    star.ps : bool
        will return `True` if the power spectrum file was successfully loaded otherwise `False`

    """
    # Now done at beginning to make sure it only does this once per star
    if glob.glob(os.path.join(args.inpdir,'%s*'%str(star.name))) != []:
        if star.verbose:
            print('\n\n----------------------------------------------------')
            print('Target: %s'%str(star.name))
            print('----------------------------------------------------')

        # Load light curve
        args, star, note = load_time_series(args, star)
        if star.verbose:
            print(note)

        # Load power spectrum
        args, star, note = load_power_spectrum(args, star)
        if star.verbose:
            print(note)

    return star


def load_file(path):
    """
    Load a light curve or a power spectrum from a basic 2xN txt file
    and stores the data into the `x` (independent variable) and `y`
    (dependent variable) arrays, where N is the length of the series.

    Parameters
    ----------
    path : str
        the file path of the data file

    Returns
    -------
    x : numpy.array
        the independent variable i.e. the time or frequency array 
    y : numpy.array
        the dependent variable, in this case either the flux or power array

    """
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    # Set values
    x = np.array([float(line.strip().split()[0]) for line in lines])
    y = np.array([float(line.strip().split()[1]) for line in lines])
    return x, y


def load_time_series(args, star, note=''):
    """
    If available, star.lc is set to `True`, the time series data
    is loaded in, and then it calculates the cadence and nyquist 
    freqency. If time series data is not provided, either the
    cadence or nyquist frequency must be provided via CLI

    Parameters
    ----------
    star : target.Target
        the pySYD pipeline object
    args : argparse.Namespace
        command line arguments
    args.cadence : int
        cadence of time series data (if known but data is not available)
    args.nyquist : float
        nyquist frequency of the provided power spectrum
    note : str
        optional suppressed verbose output

    Returns
    -------
    star : target.Target
        the pySYD pipeline object
    star.lc : bool
        will return `True` if the light curve data was loaded in properly otherwise `False`
    star.time : numpy.array
        time array in days
    star.flux : numpy.array
        relative or normalized flux array

    """
    star.lc = False
    # Try loading the light curve
    if os.path.exists(os.path.join(args.inpdir, '%s_LC.txt'%star.name)):
        star.lc = True
        star.time, star.flux = load_file(os.path.join(args.inpdir, '%s_LC.txt'%star.name))
        star.time -= min(star.time)
        star.cadence = int(round(np.nanmedian(np.diff(star.time)*24.0*60.0*60.0),0))
        star.nyquist = 10**6./(2.0*star.cadence)
        star.baseline = (max(star.time)-min(star.time))*24.*60.*60.
        note += '# LIGHT CURVE: %d lines of data read\n# Time series cadence: %d seconds'%(len(star.time),star.cadence)

    return args, star, note


def load_power_spectrum(args, star, note='', long=10**6):
    """
    Loads in the power spectrum data in for a given star,
    which will return `False` if unsuccessful and therefore, not run the rest
    of the pipeline.

    Parameters
    ----------
    star : target.Target
        the pySYD pipeline object
    args : argparse.Namespace
        command line arguments
    args.kepcorr : bool
        if true, will run the module to mitigate the Kepler artefacts in the power spectrum. Default is `False`.
    args.of_actual : int
        the oversampling factor, if the power spectrum is already oversampled. Default is `1`, assuming a critically sampled PS.
    args.of_new : float
        the oversampling factor to use for the first iterations. Default is `5`.
    note : str
        optional suppressed verbose output
    long : int
        will display a warning if length of PS is longer than 10**6 lines 

    Returns
    -------
    star : target.Target
        the pySYD pipeline object
    star.ps : bool
        will return `True` if the power spectrum file was successfully loaded otherwise `False`
    star.frequency : numpy.array
        frequency array in muHz
    star.power : numpy.array
        power spectral density array

    """
    star.ps = False
    # Try loading the power spectrum
    if not os.path.exists(os.path.join(args.inpdir, '%s_PS.txt'%star.name)):
        note += '# ERROR: %s/%s_PS.txt not found\n'%(args.inpdir, star.name)
    else:
        star.ps = True
        star.frequency, star.power = load_file(os.path.join(args.inpdir, '%s_PS.txt'%star.name))
        note += '# POWER SPECTRUM: %d lines of data read\n'%len(star.frequency)
        if len(star.frequency) >= long:
            note += '# WARNING: PS is large and will slow down the software'
        star.resolution = star.frequency[1]-star.frequency[0]
        if args.kepcorr:
            note += '# **using Kepler artefact correction**\n'
            star = remove_artefact(star)
        if star.params[star.name]['ech_mask'] is not None:
            note += '# **whitening the PS to remove mixed modes**\n'
            star = whiten_mixed(star)
        args, star, note = check_input(args, star, note)

    return args, star, note


def check_input(args, star, note):
    """
    Checks the type(s) of input data and creates any additional, optional
    arrays as well as critically-sampled power spectra (when applicable).

    Parameters
    ----------
    args : argparse.Namespace
        command line arguments
    star : target.Target
        pySYD target object
    note : str, optional
        optional verbose output

    Returns
    -------
    args : argparse.Namespace
        updated command line arguments
    star : target.Target
        updated pySYD target object
    note : str, optional
        updated optional verbose output

    """
    if star.lc:
        args.of_actual = int(round((1./((max(star.time)-min(star.time))*0.0864))/(star.frequency[1]-star.frequency[0])))
        star.freq_cs = np.array(star.frequency[args.of_actual-1::args.of_actual])
        star.pow_cs = np.array(star.power[args.of_actual-1::args.of_actual])
        if args.of_new is not None:
            note += '# Computing new PS using oversampling of %d\n'%args.of_new
            freq_os, pow_os = lomb(star.time, star.flux).autopower(method='fast', samples_per_peak=args.of_new, maximum_frequency=star.nyquist)
            star.freq_os = freq_os*(10.**6/(24.*60.*60.))
            star.pow_os = 4.*pow_os*np.var(star.flux*1e6)/(np.sum(pow_os)*(star.freq_os[1]-star.freq_os[0]))
        else:
            star.freq_os, star.pow_os = np.copy(star.frequency), np.copy(star.power)
    else:
        if args.of_actual is not None:
            star.freq_cs = np.array(star.frequency[args.of_actual-1::args.of_actual])
            star.pow_cs = np.array(star.power[args.of_actual-1::args.of_actual])
            star.freq_os, star.pow_os = np.copy(star.frequency), np.copy(star.power)                    
        else:
            star.freq_cs, star.pow_cs = np.copy(star.frequency), np.copy(star.power)
            star.freq_os, star.pow_os = np.copy(star.frequency), np.copy(star.power)
            note += '# WARNING: using input PS with no additional information'
            if args.mc_iter > 1:
                note += '# **uncertainties may not be reliable unless using a critically-sampled PS**'
        star.baseline = 1./((star.freq_cs[1]-star.freq_cs[0])*10**-6.)
    if args.of_actual is not None:
        note += '# PS is oversampled by a factor of %d\n'%args.of_actual
    else:
        note += '# Assuming PS is critically-sampled\n'
    note += '# PS resolution: %.6f muHz'%(star.freq_cs[1]-star.freq_cs[0])
    return args, star, note


def get_findex(star):
    """
    Before running the first module (find excess), this masks out any unwanted
    frequency regions and also gets appropriate bin sizes for the collapsed ACF
    function if some prior information on numax is provided.

    Parameters
    ----------
    star : target.Target
        pySYD target object

    Returns
    -------
    star : target.Target
        updated pySYD target object

    """
    # If running the first module, mask out any unwanted frequency regions
    star.frequency, star.power = np.copy(star.freq_os), np.copy(star.pow_os)
    star.resolution = star.frequency[1]-star.frequency[0]
    # mask out any unwanted frequencies
    if star.params[star.name]['lower_ex'] is not None:
        lower = star.params[star.name]['lower_ex']
    else:
        lower = star.findex['lower_ex']
    if star.params[star.name]['upper_ex'] is not None:
        upper = star.params[star.name]['upper_ex']
    else:
        upper = star.findex['upper_ex']
    star.freq = star.frequency[(star.frequency >= lower)&(star.frequency <= upper)]
    star.pow = star.power[(star.frequency >= lower)&(star.frequency <= upper)]
    if (star.params[star.name]['numax'] is not None and star.params[star.name]['numax'] <= 500.) or (star.nyquist is not None and star.nyquist <= 300.):
        star.boxes = np.logspace(np.log10(0.5), np.log10(25.), star.findex['n_trials'])*1.
    else:
        star.boxes = np.logspace(np.log10(50.), np.log10(500.), star.findex['n_trials'])*1.
    return star


def check_fitbg(star):
    """
    Checks if there is a starting value for numax as pySYD needs this information for the 
    second module (whether be it from the first module, CLI or saved to info/star_info.csv).

    Returns
    -------
    result : bool
        will return `True` if there is prior value for numax otherwise `False`.

    """
    # Check if numax was provided as input
    if star.params[star.name]['numax'] is None:
        # If not, checks if findex was run
        if glob.glob(os.path.join(star.params[star.name]['path'],'excess.csv')) != []:
            df = pd.read_csv(os.path.join(star.params[star.name]['path'],'excess.csv'))
            for col in ['numax', 'dnu', 'snr']:
                star.params[star.name][col] = df.loc[0, col]
        # Break if no numax is provided in any scenario
        else:
            print('# ERROR: the second module cannot run without any prior info for numax')
            return False
#    if star.verbose:
#        print('# Using numax=%.2f muHz as initial guess for second module'%star.params[star.name]['numax'])
    return True


def get_fitbg(star):
    """
    Gets initial guesses for granulation components (i.e. timescales and amplitudes) using
    solar scaling relations. This resets the power spectrum and has its own independent
    filter (i.e. [lower,upper] mask) to use for this subroutine.

    Parameters
    ----------
    star : target.Target
        pySYD target object
    star.oversample : bool
        if `True`, it will use an oversampled power spectrum for the first iteration or 'step'
    minimum_freq : float
        minimum frequency to use for the power spectrum if `None` is provided (via info/star_info.csv). Default = `10.0` muHz. Please note: this is typically sufficient for most stars but may affect evolved stars!
    maximum_freq : float
        maximum frequency to use for the power spectrum if `None` is provided (via info/star_info.csv). Default = `5000.0` muHz.

    Returns
    -------
    star : target.Target
        updated pySYD target object

    """
    star.frequency, star.power = np.copy(star.freq_os), np.copy(star.pow_os)
    star.resolution = star.frequency[1]-star.frequency[0]

    if star.params[star.name]['lower_bg'] is not None:
        lower = star.params[star.name]['lower_bg']
    else:
        lower = star.fitbg['lower_bg']
    if star.params[star.name]['upper_bg'] is not None:
        upper = star.params[star.name]['upper_bg']
    else:
        upper = star.fitbg['upper_bg']
    star.params[star.name]['bg_mask']=[lower,upper]

    # Mask power spectrum for fitbg module
    mask = np.ma.getmask(np.ma.masked_inside(star.frequency, star.params[star.name]['bg_mask'][0], star.params[star.name]['bg_mask'][1]))
    star.frequency, star.power = np.copy(star.frequency[mask]), np.copy(star.power[mask])
    star.random_pow = np.copy(star.power)
    # Get other relevant initial conditions
    star.i = 0
    star.fitbg['results'][star.name] = {'numax_smooth':[],'A_smooth':[],'numax_gauss':[],'A_gauss':[],
                                        'FWHM':[],'dnu':[],'white':[]}

    # Use scaling relations from sun to get starting points
    star = solar_scaling(star)

    # if lower numax adjust default smoothing filter from 2.5->0.5muHz
    # this needs to be fixed - it doesn't change for the rest
    if star.params[star.name]['numax'] <= 500.:
        star.fitbg['ind_width'] = 5.0
        star.globe['smooth_ps'] = 0.5
    else:
        star.fitbg['ind_width'] = 20.0
        star.globe['smooth_ps'] = 2.5

    return star


def solar_scaling(star, scaling='tau_sun_single', max_laws=3, times=1.5):
    """
    Uses scaling relations from the Sun to:
    1) estimate the width of the region of oscillations using numax
    2) guess starting values for granulation timescales

    Parameters
    ----------
    max_laws : int
        the maximum number of resolvable Harvey-like components

    """
    constants = Constants()
    # Use scaling relations to estimate width of oscillation region to mask out of the background fit
    width = constants.width_sun*(star.params[star.name]['numax']/constants.numax_sun)
    maxpower = [star.params[star.name]['numax']-(width*star.globe['width']), 
                star.params[star.name]['numax']+(width*star.globe['width'])]
    if star.params[star.name]['lower_ps'] is not None:
        maxpower[0] = star.params[star.name]['lower_ps']
    if star.params[star.name]['upper_ps'] is not None:
        maxpower[1] = star.params[star.name]['upper_ps']
    star.params[star.name]['ps_mask'] = [maxpower[0],maxpower[1]]

    # Use scaling relation for granulation timescales from the sun to get starting points
    scale = constants.numax_sun/star.params[star.name]['numax']
    if scaling == 'tau_sun_single':
        taus = np.array(constants.tau_sun_single)*scale
    else:
        taus = np.array(constants.tau_sun)*scale
    taus = taus[taus <= star.baseline]
    b = taus*10**-6.
    mnu = (1.0/taus)*10**5.
    star.b = b[mnu >= min(star.frequency)]
    star.mnu = mnu[mnu >= min(star.frequency)]
    if len(star.mnu)==0:
        star.b = b[mnu >= 10.] 
        star.mnu = mnu[mnu >= 10.]
    elif len(star.mnu) > max_laws:
        star.b = b[mnu >= min(star.frequency)][-max_laws:]
        star.mnu = mnu[mnu >= min(star.frequency)][-max_laws:]
    else:
        pass
    # Save copies for plotting after the analysis
    star.nlaws = len(star.mnu)
    star.nlaws_orig = len(star.mnu)
    star.mnu_orig = np.copy(star.mnu)
    star.b_orig = np.copy(star.b)

    return star


def save_file(star, formats=[">15.8f", ">18.10e"]):
    """
    Saves the corrected power spectrum, which is computed by subtracting
    the best-fit stellar background model from the power spectrum.

    Parameters
    ----------
    star : target.Target
        the pySYD pipeline target
    formats : List[str]
        2x1 list of formats to save arrays as
    star.params[star.name]['path'] : str
        path to save the background-corrected power spectrum
    star.frequency : ndarray
        frequency array
    star.bg_corr_sub : ndarray
        background-subtracted power spectrum

    """
    
    f_name=os.path.join(star.params[star.name]['path'],'bgcorr_ps.txt')
    with open(f_name, "w") as f:
        for x, y in zip(star.frequency, star.bg_corr):
            values = [x, y]
            text = '{:{}}'*len(values) + '\n'
            fmt = sum(zip(values, formats), ())
            f.write(text.format(*fmt))
    f.close()
    if star.verbose:
        print(' **background-corrected PS saved**')


def save_findex(star):
    """
    Save the results of the find excess routine into the save folder of the current star.

    Parameters
    ----------
    star : target.Target
        pipeline target with the results of the `find_excess` routine

    """
    best = star.findex['results'][star.name]['best']
    variables = ['star', 'numax', 'dnu', 'snr']
    results = [star.name, star.findex['results'][star.name][best]['numax'], star.findex['results'][star.name][best]['dnu'], star.findex['results'][star.name][best]['snr']]
    save_path = os.path.join(star.params[star.name]['path'],'excess.csv')
    ascii.write(np.array(results), save_path, names=variables, delimiter=',', overwrite=True)


def save_fitbg(star):
    """
    Saves the results of the `fit_background` module.

    Parameters
    ----------
    star : target.Target
        pipeline target with the results of the `fit_background` routine

    """
    df = pd.DataFrame(star.fitbg['results'][star.name])
    star.df = df.copy()
    new_df = pd.DataFrame(columns=['parameter', 'value', 'uncertainty'])
    for c, col in enumerate(df.columns.values.tolist()):
        new_df.loc[c, 'parameter'] = col
        new_df.loc[c, 'value'] = df.loc[0,col]
        if star.fitbg['mc_iter'] > 1:
            new_df.loc[c, 'uncertainty'] = mad_std(df[col].values)
        else:
            new_df.loc[c, 'uncertainty'] = '--'
    new_df.to_csv(os.path.join(star.params[star.name]['path'],'background.csv'), index=False)
    if star.fitbg['samples']:
        df.to_csv(os.path.join(star.params[star.name]['path'],'samples.csv'), index=False)


def verbose_output(star):
    """
    Print results of the `fit_background` routine if verbose is `True`.

    """
    note=''
    df = pd.read_csv(os.path.join(star.params[star.name]['path'],'background.csv'))
    params = get_params_dict()
    if star.fitbg['mc_iter'] > 1:
        note+='\nOutput parameters:'
        line='\n%s: %.2f +/- %.2f %s'
        for idx in df.index.values.tolist():
            note+=line%(df.loc[idx,'parameter'],df.loc[idx,'value'],df.loc[idx,'uncertainty'],params[df.loc[idx,'parameter']]['unit'])
    else:
        note+='----------------------------------------------------\nOutput parameters:'
        line='\n%s: %.2f %s'
        for idx in df.index.values.tolist():
            note+=line%(df.loc[idx,'parameter'],df.loc[idx,'value'],params[df.loc[idx,'parameter']]['unit'])
    note+='\n----------------------------------------------------'
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
    df.to_csv(os.path.join(args.params['outdir'],'excess.csv'), index=False)

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
    df.to_csv(os.path.join(args.params['outdir'],'background.csv'), index=False)


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
            'numax_smooth':{'unit':'muHz','label':r'$\rm Smooth \,\, \nu_{max} \,\, [\mu Hz]$','latex':{'label':'$\\nu_{\\mathrm{max}}$', 'format':'%.2f', 'unit':'$\\rm \\mu Hz$'}}, 
            'A_smooth':{'unit':'ppm^2/muHz','label':r'$\rm Smooth \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\\rm A_{osc}$', 'format':'%.2f', 'unit':'$\\rm ppm^{2} \\mu Hz^{-1}$'}}, 
            'numax_gauss':{'unit':'muHz','label':r'$\rm Gauss \,\, \nu_{max} \,\, [\mu Hz]$','latex':{'label':'$\\nu_{\\mathrm{max}}$', 'format':'%.2f', 'unit':'$\\rm \\mu Hz$'}}, 
            'A_gauss':{'unit':'ppm^2/muHz','label':r'$\rm Gauss \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'$\\rm A_{osc}$', 'format':'%.2f', 'unit':'$\\rm ppm^{2} \\mu Hz^{-1}$'}}, 
            'FWHM':{'unit':'muHz','label':r'$\rm Gauss \,\, FWHM \,\, [\mu Hz]$','latex':{'label':'FWHM', 'format':'%.2f', 'unit':'$\\rm \\mu Hz$'}}, 
            'dnu':{'unit':'muHz','label':r'$\rm \Delta\nu \,\, [\mu Hz]$','latex':{'label':'$\\Delta\\nu$', 'format':'%.2f', 'unit':'$\\rm \\mu Hz$'}},
            'white':{'unit':'ppm^2/muHz','label':r'$\rm White \,\, [ppm^{2} \mu Hz^{-1}]$','latex':{'label':'White noise', 'format':'%.2f', 'unit':'$\\rm ppm^{2} \\mu Hz^{-1}$'}}, 
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


def get_data_columns(type=['csv','required','table']):

    if type == 'csv':
        columns = ['stars','radius','radius_err','teff','teff_err','logg','logg_err','lower_ex','upper_ex','lower_bg','upper_bg','lower_ps','upper_ps','lower_ech','upper_ech','numax','dnu','seed']
    elif type == 'required':
        columns = ['radius','logg','teff','numax','lower_ex','upper_ex','lower_bg','upper_bg','lower_ps','upper_ps','lower_ech','upper_ech','seed']
    elif type == 'table':
        columns = ['numax_smooth', 'A_smooth', 'FWHM', 'dnu', 'white']
    else:
        print("Did not understand that input, please specify type. Choose from: ['full','required','table']")
        columns = []

    return columns


def load_single(args, params={}):
    """
    Loads information for a single star.

    Parameters
    ----------
    args : argparse.Namespace
        command line arguments

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments

    """
    assert len(args.stars) == 1, "You can only load in data for one star at a time."

    args.star = args.stars[0]
    print(args.stars, args.star)
    check_input_args(args)
    params.update({'inpdir':args.inpdir,'outdir':args.outdir,'of_new':args.of_new,'show':args.show,
                   'of_actual':args.of_actual,'info':args.info,'save':args.save,'kepcorr':args.kepcorr,})

    params['path'] = os.path.join(args.outdir, args.star)
    args.params = params
    # Get star info
    args = get_single_info(args)
    return args


def get_single_info(args):
    """
    Reads in any star information provided via args.info and is 'info/star_info.csv' by default. 
    ** Please note that this is NOT required for pySYD to run successfully **

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    columns : list
        the list of columns to provide stellar information for

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments

    """
    constants = Constants()
    columns = get_data_columns(type='required')
    # Open file if it exists
    if os.path.exists(args.info):
        df = pd.read_csv(args.info)
        stars = [str(each) for each in df.stars.values.tolist()]
        if args.star in stars:
            idx = stars.index(args.star)
            for column in columns:
                if not np.isnan(float(df.loc[idx,column])):
                    args.params[column] = float(df.loc[idx,column])
                else:
                    args.params[column] = None
        else:
            for column in columns:
                args.params[column] = None
        if args.params['numax'] is not None:
            args.params['dnu'] = 0.22*(args.params['numax']**0.797)
        else:
            if args.params['radius'] is not None and args.params['logg'] is not None:
                args.params['mass'] = ((((args.params['radius']*constants.r_sun)**(2.0))*10**(args.params['logg'])/constants.G)/constants.m_sun)
                args.params['numax'] = constants.numax_sun*args.params['mass']*(args.params['radius']**(-2.0))*((args.params['teff']/constants.teff_sun)**(-0.5))
                args.params['dnu'] = constants.dnu_sun*(args.params['mass']**(0.5))*(args.params['radius']**(-1.5))
    override={'lower_bg':args.lower_bg,'upper_bg':args.upper_bg,'lower_ex':args.lower_ex,
              'upper_ex':args.upper_ex,'lower_ps':args.lower_ps,'upper_ps':args.upper_ps,
              'lower_ech':args.lower_ech,'upper_ech':args.upper_ech,'dnu':args.dnu,'numax':args.numax}
    for each in override:
        if override[each] is not None:
            # if numax is provided via CLI, findex is skipped
            if each == 'numax':
                args.params['numax'] = override[each][0]
                args.params['dnu'] = 0.22*(args.params['numax']**0.797)
            # if dnu is provided via CLI, this value is used instead of the derived dnu
            elif each == 'dnu':
                args.params['force'] = True
                args.params['guess'] = override[each][0]
            else:
                args.params[each] = override[each][0]
                
    return args


class Constants:

    def __init__(self):
        """
        UNITS ARE IN THE SUPERIOR CGS 
        COME AT ME

        """
        # Solar values
        self.m_sun = 1.9891e33
        self.r_sun = 6.95508e10
        self.rho_sun = 1.41
        self.teff_sun = 5777.0
        self.logg_sun = 4.4
        self.teffred_sun = 8907.
        self.numax_sun = 3090.0
        self.dnu_sun = 135.1
        self.width_sun = 1300.0 
        self.tau_sun = [5.2e6,1.8e5,1.7e4,2.5e3,280.0,80.0] 
        self.tau_sun_single = [3.8e6,2.5e5,1.5e5,1.0e5,230.,70.]
        # Constants
        self.G = 6.67428e-8
        # Conversions
        self.cm2au = 6.68459e-14
        self.au2cm = 1.496e+13
        self.rad2deg = 180./np.pi
        self.deg2rad = np.pi/180.