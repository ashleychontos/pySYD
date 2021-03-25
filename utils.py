import glob
import numpy as np
import pandas as pd
from itertools import chain
from astropy.io import ascii
from astropy.stats import mad_std

from functions import *
from models import *


def get_info(args, params={}):
    """Loads todo.txt, sets up file paths and initializes other information.

    Parameters
    ----------
    args : argparse.Namespace
        command line arguments
    star_info : str
        the file path to the star_info.csv file containing target information. Default value is `'Files/star_info.csv'`
    params : dict
        the pipeline parameters

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    """

    # Open target list
    if args.target is None:
        with open(args.file, "r") as f:
            args.target = [int(float(line.strip().split()[0])) for line in f.readlines()]
    check_inputs(args)
    params['path'] = 'Files/data/'
    # Adding constants and the target list
    params.update({
        'numax_sun': 3090.0, 'dnu_sun': 135.1, 'width_sun': 1300.0, 'todo': args.target, 'G': 6.67428e-8, 'show': args.show,
        'tau_sun': [5.2e6, 1.8e5, 1.7e4, 2.5e3, 280.0, 80.0], 'teff_sun': 5777.0, 'mass_sun': 1.9891e33, 'save': args.save,
        'tau_sun_single': [3.8e6, 2.5e5, 1.5e5, 1.0e5, 230., 70.], 'radius_sun': 6.95508e10, 'keplercorr': args.keplercorr,
    })
    # Set file paths
    for target in args.target:
        params[target] = {}
        params[target]['path'] = '/'.join(params['path'].split('/')[:-2]) + '/results/%d/' % target
    args.params = params

    # Initialise parameters for the find excess routine
    args = get_excess_params(args)
    # Initialise parameters for the fit background routine
    args = get_background_params(args)
    # Get star info
    args = get_star_info(args)

    return args


def check_inputs(args):
    """ Make sure the command line inputs are proper lengths based on the specified targets."""

    checks={'lowerb':args.lowerb,'upperb':args.upperb,'lowerx':args.lowerx,
            'upperx':args.upperx,'dnu':args.dnu,'numax':args.numax}
    for check in checks:
        if checks[check] is not None:
            assert len(args.target) == len(checks[check]), "The number of values provided for %s does not equal the number of targets"%check


def get_excess_params(
        args,
        findex={},
):
    """Get the parameters for the find excess routine.

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

    pars = {
        'step': args.step,
        'binning': args.binning,
        'smooth_width': args.smooth_width,
        'n_trials': args.ntrials,
        'lower': 10.,
        'upper': 4000.,
    }
    findex.update(pars)

    # Initialise save folders
    if args.save:
        for target in args.params['todo']:
            if not os.path.exists(args.params[target]['path']):
                os.makedirs(args.params[target]['path'])
    args.findex = findex

    return args


def get_background_params(
        args,
        fitbg={},
):
    """Get the parameters for the fit background routine.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    fitbg : dict
        the parameters of the fit background routine
    box_filter : float
        the size of the box filter. Default value is `2.5`.
    ind_width : int
        the independent average smoothing width. Default value is `50`.
    n_rms : int
        TODO: Write description. Default value is `20`.
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
    lower : Optional[float]
        the lower frequency bound. Default value is `None`.
    upper : Optional[float]
        the upper frequency bound. Default value is `None`.
    slope : bool
        if true will correct for edge effects and residual slope in Gaussian fit. Default value is `False`.
    samples : bool
        TODO: Write description. Default value is `True`.

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    """

    pars = {
        'num_mc_iter': args.mciter,
        'box_filter': args.box_filter,
        'ind_width': args.ind_width,
        'n_rms': args.nrms,
        'n_peaks': args.npeaks,
        'smooth_ps': args.smooth_ps,
        'clip': args.clip,
        'clip_value': args.value,
        'smooth_ech': args.smooth_ech,
        'slope': args.slope,
        'samples': args.samples,
    }
    fitbg.update(pars)

    # Harvey components
    fitbg['functions'] = {1: harvey_one, 2: harvey_two, 3: harvey_three, 4: harvey_four, 5: harvey_five, 6: harvey_six}

    # Initialise save folders
    if args.save:
        for target in args.params['todo']:
            if not os.path.exists(args.params[target]['path']):
                os.makedirs(args.params[target]['path'])
    args.fitbg = fitbg

    return args


def get_star_info(args, cols=['rad','logg','teff','numax','lowerx','upperx','lowerb','upperb','seed']):
    """Get target information stored in `star_info.csv`.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    star_info : str
        the file path to the file `star_info.csv`
    cols : list
        the list of columns to extract information from. Default value is `['rad', 'logg', 'teff']`.

    Returns
    -------
    args : argparse.Namespace
        the updated command line arguments
    """

    # Open file if it exists
    if os.path.exists(args.info):
        df = pd.read_csv(args.info)
        targets = df.targets.values.tolist()
        for i, todo in enumerate(args.params['todo']):
            args.params[todo]['excess'] = args.excess
            args.params[todo]['background'] = args.background
            args.params[todo]['force'] = False
            if todo in targets:
                idx = targets.index(todo)
                # Update information from columns
                for col in cols:
                    if not np.isnan(float(df.loc[idx,col])):
                        args.params[todo][col] = float(df.loc[idx, col])
                    else:
                        args.params[todo][col] = None
                # Add estimate of numax if the column exists
                if args.params[todo]['numax'] is not None:
                    args.params[todo]['dnu'] = 0.22*(args.params[todo]['numax']**0.797)
                # Otherwise estimate using other stellar parameters
                else:
                    if args.params[todo]['rad'] is not None and args.params[todo]['logg'] is not None:
                        args.params[todo]['mass'] = ((((args.params[todo]['rad']*args.params['radius_sun'])**(2.0))*10**(args.params[todo]['logg'])/args.params['G'])/args.params['mass_sun'])
                        args.params[todo]['numax'] = args.params['numax_sun']*args.params[todo]['mass']*(args.params[todo]['rad']**(-2.0))*((args.params[todo]['teff']/args.params['teff_sun'])**(-0.5))
                        args.params[todo]['dnu'] = args.params['dnu_sun']*(args.params[todo]['mass']**(0.5))*(args.params[todo]['rad']**(-1.5))
            override={'lowerb':args.lowerb,'upperb':args.upperb,'lowerx':args.lowerx,
                      'upperx':args.upperx,'dnu':args.dnu,'numax':args.numax}
            for each in override:
                if override[each] is not None:
                    # if numax is provided via CLI, findex is skipped
                    if each == 'numax':
                        args.params[todo]['excess'] = False
                        args.params[todo]['numax'] = override[each][i]
                        args.params[todo]['dnu'] = 0.22*(args.params[todo]['numax']**0.797)
                    # if dnu is provided via CLI, this value is used instead of the derived dnu
                    elif each == 'dnu':
                        args.params[todo]['force'] = True
                        args.params[todo]['guess'] = override[each][i]
                    else:
                        args.params[todo][each] = override[each][i]
                
    return args


def load_data(target, data=None):
    """Loads light curve and power spectrum data for the current target.

    Returns
    -------
    success : bool
        will return `True` if both the light curve and power spectrum data files exist otherwise `False`
    """

    # Now done at beginning to make sure it only does this one per target
    if glob.glob(target.params['path']+'%d_*' % target.target) != []:
        if target.verbose:
            print('')
            print('-------------------------------------------------')
            print('Target: %d' % target.target)
            print('-------------------------------------------------')
        # Load light curve
        if not os.path.exists(target.params['path']+'%d_LC.txt' % target.target):
            if target.verbose:
                print('Error: %s%d_LC.txt not found' % (target.params['path'], target.target))
            return False
        else:
            target.time, target.flux = get_file(target.params['path'] + '%d_LC.txt' % target.target)
            target.cadence = int(np.nanmedian(np.diff(target.time)*24.0*60.0*60.0))
            target.nyquist = 10**6/(2.0*target.cadence)
            if target.verbose:
                print('# LIGHT CURVE: %d lines of data read' % len(target.time))
            if target.params[target.target]['numax'] > 500.:
                target.fitbg['smooth_ps'] = 2.5
        # Load power spectrum
        if not os.path.exists(target.params['path'] + '%d_PS.txt' % target.target):
            if target.verbose:
                print('Error: %s%d_PS.txt not found' % (target.params['path'], target.target))
            return False
        else:
            target.frequency, target.power = get_file(target.params['path'] + '%d_PS.txt' % target.target)
            if target.params['keplercorr']:
                target = remove_artefact(target)
                if target.verbose:
                    print('## Removing Kepler artefacts ##')
            if target.verbose:
                print('# POWER SPECTRUM: %d lines of data read' % len(target.frequency))
        target.oversample = int(round((1./((max(target.time)-min(target.time))*0.0864))/(target.frequency[1]-target.frequency[0])))
        target.resolution = (target.frequency[1]-target.frequency[0])*target.oversample

        if target.verbose:
            if target.oversample == 1:
                print('critically sampled')
            else:
                print('oversampled by a factor of %d' % target.oversample)
            print('time series cadence: %d seconds' % target.cadence)
            print('power spectrum resolution: %.6f muHz' % target.resolution)
        # Create critically sampled PS
        if target.oversample != 1:
            target.freq = np.copy(target.frequency)
            target.pow = np.copy(target.power)
            target.frequency = np.array(target.frequency[target.oversample-1::target.oversample])
            target.power = np.array(target.power[target.oversample-1::target.oversample])
        else:
            target.freq = np.copy(target.frequency)
            target.pow = np.copy(target.power)
            target.frequency = np.copy(target.frequency)
            target.power = np.copy(target.power)
        if target.params[target.target]['excess']:
            # Make a mask using the given frequency bounds for the find excess routine
            mask = np.ones_like(target.freq, dtype=bool)
            if target.params[target.target]['lowerx'] is not None:
                mask *= np.ma.getmask(np.ma.masked_greater_equal(target.freq, target.params[target.target]['lowerx']))
            else:
                mask *= np.ma.getmask(np.ma.masked_greater_equal(target.freq, target.findex['lower']))
            if target.params[target.target]['upperx'] is not None:
                mask *= np.ma.getmask(np.ma.masked_less_equal(target.freq, target.params[target.target]['upperx']))
            else:
                mask *= np.ma.getmask(np.ma.masked_less_equal(target.freq, target.findex['upper']))
                target.freq = target.freq[mask]
                target.pow = target.pow[mask]
                if target.params[target.target]['numax'] is not None:
                    if target.params[target.target]['numax'] <= 500.:
                        target.boxes = np.logspace(np.log10(0.5), np.log10(25.), target.findex['n_trials'])*1.
                    else:
                        target.boxes = np.logspace(np.log10(50.), np.log10(500.), target.findex['n_trials'])*1.
        data=True
    else:
        print('Error: data not found for target %d' % target.target)
    return data, target


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


def check_fitbg(target):
    """Check if there is prior knowledge about numax as SYD needs this information to work well
    (either from findex module or from star info csv).

    Returns
    -------
    result : bool
        will return `True` if there is prior value for numax otherwise `False`.
    """

    # Check whether output from findex module exists; 
    # if yes, let that override star info guesses
    if glob.glob('%sexcess.csv' % target.params[target.target]['path']) != []:
        df = pd.read_csv('%sexcess.csv' % target.params[target.target]['path'])
        for col in ['numax', 'dnu', 'snr']:
            target.params[target.target][col] = df.loc[0, col]
    # Break if no numax is provided in any scenario
    if target.params[target.target]['numax'] is None:
        print(
            'Error: SYDpy cannot run without any value for numax.'
        )
        return False
    else:
        return True


def get_initial_guesses(target):
    """Get initial guesses for the granulation background."""
    from functions import mean_smooth_ind

    # Mask power spectrum for fitbg module based on estimated/fitted numax
    mask = np.ones_like(target.frequency, dtype=bool)
    if target.params[target.target]['lowerb'] is not None:
        mask *= np.ma.getmask(np.ma.masked_greater_equal(target.frequency, target.params[target.target]['lowerb']))
        if target.params[target.target]['upperb'] is not None:
            mask *= np.ma.getmask(np.ma.masked_less_equal(target.frequency, target.params[target.target]['upperb']))
        else:
            mask *= np.ma.getmask(np.ma.masked_less_equal(target.frequency, target.nyquist))
    else:
        if target.params[target.target]['numax'] > 300.0:
            mask = np.ma.getmask(np.ma.masked_inside(target.frequency, 100.0, target.nyquist))
        else:
            mask = np.ma.getmask(np.ma.masked_inside(target.frequency, 1.0, target.nyquist))
    # if lower numax adjust default smoothing filter from 2.5->0.5muHz
    if target.params[target.target]['numax'] <= 500.:
        target.fitbg['smooth_ps'] = 0.5
    target.frequency = target.frequency[mask]
    target.power = target.power[mask]
    target.width = target.params['width_sun']*(target.params[target.target]['numax']/target.params['numax_sun'])
    target.times = target.width/target.params[target.target]['dnu']
    # Arbitrary snr cut for leaving region out of background fit
    if target.params[target.target]['snr'] < 2.0:
        target.maxpower = [
            target.params[target.target]['numax'] - target.width/2.0,
            target.params[target.target]['numax'] + target.width/2.0
        ]
    else:
        target.maxpower = [
            target.params[target.target]['numax'] - target.times*target.params[target.target]['dnu'],
            target.params[target.target]['numax'] + target.times*target.params[target.target]['dnu']
        ]
    # Bin power spectrum to estimate red noise components
    bin_freq, bin_pow, bin_err = mean_smooth_ind(target.frequency, target.power, target.fitbg['ind_width'])
    # Mask out region with power excess
    target.bin_freq = bin_freq[~((bin_freq > target.maxpower[0]) & (bin_freq < target.maxpower[1]))]
    target.bin_pow = bin_pow[~((bin_freq > target.maxpower[0]) & (bin_freq < target.maxpower[1]))]
    target.bin_err = bin_err[~((bin_freq > target.maxpower[0]) & (bin_freq < target.maxpower[1]))]

    # Adjust the lower frequency limit given numax
#    if target.params[target.target]['numax'] > 300.0:
#        target.frequency = target.frequency[target.frequency > 100.0]
#        target.power = target.power[target.frequency > 100.0]
#        target.fitbg['lower'] = 100.0
    # Use scaling relation from sun to get starting points
    scale = target.params['numax_sun']/((target.maxpower[1] + target.maxpower[0])/2.0)
    taus = np.array(target.params['tau_sun'])*scale
    b = 2.0*np.pi*(taus*1e-6)
    mnu = (1.0/taus)*1e5
    target.b = b[mnu >= min(target.frequency)]
    target.mnu = mnu[mnu >= min(target.frequency)]
    if len(target.mnu)==0:
        target.b = b[mnu >= 10] 
        target.mnu = mnu[mnu >= 10]
    target.nlaws = len(target.mnu)
    target.mnu_orig = np.copy(target.mnu)
    target.b_orig = np.copy(target.b)
    return target


def save_findex(target, best):
    """Save the results of the find excess routine into the save folder of the current target.

    Parameters
    ----------
    results : list
        the results of the find excess routine
    """

    variables = ['target', 'numax', 'dnu', 'snr']
    results = [target.target, target.findex['results'][best]['numax'], target.findex['results'][best]['dnu'], target.findex['results'][best]['snr']]
    save_path = '%sexcess.csv' % target.params[target.target]['path']
    ascii.write(np.array(results), save_path, names=variables, delimiter=',', overwrite=True)


def save_fitbg(target):
    """Save results of fit background routine"""
    df = pd.DataFrame(target.final_pars)
    target.df = df.copy()
    if target.fitbg['num_mc_iter'] > 1:
        for column in target.df.columns.values.tolist():
            target.df['%s_err' % column] = np.array([mad_std(target.df[column].values)]*len(target.df))
    new_df = pd.DataFrame(columns=['parameter', 'value', 'uncertainty'])
    for c, col in enumerate(df.columns.values.tolist()):
        new_df.loc[c, 'parameter'] = col
        new_df.loc[c, 'value'] = target.final_pars[col][0]
        if target.fitbg['num_mc_iter'] > 1:
            new_df.loc[c, 'uncertainty'] = mad_std(target.final_pars[col])
        else:
            new_df.loc[c, 'uncertainty'] = '--'
    new_df.to_csv('%sbackground.csv' % target.params[target.target]['path'], index=False)
    if target.fitbg['samples']:
        target.df.to_csv('%ssamples.csv' % target.params[target.target]['path'], index=False)


def verbose_output(target, sampling=False):

    if sampling:
        # Print results with uncertainties
        print('\nOutput parameters:')
        print('numax (smoothed): %.2f +/- %.2f muHz' % (target.final_pars['numax_smooth'][0], mad_std(target.final_pars['numax_smooth'])))
        print('maxamp (smoothed): %.2f +/- %.2f ppm^2/muHz' % (target.final_pars['amp_smooth'][0], mad_std(target.final_pars['amp_smooth'])))
        print('numax (gaussian): %.2f +/- %.2f muHz' % (target.final_pars['numax_gaussian'][0], mad_std(target.final_pars['numax_gaussian'])))
        print('maxamp (gaussian): %.2f +/- %.2f ppm^2/muHz' % (target.final_pars['amp_gaussian'][0], mad_std(target.final_pars['amp_gaussian'])))
        print('fwhm (gaussian): %.2f +/- %.2f muHz' % (target.final_pars['fwhm_gaussian'][0], mad_std(target.final_pars['fwhm_gaussian'])))
        print('dnu: %.2f +/- %.2f muHz' % (target.final_pars['dnu'][0], mad_std(target.final_pars['dnu'])))
    else:
        print('-------------------------------------------------')
        print('Output parameters:')
        # Print results with no errors
        print('numax (smoothed): %.2f muHz' % (target.final_pars['numax_smooth'][0]))
        print('maxamp (smoothed): %.2f ppm^2/muHz' % (target.final_pars['amp_smooth'][0]))
        print('numax (gaussian): %.2f muHz' % (target.final_pars['numax_gaussian'][0]))
        print('maxamp (gaussian): %.2f ppm^2/muHz' % (target.final_pars['amp_gaussian'][0]))
        print('fwhm (gaussian): %.2f muHz' % (target.final_pars['fwhm_gaussian'][0]))
        print('dnu: %.2f' % (target.final_pars['dnu'][0]))
    print('-------------------------------------------------')
    print()


def scrape_output(path = 'Files/results/**/'):
    """
    Grabs each individual target's results and concatenates results into a single csv in Files/ for each submodulel
    (i.e. findex.csv and globalpars.csv). This is automatically called at the end of the main SYD module.
    """

    # Findex outputs
    output = '%s*excess.csv'%path
    files = glob.glob(output)
    df = pd.read_csv(files[0])
    for i in range(1,len(files)):
        df_new = pd.read_csv(files[i])
        df = pd.concat([df, df_new])
    df.to_csv('Files/excess.csv', index=False)

    # Fitbg outputs
    output = '%s*background.csv'%path
    files = glob.glob(output)
    df = pd.DataFrame(columns=['target'])

    for i, file in enumerate(files):
	       df_new = pd.read_csv(file)
	       df_new.set_index('parameter',inplace=True,drop=False)
	       df.loc[i,'target']=file.strip().split('/')[-2]
	       new_header_names=[[i,i+'_err'] for i in df_new.index.values.tolist()] #add columns to get error
	       new_header_names=list(chain.from_iterable(new_header_names))          
	       for col in new_header_names:
		          if '_err' in col:
			             df.loc[i,col]=df_new.loc[col[:-4],'uncertainty']
		          else:
			             df.loc[i,col]=df_new.loc[col,'value']

    df.fillna('--', inplace=True)
    df.to_csv('Files/background.csv', index=False)


def set_seed(target):
    """For Kepler targets that require a correction via CLI (--kc), a random seed is generated
    from U~[1,10^6] and stored in stars_info.csv for reproducible results in later runs."""
    seed = list(np.random.randint(1,high=10000000,size=1))
    df = pd.read_csv('Files/star_info.csv')
    targets = df.targets.values.tolist()
    idx = targets.index(target.target)
    df.loc[idx,'seed'] = int(seed[0])
    target.params[target.target]['seed'] = seed[0]
    df.to_csv('Files/star_info.csv',index=False)
    return target
