import os
import pdb
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.convolution import (
    Box1DKernel, Gaussian1DKernel, convolve, convolve_fft
)
from scipy.ndimage import filters
from scipy.special import erf


##########################################################################################
#                                                                                        #
#                                      DICTIONARIES                                      #
#                                                                                        #
##########################################################################################


def get_info(args, star_info='Files/star_info.csv', params={}):
    """Loads todo.txt, sets up file paths and initialises other information.

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
    with open('Files/todo.txt', "r") as f:
        todo = np.array([int(float(line.strip().split()[0])) for line in f.readlines()])
    params['path'] = 'Files/data/'
    # Adding constants and the target list
    params.update({
        'numax_sun': 3090.0, 'dnu_sun': 135.1, 'width_sun': 1300.0, 'todo': todo, 'G': 6.67428e-8,
        'tau_sun': [5.2e6, 1.8e5, 1.7e4, 2.5e3, 280.0, 80.0], 'teff_sun': 5777.0, 'mass_sun': 1.9891e33,
        'tau_sun_single': [3.8e6, 2.5e5, 1.5e5, 1.0e5, 230., 70.], 'radius_sun': 6.95508e10
    })
    # Set file paths
    for target in todo:
        params[target] = {}
        params[target]['path'] = '/'.join(params['path'].split('/')[:-2]) + '/results/%d/' % target
    args.params = params

    # Initialise parameters for the find excess routine
    if args.excess:
        args = get_excess_params(args)
    else:
        args = get_excess_params(args)
        args.findex['do'] = False
    # Initialise parameters for the fit background routine
    if args.background:
        args = get_bg_params(args)
    else:
        args = get_bg_params(args)
        args.fitbg['do'] = False
    # Get star info
    args = get_star_info(args, star_info)

    return args


def get_excess_params(
        args,
        findex={},
        save=True,
        step=0.25,
        binning=0.005,
        n_trials=3,
        lower=10.0,
        upper=4000.0
):
    """Get the parameters for the find excess routine.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    findex : dict
        the parameters of the find excess routine
    save : bool
        if true will save the results of the find excess routine. Default value is `True`.
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

    findex['do'] = True
    pars = {
        'save': save,
        'step': step,
        'binning': binning,
        'smooth_width': args.filter,
        'n_trials': n_trials,
        'lower': lower,
        'upper': upper,
    }
    findex.update(pars)

    # Initialise save folders
    if findex['save']:
        for target in args.params['todo']:
            if not os.path.exists(args.params[target]['path']):
                os.makedirs(args.params[target]['path'])
    args.findex = findex

    return args


def get_bg_params(
        args,
        fitbg={},
        save=True,
        box_filter=2.5,
        ind_width=50,
        n_rms=20,
        n_peaks=10,
        force=False,
        guess=140.24,
        clip=True,
        clip_value=0.0,
        ech_smooth=True,
        ech_filter=1.0,
        smooth_ps=1.0,
        lower_numax=None,
        upper_numax=None,
        lower=None,
        upper=None,
        slope=False,
        samples=True
):
    """Get the parameters for the fit background routine.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    fitbg : dict
        the parameters of the fit background routine
    save : bool
        if true will save the results of the background routine. Default value is `True`.
    box_filter : float
        the size of the box filter. Default value is `2.5`.
    ind_width : int
        the independent average smoothing width. Default value is `50`.
    n_rms : int
        TODO: Write description. Default value is `20`.
    n_peaks : int
        the number of peaks to select. Default value is `10`.
    force : bool
        if true will force dnu to be the equal to `guess`. Default value is `False`.
    guess : float
        the estimate of dnu. Default value is `140.24`.
    clip : bool
        if true will set the minimum frequency value of the echelle plot to `clip_value`. Default value is `True`.
    clip_value : float
        the minimum frequency of the echelle plot. Default value is `0.0`.
    ech_smooth : bool
        TODO: Write description. Default value is `True`.
    ech_filter : float
        TODO: Write description. Default value is `1.0`.
    smooth_ps : float
        TODO: Write description. Default value is `1.0`.
    lower_numax : Optional[float]
        the lower bound on numax. Default value is `None`.
    upper_numax : Optional[float]
        the upper bound on numax. Default value is `None`.
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

    fitbg['do'] = True
    pars = {
        'save': save,
        'num_mc_iter': args.mciter,
        'lower': lower,
        'upper': upper,
        'box_filter': box_filter,
        'ind_width': ind_width,
        'n_rms': n_rms,
        'n_peaks': n_peaks,
        'smooth_ps': smooth_ps,
        'lower_numax': lower_numax,
        'upper_numax': upper_numax,
        'force': force,
        'guess': guess,
        'clip': clip,
        'clip_value': clip_value,
        'ech_smooth': ech_smooth,
        'ech_filter': ech_filter,
        'slope': slope,
        'samples': samples,
    }
    fitbg.update(pars)

    # Harvey components
    fitbg['functions'] = {1: harvey_one, 2: harvey_two, 3: harvey_three, 4: harvey_four, 5: harvey_five, 6: harvey_six}

    # Initialise save folders
    if fitbg['save']:
        for target in args.params['todo']:
            if not os.path.exists(args.params[target]['path']):
                os.makedirs(args.params[target]['path'])
    args.fitbg = fitbg

    return args


def get_star_info(args, star_info, cols=['rad', 'logg', 'teff']):
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
    if os.path.isfile(star_info):
        df = pd.read_csv(star_info)
        targets = df.targets.values.tolist()
        for todo in args.params['todo']:
            if todo in targets:
                idx = targets.index(todo)
                # Update information from columns
                for col in cols:
                    args.params[todo][col] = df.loc[idx, col]
                # Add estimate of numax if the column exists
                if 'numax' in df.columns.values.tolist():
                    args.params[todo]['numax'] = df.loc[idx, 'numax']
                    args.params[todo]['dnu'] = 0.22*(df.loc[idx, 'numax']**0.797)
                # Otherwise estimate using other stellar parameters
                else:
                    args.params[todo]['mass'] = ((((args.params[todo]['rad']*args.params['radius_sun'])**(2.0))*10**(args.params[todo]['logg'])/args.params['G'])/args.params['mass_sun'])
                    args.params[todo]['numax'] = args.params['numax_sun']*args.params[todo]['mass']*(args.params[todo]['rad']**(-2.0))*((args.params[todo]['teff']/args.params['teff_sun'])**(-0.5))
                    args.params[todo]['dnu'] = args.params['dnu_sun']*(args.params[todo]['mass']**(0.5))*(args.params[todo]['rad']**(-1.5))
                # Add in the frequency bounds for the findex routine (lowerx, upperx) and
                # the frequency bounds for the fitbg routine (lowerb and upperb)
                # Note: this also loads in the random seed for Kepler targets that needed correction
                for col in ['lowerx', 'upperx', 'lowerb', 'upperb','seed']:
                    if np.isnan(df.loc[idx, col]):
                        args.params[todo][col] = None
                    else:
                        args.params[todo][col] = df.loc[idx, col]

    return args


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


def gaussian_bounds(x, y, best_x=None, sigma=None):
    """Get the bounds for the parameters of a Gaussian fit to the data.

    Parameters
    ----------
    x : np.ndarray
        the x values of the data
    y : np.ndarray
        the y values of the data
    best_x : Optional[float]
        TODO: Write description. Default value is `None`.
    sigma : Optional[float]
        TODO: Write description. Default value is `None`.

    Returns
    -------
    bb : List[Tuple]
        list of parameter bounds of a Gaussian fit to the data
    """

    if sigma is None:
        sigma = (max(x)-min(x))/8.0/np.sqrt(8.0*np.log(2.0))
    bb = []
    b = np.zeros((2, 4)).tolist()
    b[1][0] = np.inf
    b[1][1] = 2.0*np.max(y)
    if not int(np.max(y)):
        b[1][1] = np.inf
    # set bounds on center of Gaussian. If center is known, set to center+/-3sigma
    # commented out for now since sigma seems to small
    #if best_x is not None:
    #    b[0][2] = best_x - 3*sigma
    #    b[1][2] = best_x + 3*sigma
    #else:
    # Set the whole range for center of Gaussian. This should be robust for most cases.
    b[0][2] = np.min(x)
    b[1][2] = np.max(x)
    
    b[0][3] = sigma
    b[1][3] = np.max(x)-np.min(x)
    bb.append(tuple(b))

    return bb


def max_elements(x, y, npeaks):
    """Get the first n peaks of the given data.

    Parameters
    ----------
    x : np.ndarray
        the x values of the data
    y : np.ndarray
        the y values of the data
    npeaks : int
        the first n peaks

    Returns
    -------
    peaks_x : np.ndarray
        the x co-ordinates of the first `npeaks`
    peaks_y : np.ndarray
        the y co-ordinates of the first `npeaks`
    """

    s = np.argsort(y)
    peaks_y = y[s][-int(npeaks):][::-1]
    peaks_x = x[s][-int(npeaks):][::-1]

    return peaks_x, peaks_y


def return_max(array, index=False, dnu=False, exp_dnu=None):
    """Return the either the value of peak or the index of the peak corresponding to the most likely dnu given a prior estimate,
    otherwise just the maximum value.

    Parameters
    ----------
    array : np.ndarray
        the data series
    index : bool
        if true will return the index of the peak instead otherwise it will return the value. Default value is `False`.
    dnu : bool
        if true will choose the peak closest to the expected dnu `exp_dnu`. Default value is `False`.
    exp_dnu : Optional[float]
        the expected dnu. Default value is `None`.

    Returns
    -------
    result : Union[int, float]
        if `index` is `True`, result will be the index of the peak otherwise if `index` is `False` it will instead return the
        value of the peak.
    """
    if dnu and exp_dnu is not None:
        lst = list(np.absolute(np.copy(array)-exp_dnu))
        idx = lst.index(min(lst))
    else:
        lst = list(array)
        idx = lst.index(max(lst))
    if index:
        return idx
    else:
        return lst[idx]

##########################################################################################
#                                                                                        #
#                                   DIFFERENT MODELS                                     #
#                                                                                        #
##########################################################################################


def power_law(frequency, pars, compare=False, power=None, error=None):
    """Power law. TODO: Write description.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    pars : list
        the parameters of the power law
    compare : bool
        if true will compare the power law model with the actual power spectrum. Default value is `False`.
    power : Optional[np.ndarray]
        the power of the power spectrum. Default value is `None`.
    error : Optional[np.ndarray]
        the error of the power spectrum. Default value is `None`.

    Returns
    -------
    result : np.ndarray
        if `compare` is `True`, `result` will be the comparison between the power and the power law model. Otherwise the power law
        model will be returned instead.
    """

    model = np.array([pars[0]/(f**pars[1]) for f in frequency])
    if compare and power is not None and error is not None:
        return (power - model)/error
    else:
        return model


def lorentzian(frequency, pars, compare=False, power=None, error=None):
    """Lorentzian. TODO: Write description.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    pars : list
        the parameters of the Lorentzian
    compare : bool
        if true will compare the Lorentzian model with the actual power spectrum. Default value is `False`.
        TODO: Currently not supported!
    power : Optional[np.ndarray]
        the power of the power spectrum. Default value is `None`.
    error : Optional[np.ndarray]
        the error of the power spectrum. Default value is `None`.

    Returns
    -------
    model : np.ndarray
        the Lorentzian model
    """

    model = np.array([pars])

    return model


def harvey(frequency, pars, mode='regular', gaussian=False, total=False):
    """Harvey model of the stellar granulation background of a target. TODO: Write description.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    pars : list
        the parameters of the Harvey model
    mode : str
        the mode of the Harvey model.
        The regular mode means both the second order and fourth order terms are added.
        The second mode means only the second order terms are added.
        The fourth mode means only the fourth order terms are added.
        Default value is `'regular'`.
    gaussian : bool
        if true will add a Gaussian skew. Default value is `False`.
    total : bool
        TODO: Write description. Default value is `False`.

    Returns
    -------
    model : np.ndarray
        the Harvey model
    """

    if gaussian:
        nlaws = int((len(pars)-6)/2.0)
    else:
        nlaws = int((len(pars)-1)/2.0)
    model = np.zeros_like(frequency)

    if mode == 'regular':
        for i in range(nlaws):
            model += pars[i*2]/(1.0+(pars[(i*2)+1]*frequency)**2.0+(pars[(i*2)+1]*frequency)**4.0)
    elif mode == 'second':
        for i in range(nlaws):
            model += pars[i*2]/(1.0+(pars[(i*2)+1]*frequency)**2.0)
    elif mode == 'fourth':
        for i in range(nlaws):
            model += pars[i*2]/(1.0+(pars[(i*2)+1]*frequency)**4.0)
    else:
        print('Wrong mode input for the harvey model function.')

    if gaussian:
        model += gaussian_skew(frequency, pars[2*nlaws+1:])
    if total:
        model += pars[2*nlaws]
    return model


def generate_model(frequency, pars, pars_errs, nyquist):
    """TODO: Write description.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    pars : np.ndarray
        model parameters
    pars_errs : np.ndarray
        TODO: Write description
    nyquist : float
        the Nyquist frequency

    Returns
    -------
    TODO: Write return argument
    """

    ps = np.zeros_like(frequency)

    for i, f in enumerate(frequency):

        r = (np.sin((np.pi*f)/(2.0*nyquist))/((np.pi*f)/(2.0*nyquist)))**2
        paras = [p+np.random.randn()*p_e for p, p_e in zip(pars, pars_errs)]
        nlaws = int((len(paras)-1.0)/2.0)
        m = 0
        for j in range(nlaws):
            m += paras[j*2]/(1.0+(paras[(j*2)+1]*f)**2.0+(paras[(j*2)+1]*f)**4.0)
        m *= r
        m += pars[-1] + np.random.random_integers(-1, 1)*(pars[-1]/2.0)**(np.random.randn()-1.0)
        if m < 0.:
            m = (10**(np.random.randn()))*r
        ps[i] = m

    return list(ps)


def gaussian(frequency, offset, amplitude, center, width):
    """The Gaussian function.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    offset : float
        the vertical offset
    amplitude : float
        the amplitude
    center : float
        the center
    width : float
        the width

    Returns
    -------
    result : np.ndarray
        the Gaussian function
    """

    return offset + amplitude*np.exp(-(center-frequency)**2.0/(2.0*width**2))


def harvey_one(frequency, a1, b1, white_noise):
    """The first Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the first Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += white_noise

    return model


def harvey_two(frequency, a1, b1, a2, b2, white_noise):
    """The second Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    a2 : float
        TODO: Write description
    b2 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the second Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += a2/(1.0+(b2*frequency)**2.0+(b2*frequency)**4.0)
    model += white_noise

    return model


def harvey_three(frequency, a1, b1, a2, b2, a3, b3, white_noise):
    """The third Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    a2 : float
        TODO: Write description
    b2 : float
        TODO: Write description
    a3 : float
        TODO: Write description
    b3 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the third Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += a2/(1.0+(b2*frequency)**2.0+(b2*frequency)**4.0)
    model += a3/(1.0+(b3*frequency)**2.0+(b3*frequency)**4.0)
    model += white_noise

    return model


def harvey_four(frequency, a1, b1, a2, b2, a3, b3, a4, b4, white_noise):
    """The fourth Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    a2 : float
        TODO: Write description
    b2 : float
        TODO: Write description
    a3 : float
        TODO: Write description
    b3 : float
        TODO: Write description
    a4 : float
        TODO: Write description
    b4 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the fourth Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += a2/(1.0+(b2*frequency)**2.0+(b2*frequency)**4.0)
    model += a3/(1.0+(b3*frequency)**2.0+(b3*frequency)**4.0)
    model += a4/(1.0+(b4*frequency)**2.0+(b4*frequency)**4.0)
    model += white_noise

    return model


def harvey_five(frequency, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, white_noise):
    """The fifth Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    a2 : float
        TODO: Write description
    b2 : float
        TODO: Write description
    a3 : float
        TODO: Write description
    b3 : float
        TODO: Write description
    a4 : float
        TODO: Write description
    b4 : float
        TODO: Write description
    a5 : float
        TODO: Write description
    b5 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the fifth Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += a2/(1.0+(b2*frequency)**2.0+(b2*frequency)**4.0)
    model += a3/(1.0+(b3*frequency)**2.0+(b3*frequency)**4.0)
    model += a4/(1.0+(b4*frequency)**2.0+(b4*frequency)**4.0)
    model += a5/(1.0+(b5*frequency)**2.0+(b5*frequency)**4.0)
    model += white_noise

    return model


def harvey_six(frequency, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, white_noise):
    """The sixth Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    a2 : float
        TODO: Write description
    b2 : float
        TODO: Write description
    a3 : float
        TODO: Write description
    b3 : float
        TODO: Write description
    a4 : float
        TODO: Write description
    b4 : float
        TODO: Write description
    a5 : float
        TODO: Write description
    b5 : float
        TODO: Write description
    a6 : float
        TODO: Write description
    b6 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the sixth Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += a2/(1.0+(b2*frequency)**2.0+(b2*frequency)**4.0)
    model += a3/(1.0+(b3*frequency)**2.0+(b3*frequency)**4.0)
    model += a4/(1.0+(b4*frequency)**2.0+(b4*frequency)**4.0)
    model += a5/(1.0+(b5*frequency)**2.0+(b5*frequency)**4.0)
    model += a6/(1.0+(b6*frequency)**2.0+(b6*frequency)**4.0)
    model += white_noise

    return model


def gaussian_skew(frequency, pars):
    """Computes a Gaussian skew. TODO: Write description.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    pars : list
        the skew parameters
    """

    # TODO: `x` is used here but is never initialised/defined
    model = np.array([2.0*gaussian(f, pars[0:4])*0.5*(1.0+erf(pars[4]*((x-pars[1])/pars[2])/np.sqrt(2.0))) for f in frequency])

    return model


##########################################################################################
#                                                                                        #
#                                DATA MANIPULATION ROUTINES                              #
#                                                                                        #
##########################################################################################


def mean_smooth_ind(x, y, width):
    """Smooths the data using independent mean smoothing and binning.

    Parameters
    ----------
    x : np.ndarray
        the x values of the data
    y : np.ndarray
        the y values of the data
    width : float
        independent average smoothing width

    Returns
    -------
    sx : np.ndarray
        binned mean smoothed x data
    sy : np.ndarray
        binned mean smoothed y data
    se : np.ndarray
        standard error
    """

    step = width-1
    sx = np.zeros_like(x)
    sy = np.zeros_like(x)
    se = np.zeros_like(x)

    j = 0

    while (j+step < len(x)-1):

        sx[j] = np.mean(x[j:j+step])
        sy[j] = np.mean(y[j:j+step])
        se[j] = np.std(y[j:j+step])/np.sqrt(width)
        j += step

    sx = sx[(sx != 0.0)]
    se = se[(sy != 0.0)]
    sy = sy[(sy != 0.0)]
    se[(se == 0.0)] = np.median(se)

    return sx, sy, se


def bin_data(x, y, params):
    """Bins data logarithmically.

    Parameters
    ----------
    x : np.ndarray
        the x values of the data
    y : np.ndarray
        the y values of the data
    params : list
        binning parameters

    Returns
    -------
    bin_freq : np.ndarray
        binned frequencies
    bin_pow : np.ndarray
        binned power
    """

    mi = np.log10(min(x))
    ma = np.log10(max(x))
    no = np.int(np.ceil((ma-mi)/params['binning']))
    bins = np.logspace(mi, mi+(no+1)*params['binning'], no)

    digitized = np.digitize(x, bins)
    bin_freq = np.array([x[digitized == i].mean() for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
    bin_pow = np.array([y[digitized == i].mean() for i in range(1, len(bins)) if len(y[digitized == i]) > 0])

    return bin_freq, bin_pow


def smooth(array, width, params, method='box', mode=None, fft=False, silent=False):
    """Smooths using a variety of methods. TODO: Write description.

    Parameters
    ----------
    array : np.ndarray
        the data
    TODO: Add parameters

    Returns
    -------
    TODO: Add return arguments
    """

    if method == 'box':

        if isinstance(width, int):
            kernel = Box1DKernel(width)
        else:
            width = int(np.ceil(width/params['resolution']))
            kernel = Box1DKernel(width)

        if fft:
            smoothed_array = convolve_fft(array, kernel)
        else:
            smoothed_array = convolve(array, kernel)

        if not silent:
            print('%s kernel: kernel size = %.2f muHz' % (method, width*params['resolution']))

    elif method == 'gaussian':

        n = 2*len(array)
        forward = array[:].tolist()
        reverse = array[::-1].tolist()

        if n % 4 != 0:
            start = int(np.ceil(n/4))
        else:
            start = int(n/4)
        end = len(array)

        final = np.array(reverse[start:end]+forward[:]+reverse[:start])

        if isinstance(width, int):
            kernel = Gaussian1DKernel(width)
        else:
            width = int(np.ceil(width/params['resolution']))
            kernel = Gaussian1DKernel(width, mode=mode)

        if fft:
            smoothed = convolve_fft(final, kernel)
        else:
            smoothed = convolve(final, kernel)

        smoothed_array = smoothed[int(n/4):int(3*n/4)]

        if not silent:
            print('%s kernel: sigma = %.2f muHz' % (method, width*params['resolution']))
    else:
        print('Do not understand the smoothing method chosen.')

    return smoothed_array


def max_elements(array, N, resolution, limit=[False, None]):
    """Returns the indices of the maximum elements. TODO: Write description.

    Parameters
    ----------
    TODO: Add parameters

    Returns
    -------
    TODO: Add return arguments
    """

    indices = []

    while len(indices) < N:

        new_max = max(array)
        idx = array.index(new_max)
        add = True
        if indices != [] and limit[0]:
            for index in indices:
                if np.absolute((index-idx)*resolution) < limit[1]:
                    add = False
                    break
        if add:
            indices.append(idx)
        array[idx] = 0.0

    return np.array(indices)


def smooth_gauss(array, fwhm, params, silent=False):
    """TODO: Write description.

    Parameters
    ----------
    TODO: Add parameters

    Returns
    -------
    TODO: Add return arguments
    """

    sigma = fwhm/np.sqrt(8.0*np.log(2.0))

    n = 2*len(array)
    N = np.arange(1, n+1, 1)
    mu = len(array)
    total = np.sum((1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-0.5*(((N-mu)/sigma)**2.0)))
    weights = ((1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-0.5*(((N-mu)/sigma)**2.0)))/total

    forward = array[:]
    reverse = array[::-1]

    if n % 4 != 0:
        start = int(np.ceil(n/4))
    else:
        start = int(n/4)
    end = int(n/2)

    final = np.array(reverse[start:end]+forward[:]+reverse[:start])
    fft = np.fft.irfft(np.fft.rfft(final)*np.fft.rfft(weights))
    dq = deque(fft)
    dq.rotate(int(n/2))
    smoothed = np.array(dq)
    smoothed_array = smoothed[int(n/4):int(3*n/4)]
    if not silent:
        print('gaussian kernel using ffts: sigma = %.2f muHz' % (sigma*params['resolution']))
    if params['edge'][0]:
        smoothed_array = smoothed_array[:-params['edge'][1]]

    return np.array(smoothed_array)


def corr(frequency, power, params):
    """Computes the auto-correlation function. TODO: Write description.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    power : np.ndarray
        the power of the power spectrum
    params : list
        the pipeline parameters

    Returns
    -------
    lag : np.ndarray
        the frequency lag
    auto : np.ndarray
        the auto-correlation
    """

    f = frequency[:]
    p = power[:]

    n = len(p)
    mean = np.mean(p)
    var = np.var(p)
    N = np.arange(n)

    lag = N*params['resolution']

    auto = np.correlate(p - mean, p - mean, "full")
    auto = auto[int(auto.size/2):]

    mask = np.ma.getmask(np.ma.masked_inside(lag, params['fitbg']['lower_lag'], params['fitbg']['upper_lag']))

    lag = lag[mask]
    auto = auto[mask]

    return lag, auto


def delta_nu(numax):
    """Estimates dnu using numax scaling relation.

    Parameters
    ----------
    numax : float
        the estimated numax

    Returns
    -------
    dnu : float
        the estimated dnu
    """

    return 0.22*(numax**0.797)
