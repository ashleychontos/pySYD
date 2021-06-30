import numpy as np
import pandas as pd
from collections import deque
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft



def set_seed(star, lower=1, upper=10**7, size=1):
    """
    For Kepler targets that require a correction via CLI (--kc), a random seed is generated
    from U~[1,10^7] and stored in stars_info.csv for reproducible results in later runs.

    Parameters
    ----------
    star : target.Target
        the pySYD pipeline object
    lower : int 
        lower limit for random seed value. Default value is `1`.
    upper : int
        upper limit for random seed value. Default value is `10**7`.
    size : int
        number of seed values returned. Default value is `1`.

    Returns
    -------
    star : target.Target
        the pySYD pipeline object
        
    """

    seed = list(np.random.randint(lower,high=upper,size=size))
    df = pd.read_csv(star.params['info'])
    stars = [str(each) for each in df.stars.values.tolist()]
    idx = stars.index(star.name)
    df.loc[idx,'seed'] = int(seed[0])
    star.params[star.name]['seed'] = seed[0]
    df.to_csv(star.params['info'],index=False)
    return star


def remove_artefact(star, lcp=1.0/(29.4244*60*1e-6), lf_lower=[240.0,500.0], lf_upper =[380.0,530.0], 
                    hf_lower = [4530.0,5011.0,5097.0,5575.0,7020.0,7440.0,7864.0],
                    hf_upper = [4534.0,5020.0,5099.0,5585.0,7030.0,7450.0,7867.0],):
    """
    Removes SC artefacts in Kepler power spectra by replacing them with noise (using linear interpolation)
    following a chi-squared distribution. 

    Known artefacts are:
    1) 1./LC harmonics
    2) high frequency artefacts (>5000 muHz)
    3) low frequency artefacts 250-400 muHz (mostly present in Q0 and Q3 data)

    Parameters
    ----------
    star : target.Target
        the pySYD pipeline object
    lcp : float
        long cadence period in Msec
    lf_lower : List[float]
        lower limit of low frequency artefact
    lf_upper : List[float]
        upper limit of low frequency artefact
    hf_lower : List[float]
        lower limit of high frequency artefact
    hf_upper : List[float]
        upper limit of high frequency artefact
    Returns
    -------
    star : target.Target
        the pySYD pipeline object
    """

    if star.params[star.name]['seed'] is None:
        star = set_seed(star)
    # LC period in Msec -> 1/LC ~muHz
    artefact = (1.0+np.arange(14))*lcp
    # Estimate white noise
    white = np.mean(star.power[(star.frequency >= star.nyquist-100.0)&(star.frequency <= star.nyquist-50.0)])

    np.random.seed(int(star.params[star.name]['seed']))
    # Routine 1: remove 1/LC artefacts by subtracting +/- 5 muHz given each artefact
    for i in range(len(artefact)):
        if artefact[i] < np.max(star.frequency):
            mask = np.ma.getmask(np.ma.masked_inside(star.frequency, artefact[i]-5.0*star.resolution, artefact[i]+5.0*star.resolution))
            if np.sum(mask) != 0:
                star.power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0

    np.random.seed(int(star.params[star.name]['seed']))
    # Routine 2: fix high frequency artefacts
    for lower, upper in zip(hf_lower, hf_upper):
        if lower < np.max(star.frequency):
            mask = np.ma.getmask(np.ma.masked_inside(star.frequency, lower, upper))
            if np.sum(mask) != 0:
                star.power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0

    np.random.seed(int(star.params[star.name]['seed']))
    # Routine 3: remove wider, low frequency artefacts 
    for lower, upper in zip(lf_lower, lf_upper):
        low = np.ma.getmask(np.ma.masked_outside(star.frequency, lower-20., lower))
        upp = np.ma.getmask(np.ma.masked_outside(star.frequency, upper, upper+20.))
        # Coeffs for linear fit
        m, b = np.polyfit(star.frequency[~(low*upp)], star.power[~(low*upp)], 1)
        mask = np.ma.getmask(np.ma.masked_inside(star.frequency, lower, upper))
        # Fill artefact frequencies with noise
        star.power[mask] = ((star.frequency[mask]*m)+b)*(np.random.chisquare(2, np.sum(mask))/2.0)

    return star


def whiten_mixed(star):
    """
    Generates random white noise in place of ell=1 for subgiants with mixed modes to better
    constrain the characteristic frequency spacing.

    Parameters
    ----------
    star : target.Target
        pySYD pipeline target
    star.frequency : np.ndarray
        the frequency of the power spectrum
    star.power : np.ndarray
        the power spectrum

    """
    if star.params[star.name]['seed'] is None:
        star = set_seed(star)
    # Estimate white noise
    white = np.mean(star.power[(star.frequency >= star.nyquist-100.0)&(star.frequency <= star.nyquist-50.0)])

    # Take the provided dnu and "fold" the power spectrum
    folded_freq = np.copy(star.frequency)%star.params[star.name]['dnu']
    mask = np.ma.getmask(np.ma.masked_inside(folded_freq, star.params[star.name]['ech_mask'][0], star.params[star.name]['ech_mask'][1]))
    np.random.seed(int(star.params[star.name]['seed']))
    # Routine 1: remove 1/LC artefacts by subtracting +/- 5 muHz given each artefact
    if np.sum(mask) != 0:
        star.power[mask] = white*np.random.chisquare(2,np.sum(mask))/2.0

    return star


def log_likelihood(observations, model):
    """
    Until we figure out a better method, we are computing the likelhood using
    the mean squared error.

    Parameters
    ----------
    observations : ndarray
        the observed power spectrum
    model : ndarray
        model generated at the observed frequencies

    Returns
    -------
    LL : float
        the natural logarithm of the likelihood (or the MSE)

    """

    return -0.5*(np.sum((observations-model)**2.))


def compute_aic(observations, model, n_parameters):
    """
    Computes the Akaike Information Criterion (AIC) given the modeled
    power spectrum.

    Parameters
    ----------
    observations : ndarray
        the observed power spectrum
    model : ndarray
        model generated at the observed frequencies
    n_parameters : int
        number of free parameters in the given model

    Returns
    -------
    aic : float
        AIC value

    """
    N = len(observations)
    LL = log_likelihood(observations, model)
    aic = (-2.*LL)/N + (2.*n_parameters)/N

    return aic


def compute_bic(observations, model, n_parameters):
    """
    Computes the Bayesian Information Criterion (BIC) given the modeled 
    power spectrum.

    Parameters
    ----------
    observations : ndarray
        the observed power spectrum
    model : ndarray
        model generated at the observed frequencies
    n_parameters : int
        number of free parameters in the given model

    Returns
    -------
    aic : float
        AIC value

    """
    N = len(observations)
    LL = log_likelihood(observations, model)
    bic = -2.*LL + np.log(N)*n_parameters

    return bic


def gaussian_bounds(x, y, guesses, best_x=None, sigma=None):
    """
    Get the bounds for the parameters of a Gaussian fit to the data.

    Parameters
    ----------
    x : np.ndarray
        the x values of the data
    y : np.ndarray
        the y values of the data
    guesses : List[float]
        initial guesses for a gaussian fit
    sigma : Optional[float]
        sigma from a standard Normal distribution 

    Returns
    -------
    bb : List[Tuple]
        list of parameter bounds of a Gaussian fit to the data
    """

    offset, amp, center, width = guesses
    if sigma is None:
        sigma = (max(x)-min(x))/8.0/np.sqrt(8.0*np.log(2.0))
    bb = []
    b = np.zeros((2, 4)).tolist()
    
    # offset bound:
    b[1][0] = np.inf            #upper bound
    # amplitude bounds:
    if amp > 0:
        b[1][1] = 2.0*np.max(y) #upper bound
    else:
        b[0][1]=-np.inf         #lower bound
        b[1][1]=0               #upper bound
    # center bounds:
    b[0][2] = np.min(x)
    b[1][2] = np.max(x)
    # width bounds
    b[0][3] = sigma
    b[1][3] = (np.max(x)-np.min(x))*2.
    bb.append(tuple(b))
    
    return bb


def max_elements(x, y, npeaks):
    """
    Get the first n peaks of the given data.

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


def return_max(x_array, y_array, exp_dnu=None, index=False):
    """
    Return the either the value of peak or the index of the peak corresponding to the most likely dnu given a prior estimate,
    otherwise just the maximum value.

    Parameters
    ----------
    x_array : np.ndarray
        the independent axis (i.e. time, frequency)
    y_array : np.ndarray
        the dependent axis
    method : str
        which method to use for determing the max elements in an array
    index : bool
        if true will return the index of the peak instead otherwise it will return the value. Default value is `False`.
    dnu : bool
        if true will choose the peak closest to the expected dnu `exp_dnu`. Default value is `False`.
    exp_dnu : Required[float]
        the expected dnu. Default value is `None`.

    Returns
    -------
    result : Union[int, float]
        if `index` is `True`, result will be the index of the peak otherwise if `index` is `False` it will instead return the
        value of the peak.

    """
    if exp_dnu is None:
        lst = list(y_array)
        idx = lst.index(max(lst))
        weights = None
    else:
        sig = 0.35*exp_dnu/2.35482 
        weights = 1./(sig*np.sqrt(2.*np.pi))*np.exp(-(x_array-exp_dnu)**2./(2.*sig**2))
        lst = list(weights*y_array)
        idx = lst.index(max(lst))
    if index:
        return idx
    else:
        return x_array[idx], y_array[idx]


def bin_data(x, y, width, log=False, mode=['mean', 'median', 'gaussian']):
    """
    Bins a series of data.

    Parameters
    ----------
    x : np.ndarray
        the x values of the data
    y : np.ndarray
        the y values of the data
    width : float
        bin width in muHz
    log : bool
        creates bins by using the log of the min/max values (i.e. not equally spaced in log if `True`)

    Returns
    -------
    bin_x : np.ndarray
        binned frequencies
    bin_y : np.ndarray
        binned power
    bin_yerr : numpy.ndarray
        standard deviation of the binned y data

    """
    if log:
        mi = np.log10(min(x))
        ma = np.log10(max(x))
        no = np.int(np.ceil((ma-mi)/width))
        bins = np.logspace(mi, mi+(no+1)*width, no)
    else:
        bins = np.arange(min(x), max(x)+width, width)

    digitized = np.digitize(x, bins)
    if mode == 'mean':
        bin_x = np.array([x[digitized == i].mean() for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
        bin_y = np.array([y[digitized == i].mean() for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
    elif mode == 'median':
        bin_x = np.array([np.median(x[digitized == i]) for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
        bin_y = np.array([np.median(y[digitized == i]) for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
    else:
        pass
    bin_yerr = np.array([y[digitized == i].std()/np.sqrt(len(y[digitized == i])) for i in range(1, len(bins)) if len(x[digitized == i]) > 0])

    return bin_x, bin_y, bin_yerr


def delta_nu(numax):
    """
    Estimates dnu using numax scaling relation.

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

