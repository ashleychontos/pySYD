import numpy as np


def background(frequency, guesses, mode='regular', ab=False, noise=None):
    """
    The main model for the stellar background fitting

    Parameters
    ----------
    frequency : numpy.ndarray
        the frequency of the power spectrum
    guesses : list
        the parameters of the Harvey model
    mode : {'regular', 'second', 'fourth'}
        the mode of which Harvey model parametrization to use. Default mode is `regular`.
        The 'regular' mode is when both the second and fourth order terms are added in the denominator
        whereas, 'second' only adds the second order term and 'fourth' only adds the fourth order term.
    total : bool
        If `True`, returns the summed model over multiple components.

    Returns
    -------
    model : np.ndarray
        the stellar background model

    """
    nlaws = int(len(guesses)//2)
    model = np.zeros_like(frequency)

    if mode == 'regular':
        for i in range(nlaws):
            if not ab:
                model += (4.*(guesses[(i*2)+1]**2.)*guesses[i*2])/(1.0+(2.*np.pi*guesses[i*2]*frequency)**2.0+(2.*np.pi*guesses[i*2]*frequency)**4.0)
            else:
                model += guesses[i*2]/(1.0+(guesses[(i*2)+1]*frequency)**2.0+(guesses[(i*2)+1]*frequency)**4.0)
    elif mode == 'second':
        for i in range(nlaws):
            if not ab:
                model += (4.*(guesses[(i*2)+1]**2.)*guesses[i*2])/(1.0+(2.*np.pi*guesses[i*2]*frequency)**2.0)
            else:
                model += guesses[i*2]/(1.0+(guesses[(i*2)+1]*frequency)**2.0)
    elif mode == 'fourth':
        for i in range(nlaws):
            if not ab:
                model += (4.*(guesses[(i*2)+1]**2.)*guesses[i*2])/(1.0+(2.*np.pi*guesses[i*2]*frequency)**4.0)
            else:
                model += guesses[i*2]/(1.0+(guesses[(i*2)+1]*frequency)**4.0)
    else:
        pass

    if not int(len(guesses)%2):
        if noise is not None:
            model += noise
    else:
        model += guesses[-1]

    return model


def gaussian(frequency, offset, amplitude, center, width):
    """
    The Gaussian function.

    Parameters
    ----------
    frequency : numpy.ndarray
        the frequency array
    offset : float
        the vertical offset
    amplitude : float
        amplitude of the Gaussian
    center : float
        center of the Gaussian
    width : float
        the width of the Gaussian

    Returns
    -------
    result : np.ndarray
        the Gaussian function

    """

    model = np.zeros_like(frequency)
    model += amplitude*np.exp(-(center-frequency)**2.0/(2.0*width**2))
    model += offset

    return model


def harvey_none(frequency, white_noise, ab=False):
    """
    No Harvey model

    Parameters
    ----------
    frequency : numpy.ndarray
        the frequency array
    white_noise : float
        the white noise component

    Returns
    -------
    model : numpy.ndarray
        the no-Harvey (white noise) model

    """

    model = np.zeros_like(frequency)
    model += white_noise

    return model


def harvey_one(frequency, tau_1, sigma_1, white_noise, ab=False):
    """
    One Harvey model

    Parameters
    ----------
    frequency : numpy.ndarray
        the frequency array
    tau_1 : float
        timescale of the first harvey component [Ms]
    sigma_1 : float
        amplitude of the first harvey component
    white_noise : float
        the white noise component

    Returns
    -------
    model : numpy.ndarray
        the one-Harvey model

    """

    model = np.zeros_like(frequency)
    if not ab:
        model += (4.*(sigma_1**2.)*tau_1)/(1.0+(2.*np.pi*tau_1*frequency)**2.0+(2.*np.pi*tau_1*frequency)**4.0)
    else:
        model += tau_1/(1.0+(sigma_1*frequency)**2.0+(sigma_1*frequency)**4.0)
    model += white_noise

    return model


def harvey_two(frequency, tau_1, sigma_1, tau_2, sigma_2, white_noise, ab=False):
    """
    Two Harvey model

    Parameters
    ----------
    frequency : numpy.ndarray
        the frequency array
    tau_1 : float
        timescale of the first harvey component
    sigma_1 : float
        amplitude of the first harvey component
    tau_2 : float
        timescale of the second harvey component
    sigma_2 : float
        amplitude of the second harvey component
    white_noise : float
        the white noise component

    Returns
    -------
    model : numpy.ndarray
        the two-Harvey model

    """

    model = np.zeros_like(frequency)
    if not ab:
        model += (4.*(sigma_1**2.)*tau_1)/(1.0+(2.*np.pi*tau_1*frequency)**2.0+(2.*np.pi*tau_1*frequency)**4.0)
        model += (4.*(sigma_2**2.)*tau_2)/(1.0+(2.*np.pi*tau_2*frequency)**2.0+(2.*np.pi*tau_2*frequency)**4.0)
    else:
        model += tau_1/(1.0+(sigma_1*frequency)**2.0+(sigma_1*frequency)**4.0)
        model += tau_2/(1.0+(sigma_2*frequency)**2.0+(sigma_2*frequency)**4.0)
    model += white_noise

    return model


def harvey_three(frequency, tau_1, sigma_1, tau_2, sigma_2, tau_3, sigma_3, white_noise, ab=False):
    """
    Three Harvey model

    Parameters
    ----------
    frequency : numpy.ndarray
        the frequency array
    tau_1 : float
        timescale of the first harvey component
    sigma_1 : float
        amplitude of the first harvey component
    tau_2 : float
        timescale of the second harvey component
    sigma_2 : float
        amplitude of the second harvey component
    tau_3 : float
        timescale of the third harvey component
    sigma_3 : float
        amplitude of the third harvey component
    white_noise : float
        the white noise component

    Returns
    -------
    model : numpy.ndarray
        the three-Harvey model
    """

    model = np.zeros_like(frequency)
    if not ab:
        model += (4.*(sigma_1**2.)*tau_1)/(1.0+(2.*np.pi*tau_1*frequency)**2.0+(2.*np.pi*tau_1*frequency)**4.0)
        model += (4.*(sigma_2**2.)*tau_2)/(1.0+(2.*np.pi*tau_2*frequency)**2.0+(2.*np.pi*tau_2*frequency)**4.0)
        model += (4.*(sigma_3**2.)*tau_3)/(1.0+(2.*np.pi*tau_3*frequency)**2.0+(2.*np.pi*tau_3*frequency)**4.0)
    else:
        model += tau_1/(1.0+(sigma_1*frequency)**2.0+(sigma_1*frequency)**4.0)
        model += tau_2/(1.0+(sigma_2*frequency)**2.0+(sigma_2*frequency)**4.0)
        model += tau_3/(1.0+(sigma_3*frequency)**2.0+(sigma_3*frequency)**4.0)
    model += white_noise

    return model


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