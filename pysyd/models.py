import numpy as np


def background(frequency, guesses, mode='regular', ab=False, noise=None):
    """
    The main model for the stellar background fitting

    Parameters
        frequency : numpy.ndarray
            the frequency of the power spectrum
        guesses : list
            the parameters of the Harvey model
        mode : {'regular', 'second', 'fourth'}
            the mode of which Harvey model parametrization to use. Default mode is `regular`.
            The 'regular' mode is when both the second and fourth order terms are added in the denominator
            whereas, 'second' only adds the second order term and 'fourth' only adds the fourth order term.
        total : bool
            If `True`, returns the summed model over multiple components. This is deprecated.
        ab : bool, optional
            If `True`, changes to the traditional a, b parametrization as opposed to the ``SYD``
        noise : None, optional
            If not `None`, it will fix the white noise to this value and not model it, reducing the dimension
            of the problem/model

    Returns
        model : np.ndarray
            the stellar background model
            
    TODO
        option to fix the white noise (i.e. ``noise`` option)
        option to change the parametrization (i.e. ``ab`` option)
        option to add power law

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
    Gaussian model
    
    Observed solar-like oscillations have a Gaussian-like profile and
    therefore, detections are modeled as a Gaussian distribution.

    Parameters
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
        result : np.ndarray
            the Gaussian distribution

    """

    model = np.zeros_like(frequency)
    model += amplitude*np.exp(-(center-frequency)**2.0/(2.0*width**2))
    model += offset

    return model


def harvey_none(frequency, white_noise, ab=False):
    """
    No Harvey model
    
    Stellar background model that does not contain any Harvey-like components
    i.e. this is the simplest model of all - consisting of a single white-noise
    component. This was added with the hopes that it would be preferred in the
    model selection for non-detections.
    
    .. warning::
        check if this is working for null detections

    Parameters
        frequency : numpy.ndarray
            the frequency array
        white_noise : float
            the white noise component

    Returns
        model : numpy.ndarray
            the no-Harvey (white noise) model


    """

    model = np.zeros_like(frequency)
    model += white_noise

    return model


def harvey_one(frequency, tau_1, sigma_1, white_noise, ab=False):
    """
    One Harvey model
    
    Stellar background model consisting of a single Harvey-like component

    Parameters
        frequency : numpy.ndarray
            the frequency array
        tau_1 : float
            timescale of the first harvey component
        sigma_1 : float
            amplitude of the first harvey component
        white_noise : float
            the white noise component

    Returns
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
    
    Stellar background model consisting of two Harvey-like components

    Parameters
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
    
    Stellar background model consisting of three Harvey-like components

    Parameters
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
        observations : numpy.ndarray
            the observed power spectrum
        model : numpy.ndarray
            model generated at the observed frequencies

    Returns
        LL : float
            the natural logarithm of the likelihood (or the MSE)

    """

    return -0.5*(np.sum((observations-model)**2.))


def compute_aic(observations, model, n_parameters):
    """
    Computes the Akaike Information Criterion (AIC) given the 
    background model of the power spectrum

    Parameters
        observations : numpy.ndarray
            the observed power spectrum
        model : numpy.ndarray
            model generated at the observed frequencies
        n_parameters : int
            number of free parameters in the given model

    Returns
        aic : float
            AIC value

    """
    N = len(observations)
    LL = log_likelihood(observations, model)
    aic = (-2.*LL)/N + (2.*n_parameters)/N

    return aic


def compute_bic(observations, model, n_parameters):
    """
    Computes the Bayesian Information Criterion (BIC) given the 
    background model of the power spectrum

    Parameters
        observations : numpy.ndarray
            the observed power spectrum
        model : numpy.ndarray
            model generated at the observed frequencies
        n_parameters : int
            number of free parameters in the given model

    Returns
        bic : float
            BIC value

    """
    N = len(observations)
    LL = log_likelihood(observations, model)
    bic = -2.*LL + np.log(N)*n_parameters

    return bic