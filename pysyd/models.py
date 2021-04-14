import numpy as np


def power_law(frequency, a, b):
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

    model = np.zeros_like(frequency)
    model += a*(frequency**b)

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


def harvey(frequency, pars, nlaws=None, mode='regular', total=False):
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
    total : bool
        if the model contains more than one harvey model, it will add up all background contributions.
        Default valus is `False`.

    Returns
    -------
    model : np.ndarray
        the Harvey model
    """

    if nlaws is None:
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

    if total:
        model += pars[2*nlaws]
    return model


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

    model = np.zeros_like(frequency)

    model += amplitude*np.exp(-(center-frequency)**2.0/(2.0*width**2))
    model += offset

    return model


def harvey_one(frequency, a1, b1, white_noise):
    """The first Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        the amplitude of the first harvey model
    b1 : float
        the characteristic frequency of the first harvey model
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
        the amplitude of the first harvey model
    b1 : float
        the characteristic frequency of the first harvey model
    a2 : float
        the amplitude of the second harvey model
    b2 : float
        the characteristic frequency of the second harvey model
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
        the amplitude of the first harvey model
    b1 : float
        the characteristic frequency of the first harvey model
    a2 : float
        the amplitude of the second harvey model
    b2 : float
        the characteristic frequency of the second harvey model
    a3 : float
        the amplitude of the third harvey model
    b3 : float
        the characteristic frequency of the third harvey model
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
        the amplitude of the first harvey model
    b1 : float
        the characteristic frequency of the first harvey model
    a2 : float
        the amplitude of the second harvey model
    b2 : float
        the characteristic frequency of the second harvey model
    a3 : float
        the amplitude of the third harvey model
    b3 : float
        the characteristic frequency of the third harvey model
    a4 : float
        the amplitude of the fourth harvey model
    b4 : float
        the characteristic frequency of the fourth harvey model
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
        the amplitude of the first harvey model
    b1 : float
        the characteristic frequency of the first harvey model
    a2 : float
        the amplitude of the second harvey model
    b2 : float
        the characteristic frequency of the second harvey model
    a3 : float
        the amplitude of the third harvey model
    b3 : float
        the characteristic frequency of the third harvey model
    a4 : float
        the amplitude of the fourth harvey model
    b4 : float
        the characteristic frequency of the fourth harvey model
    a5 : float
        the amplitude of the fifth harvey model
    b5 : float
        the characteristic frequency of the fifth harvey model
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
        the amplitude of the first harvey model
    b1 : float
        the characteristic frequency of the first harvey model
    a2 : float
        the amplitude of the second harvey model
    b2 : float
        the characteristic frequency of the second harvey model
    a3 : float
        the amplitude of the third harvey model
    b3 : float
        the characteristic frequency of the third harvey model
    a4 : float
        the amplitude of the fourth harvey model
    b4 : float
        the characteristic frequency of the fourth harvey model
    a5 : float
        the amplitude of the fifth harvey model
    b5 : float
        the characteristic frequency of the fifth harvey model
    a6 : float
        the amplitude of the sixth harvey model
    b6 : float
        the characteristic frequency of the sixth harvey model
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
