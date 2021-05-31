import numpy as np


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
        parametrization of the amplitude of the first harvey model
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
        parametrization of the amplitude of the first harvey model
    b1 : float
        the characteristic frequency of the first harvey model
    a2 : float
        parametrization of the amplitude of the second harvey model
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
        parametrization of the amplitude of the first harvey model
    b1 : float
        the characteristic frequency of the first harvey model
    a2 : float
        parametrization of the amplitude of the second harvey model
    b2 : float
        the characteristic frequency of the second harvey model
    a3 : float
        parametrization of the amplitude of the third harvey model
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