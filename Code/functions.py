import numpy as np
from collections import deque
from scipy.special import erf
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel, convolve_fft
from scipy.ndimage import filters

##########################################################################################
#                                                                                        #
#                                   DIFFERENT MODELS                                     #
#                                                                                        #
##########################################################################################


def power_law(frequency, pars, compare = False, power = None, error = None):

    model = np.array([pars[0]/(f**pars[1]) for f in frequency])
    if compare:
        return (power-model)/error
    else:
        return model


def lorentzian(frequency, pars, compare = False, power = None, error = None):

    model = np.array([pars])

    return model


def harvey(frequency, pars, mode = 'regular', gaussian = False, total = False):

    if gaussian:
        nlaws = int((len(pars)-6)/2.)
    else:
        nlaws = int((len(pars)-1)/2.)
    model = np.zeros_like(frequency)

    if mode == 'regular':
        for i in range(nlaws):
            model += pars[i*2]/(1.+(pars[(i*2)+1]*frequency)**2.+(pars[(i*2)+1]*frequency)**4.)
    elif mode == 'second':
        for i in range(nlaws):
            model += pars[i*2]/(1.+(pars[(i*2)+1]*frequency)**2.)
    elif mode == 'fourth':
        for i in range(nlaws):
            model += pars[i*2]/(1.+(pars[(i*2)+1]*frequency)**4.)
    else:
        print('Wrong mode input for the harvey model function.')

    if gaussian:
        model += gauss_skew(frequency, pars[2*nlaws+1:])
    if total:
        model += pars[2*nlaws]
    return model


def generate_model(frequency, pars, pars_errs, nyquist):

    ps = np.zeros_like(frequency)

    for i, f in enumerate(frequency):

        r = (np.sin((np.pi*f)/(2.*nyquist))/((np.pi*f)/(2.*nyquist)))**2
        paras = [p+np.random.randn()*p_e for p, p_e in zip(pars, pars_errs)]
        nlaws = int((len(paras)-1.)/2.)
        m = 0
        for j in range(nlaws):
            m += paras[j*2]/(1.+(paras[(j*2)+1]*f)**2.+(paras[(j*2)+1]*f)**4.)
        m *= r
        m += pars[-1] + np.random.random_integers(-1,1)*(pars[-1]/2.)**(np.random.randn()-1.)
        if m < 0.:
            m = (10**(np.random.randn()))*r
        ps[i] = m

    return list(ps)


def gaussian(frequency, offset, amplitude, center, width):

    return offset + amplitude*np.exp(-(center-frequency)**2./(2.*width**2))


def harvey_one(frequency, a1, b1, white_noise):

    model = np.zeros_like(frequency)

    model += a1/(1.+(b1*frequency)**2.+(b1*frequency)**4.)
    model += white_noise

    return model


def harvey_two(frequency, a1, b1, a2, b2, white_noise):

    model = np.zeros_like(frequency)

    model += a1/(1.+(b1*frequency)**2.+(b1*frequency)**4.)
    model += a2/(1.+(b2*frequency)**2.+(b2*frequency)**4.)
    model += white_noise

    return model


def harvey_three(frequency, a1, b1, a2, b2, a3, b3, white_noise):

    model = np.zeros_like(frequency)

    model += a1/(1.+(b1*frequency)**2.+(b1*frequency)**4.)
    model += a2/(1.+(b2*frequency)**2.+(b2*frequency)**4.)
    model += a3/(1.+(b3*frequency)**2.+(b3*frequency)**4.)
    model += white_noise

    return model
    

def gaussian_skew(frequency, pars):

    model = np.array([2.*gauss(f, pars[0:4])*0.5*(1.+erf(pars[4]*((x-pars[1])/pars[2])/np.sqrt(2.))) for f in frequency])

    return model


##########################################################################################
#                                                                                        #
#                                DATA MANIPULATION ROUTINES                              #
#                                                                                        #
##########################################################################################


def mean_smooth_ind(x, y, width):

    step = width - 1
    sx = np.zeros_like(x)
    sy = np.zeros_like(x)
    se = np.zeros_like(x)

    j = 0

    while (j+step < len(x)-1):

        sx[j] = np.mean(x[j:j+step])
        sy[j] = np.mean(y[j:j+step])
        se[j] = np.std(y[j:j+step])/np.sqrt(width)
        j += step

    sx = sx[(sx!=0.)]
    se = se[(sy!=0.)]
    sy = sy[(sy!=0.)]
    se[(se==0.)] = np.median(se)

    return sx, sy, se


def bin_data(x, y, binning, log = True):

    if log:
        mi = min(np.log10(x))
        ma = max(np.log10(x))
        no = np.int(np.ceil((ma-mi)/binning))
        bins = np.logspace(mi, mi+no*binning, no)
    else:
        bins = np.arange(min(x), max(x)+binning, binning)
    
    digitized = np.digitize(x, bins)
    bin_means = [y[digitized == i].mean() for i in range(1, len(bins))]
    
    new_bins = np.zeros((len(bins)-1))
    for i in range(len(new_bins)):
        new_bins[i] = (bins[i] + bins[i+1])/2.
    
    return new_bins, bin_means


def smooth(array, width, params, method = 'box', mode = None, fft = False, silent = False):

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
            print('%s kernel: kernel size = %.2f muHz'%(method, width*params['resolution']))

    elif method == 'gaussian':

        n = 2*len(array)
        forward = array[:].tolist()
        reverse = array[::-1].tolist()

        if n%4 != 0:
            start = int(np.ceil(n/4))
        else:
            start = int(n/4)
        end = len(array)

        final = np.array(reverse[start:end]+forward[:]+reverse[:start])

        if isinstance(width, int):
            kernel = Gaussian1DKernel(width)
        else:
            width = int(np.ceil(width/params['resolution']))
            kernel = Gaussian1DKernel(width, mode = mode)

        if fft:
            smoothed = convolve_fft(final, kernel)
        else:
            smoothed = convolve(final, kernel)

        smoothed_array = smoothed[int(n/4):int(3*n/4)]

        if not silent:
            print('%s kernel: sigma = %.2f muHz'%(method, width*params['resolution']))
    else:
        print('Do not understand the smoothing method chosen.')

    return smoothed_array


def max_elements(array, N, resolution, limit = [False, None]):

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
        array[idx] = 0.

    return np.array(indices)


def smooth_gauss(array, fwhm, params, silent = False):

    sigma = fwhm/np.sqrt(8.*np.log(2.))

    n = 2*len(array)
    N = np.arange(1,n+1,1)
    mu = len(array)
    total = np.sum((1./(sigma*np.sqrt(2.*np.pi)))*np.exp(-0.5*(((N-mu)/sigma)**2.)))
    weights = ((1./(sigma*np.sqrt(2.*np.pi)))*np.exp(-0.5*(((N-mu)/sigma)**2.)))/total

    forward = array[:]
    reverse = array[::-1]

    if n%4 != 0:
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
        print('gaussian kernel using ffts: sigma = %.2f muHz'%(sigma*params['resolution']))
    if params['edge'][0]:
        smoothed_array = smoothed_array[:-params['edge'][1]]

    return np.array(smoothed_array)


def corr(frequency, power, params):

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