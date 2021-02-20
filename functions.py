import os
import numpy as np
import pandas as pd
from collections import deque
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel, convolve_fft
import pdb

##########################################################################################
#                                                                                        #
#                                      DICTIONARIES                                      #
#                                                                                        #
##########################################################################################

def get_info(args, star_info='Files/star_info.csv', params={}):
    
    with open('Files/todo.txt', "r") as f:
        todo = np.array([int(float(line.strip().split()[0])) for line in f.readlines()])
    params['path'] = 'Files/data/'
    params.update({'numax_sun':3090.,'dnu_sun':135.1,'width_sun':1300.,'todo':todo,'G':6.67428e-8, 
                   'tau_sun':[5.2e6, 1.8e5, 1.7e4, 2.5e3, 280., 80.],'teff_sun':5777.,'mass_sun':1.9891e33,
                   'tau_sun_single':[3.8e6, 2.5e5, 1.5e5, 1.0e5, 230., 70.],'radius_sun':6.95508e10})
    for target in todo:
        params[target] = {}
        params[target]['path'] = '/'.join(params['path'].split('/')[:-2])+'/results/%d/'%target
    args.params=params
    if args.excess:
        args = get_excess_params(args)
    else:
        args = get_excess_params(args)
        args.findex['do'] = False
    if args.background:
        args = get_bg_params(args)
    else:
        args = get_bg_params(args)
        args.fitbg['do'] = False
    args = get_star_info(args, star_info)
    return args

def get_excess_params(args, findex={}, save=True, step=0.25, binning=0.005, n_trials=3, 
                      lower=10., upper=4000.):
    findex['do']=True
    pars = ['save','step','binning','smooth_width','n_trials','lower','upper']
    vals = [save,step,binning,args.filter,n_trials,lower,upper]
    findex.update(dict(zip(pars, vals)))
    if findex['save']:
        for target in args.params['todo']:
            if not os.path.exists(args.params[target]['path']):
                os.makedirs(args.params[target]['path'])
    args.findex=findex
    return args

def get_bg_params(args, fitbg={}, save=True, box_filter=2.5, ind_width=50, n_rms=20, n_peaks=10,
                  force=False, guess=140.24, clip=True, ech_smooth=True, ech_filter=1., smooth_ps=1.0,
                  lower_numax=None, upper_numax=None, clip_value=0., lower=None, upper=None, slope=False,
                  samples=True):
    fitbg['do']=True
    pars = ['save','num_mc_iter','lower','upper','box_filter','ind_width','n_rms','n_peaks','smooth_ps','lower_numax','upper_numax','force','guess','clip','clip_value','ech_smooth','ech_filter','slope','samples']
    vals = [save,args.mciter,lower,upper,box_filter,ind_width,n_rms,n_peaks,smooth_ps,lower_numax,upper_numax,force,guess,clip,clip_value,ech_smooth,ech_filter,slope,samples]
    fitbg.update(dict(zip(pars, vals)))
    fitbg['functions'] = {1:harvey_one, 2:harvey_two, 3:harvey_three, 4:harvey_four, 5:harvey_five, 6:harvey_six}
    if fitbg['save']:
        for target in args.params['todo']:
            if not os.path.exists(args.params[target]['path']):
                os.makedirs(args.params[target]['path'])
    args.fitbg=fitbg
    return args

def get_star_info(args, star_info, cols=['rad','logg','teff']):
    if os.path.exists(star_info):
        df = pd.read_csv(star_info)
        targets = df.targets.values.tolist()
        for todo in args.params['todo']:
            if todo in targets:
                idx = targets.index(todo)
                for col in cols:
                    args.params[todo][col] = df.loc[idx,col]
                if 'numax' in df.columns.values.tolist():
                    args.params[todo]['numax'] = df.loc[idx,'numax']
                    args.params[todo]['dnu'] =  0.22*(df.loc[idx,'numax']**0.797)
                else:
                    args.params[todo]['mass'] = (((params[todo]['rad']*params['radius_sun'])**(2.))*10**(params[todo]['logg'])/params['G'])/params['mass_sun']
                    args.params[todo]['numax'] = params['numax_sun']*params[todo]['mass']*(params[todo]['rad']**(-2.))*((params[todo]['teff']/params['teff_sun'])**(-0.5))
                    args.params[todo]['dnu'] = params['dnu_sun']*(params[todo]['mass']**(0.5))*(params[todo]['rad']**(-1.5)) 
                for col in ['lowerx','upperx','lowerb','upperb']:
                    if np.isnan(df.loc[idx,col]):
                        args.params[todo][col] = None
                    else:
                        args.params[todo][col] = df.loc[idx,col]
    return args

def set_plot_params():
    plt.style.use('dark_background')
    plt.rcParams.update({
                         'agg.path.chunksize': 10000,
                         'mathtext.fontset': 'stix',
                         'figure.autolayout': True,
                         'lines.linewidth': 1,
                         'axes.titlesize': 18.,
                         'axes.labelsize': 16.,
                         'axes.linewidth': 1.25,
                         'axes.formatter.useoffset': False,
                         'xtick.major.size':10.,
                         'xtick.minor.size':5.,
                         'xtick.major.width':1.25,
                         'xtick.minor.width':1.25,
                         'xtick.direction': 'inout',
                         'ytick.major.size':10.,
                         'ytick.minor.size':5.,
                         'ytick.major.width':1.25,
                         'ytick.minor.width':1.25,
                         'ytick.direction': 'inout',
    })

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as mcolors
    new_cmap = mcolors.LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def gaussian_bounds(x, y, best_x=None, sigma=None):

    if sigma is None:
        sigma = (max(x)-min(x))/8./np.sqrt(8.*np.log(2.))
    bb = []
    b = np.zeros((2,4)).tolist()
    b[1][0] = np.inf
    b[1][1] = 2.*np.max(y)
    if not int(np.max(y)):
        b[1][1] = np.inf
    if best_x is not None:
        b[0][2] = best_x - 0.001*best_x
        b[1][2] = best_x + 0.001*best_x
    else:
        b[0][2] = np.min(x)
        b[1][2] = np.max(x)
    b[0][3] = sigma
    b[1][3] = np.max(x)-np.min(x)
    bb.append(tuple(b))
    return bb

def max_elements(x, y, npeaks):
    print(npeaks)
    s = np.argsort(y)
    peaks_y = y[s][-int(npeaks):][::-1]
    peaks_x = x[s][-int(npeaks):][::-1]
    return peaks_x, peaks_y

def return_max(array, index=False, dnu=False, exp_dnu=None):
    if dnu:
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


def power_law(frequency, pars, compare = False, power = None, error = None):

    model = np.array([pars[0]/(f**pars[1]) for f in frequency])
    if compare:
        return (power-model)/error
    else:
        return model


def lorentzian(frequency, pars, compare = False, power = None, error = None):

    model = np.array([pars])

    return model


def harvey(frequency, pars, mode='regular', gaussian=False, total=False):

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


def harvey_four(frequency, a1, b1, a2, b2, a3, b3, a4, b4, white_noise):

    model = np.zeros_like(frequency)

    model += a1/(1.+(b1*frequency)**2.+(b1*frequency)**4.)
    model += a2/(1.+(b2*frequency)**2.+(b2*frequency)**4.)
    model += a3/(1.+(b3*frequency)**2.+(b3*frequency)**4.)
    model += a4/(1.+(b4*frequency)**2.+(b4*frequency)**4.)
    model += white_noise

    return model


def harvey_five(frequency, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, white_noise):

    model = np.zeros_like(frequency)

    model += a1/(1.+(b1*frequency)**2.+(b1*frequency)**4.)
    model += a2/(1.+(b2*frequency)**2.+(b2*frequency)**4.)
    model += a3/(1.+(b3*frequency)**2.+(b3*frequency)**4.)
    model += a4/(1.+(b4*frequency)**2.+(b4*frequency)**4.)
    model += a5/(1.+(b5*frequency)**2.+(b5*frequency)**4.)
    model += white_noise

    return model


def harvey_six(frequency, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, white_noise):

    model = np.zeros_like(frequency)

    model += a1/(1.+(b1*frequency)**2.+(b1*frequency)**4.)
    model += a2/(1.+(b2*frequency)**2.+(b2*frequency)**4.)
    model += a3/(1.+(b3*frequency)**2.+(b3*frequency)**4.)
    model += a4/(1.+(b4*frequency)**2.+(b4*frequency)**4.)
    model += a5/(1.+(b5*frequency)**2.+(b5*frequency)**4.)
    model += a6/(1.+(b6*frequency)**2.+(b6*frequency)**4.)
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

    sx = sx[(sx!=0.)]
    se = se[(sy!=0.)]
    sy = sy[(sy!=0.)]
    se[(se==0.)] = np.median(se)

    return sx, sy, se


def bin_data(x, y, params):

    mi = np.log10(min(x))
    ma = np.log10(max(x))
    no = np.int(np.ceil((ma-mi)/params['binning']))
    bins = np.logspace(mi, mi+(no+1)*params['binning'], no)
    
    digitized = np.digitize(x, bins)
    bin_freq = np.array([x[digitized == i].mean() for i in range(1, len(bins)) if len(x[digitized == i]) > 0])
    bin_pow = np.array([y[digitized == i].mean() for i in range(1, len(bins)) if len(y[digitized == i]) > 0])
    
    return bin_freq, bin_pow


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


def delta_nu(numax):
    
    return 0.22*(numax**0.797)