import os
import ast
import glob
import subprocess
import numpy as np
import pandas as pd
from astropy.io import ascii
import multiprocessing as mp
from astropy.stats import mad_std
from scipy.signal import find_peaks


# Package mode
from . import models
from . import _ROOT, PACKAGEDIR



class InputError(Exception):
    """Class for pySYD user input errors (i.e., halts execution)."""
    def __init__(self, error, width=60):
        self.msg, self.width = error, width
    def __repr__(self):
        return "pysyd.utils.InputError(error=%r)" % self.msg
    def __str__(self):
        return "<InputError>"

class ProcessingError(Exception):
    """Class for pySYD processing errors (i.e., halts execution)."""
    def __init__(self, error, width=60):
        self.msg, self.width = error, width
    def __repr__(self):
        return "pysyd.utils.ProcessingError(error=%r)" % self.msg
    def __str__(self):
        return "<ProcessingError>"

class InputWarning(Warning):
    """Class for pySYD user input warnings."""
    def __init__(self, warning, width=60):
        self.msg, self.width = warning, width
    def __repr__(self):
        return "pysyd.utils.InputWarning(warning=%r)" % self.msg
    def __str__(self):
        return "<InputWarning>"

class ProcessingWarning(Warning):
    """Class for pySYD user input warnings."""
    def __init__(self, warning, width=60):
        self.msg, self.width = warning, width
    def __repr__(self):
        return "pysyd.utils.ProcessingWarning(warning=%r)" % self.msg
    def __str__(self):
        return "<ProcessingWarning>"


class Constants:
    """Constants
    
    Container class for constants and known values -- which is
    primarily solar asteroseismic values for our purposes.
    
    """

    def __init__(self):
        """
        UNITS ARE IN THE SUPERIOR CGS 
        COME AT ME

        """
        self.constants = {
            'm_sun' : 1.9891e33,
            'r_sun' : 6.95508e10,
            'rho_sun' : 1.41,
            'teff_sun' : 5777.0,
            'logg_sun' : 4.4,
            'teffred_sun' : 8907.0,
            'numax_sun' : 3090.0,
            'dnu_sun' : 135.1,
            'width_sun' : 1300.0,
            'tau_sun' : [5.2e6,1.8e5,1.7e4,2.5e3,280.0,80.0],
            'tau_sun_single' : [3.8e6,2.5e5,1.5e5,1.0e5,230.,70.],
            'G' : 6.67428e-8,
            'cm2au' : 6.68459e-14,
            'au2cm' : 1.496e+13,
            'rad2deg' : 180./np.pi,
            'deg2rad' : np.pi/180.,
        }

    def __str__(self):
        return "<Constants>"

    def __repr__(self):
        return "utils.Constants()"


class Parameters(Constants):
    """Container class for ``pySYD`` parameters

    """
    def __init__(self, args=None):
        """
        Calls super method to inherit all relevant constants and then
        stores the default values for all pysyd modules

        Methods

        """
        # makes sure to inherit constants
        super().__init__()
        self.get_defaults()
        if args is not None:
            self.add_cli(args)


    def __str__(self):
        return "<Parameters>"


    def __repr__(self):
        return "utils.Parameters(Constants, params={})"


    def get_defaults(self):
        """Load defaults

        Gets default pySYD parameters by calling functions which are analogous to 
        available command-line parsers and arguments

        Attributes
            params : Dict[str[Dict[,]]]
                container class for ``pySYD`` parameters
    
        Calls
            - :mod:`pysyd.utils.Parameters.get_parent`
            - :mod:`pysyd.utils.Parameters.get_data`
            - :mod:`pysyd.utils.Parameters.get_main`
            - :mod:`pysyd.utils.Parameters.get_plot`

        """
        self.params = {}
        # Initialize high level functionality
        self.get_parent()
        # Get parameters related to data loading/saving
        self.get_data()
        # Initialize main 'params' dictionary
        self.get_main()
        # Get plotting info
        self.get_plot()


    def get_parent(self, inpdir='data', infdir='info', outdir='results', save=True, test=False,
                   verbose=False, overwrite=False, warnings=False, cli=True, notebook=False):
        """Get parent parser
   
        Load parameters available in the parent parser i.e. higher-level software functionality

        Attributes
            params : Dict[str,Dict[,]]
                the updated parameters

        """
        self.params.update({
            'inpdir' : os.path.join(_ROOT, 'data'),
            'infdir' : os.path.join(_ROOT, 'info'),
            'outdir' : os.path.join(_ROOT, 'results'),
            'save' : True,
            'test' : False,
            'verbose' : False,
            'overwrite' : False,
            'warnings' : False,
            'cli' : True,
            'notebook' : False,
        })


    def get_data(self, info='info/star_info.csv', todo='info/todo.txt', stars=None, mode='run',
                 gap=20, stitch=False, oversampling_factor=None, kep_corr=False, notching=False,
                 dnu=None, lower_ech=None, upper_ech=None):
        """Get data parser
   
        Load parameters available in the data parser, which is mostly related to initial
        data loading and manipulation

        Attributes
            params : Dict[str,Dict[,]]
                the updated parameters

        """
        self.params.update({
            'info' : os.path.join(_ROOT, 'info', 'star_info.csv'),
            'todo' : os.path.join(_ROOT, 'info', 'todo.txt'),
            'stars' : None,
            'mode' : 'run',
            'gap' : 20,
            'stitch' : False,
            'oversampling_factor' : None,
            'kep_corr' : False,
            'force' : False,
            'dnu' : None,
            'lower_ech' : None,
            'upper_ech' : None,
            'notching' : False,
        })


    def get_main(self):
        """Get main parser
   
        Load parameters available in the main parser i.e. core software functionality

        Attributes
            params : Dict[str,Dict[,]]
                the updated parameters

        Calls
            - :mod:`pysyd.utils.Parameters.get_estimate`
            - :mod:`pysyd.utils.Parameters.get_background`
            - :mod:`pysyd.utils.Parameters.get_global`
            - :mod:`pysyd.utils.Parameters.get_sampling`

        """
        # Initialize parameters for the estimate routine
        self.get_estimate()
        # Initialize parameters for the fit background routine
        self.get_background()
        # Initialize parameters relevant for estimating global parameters
        self.get_global()
        # Estimate parameters
        self.get_sampling()


    def get_estimate(self, estimate=True, smooth_width=20.0, binning=0.005, bin_mode='mean', step=0.25,
                     n_trials=3, ask=False, lower_ex=None, upper_ex=None):
        """Search and estimate parameters
    
        Get parameters relevant for the optional first module that looks for and identifies
        power excess due to solar-like oscillations and then estimates its properties 

        Attributes
            params : Dict[str,Dict[,]]
                the updated parameters

        """
        self.params.update({
            'estimate' : True,
            'smooth_width' : 20.0,
            'binning' : 0.005,
            'bin_mode' : 'mean',
            'step' : 0.25,
            'n_trials' : 3,
            'ask' : False,
            'lower_ex' : None,
            'upper_ex' : None,
        })


    def get_background(self, background=True, basis='tau_sigma', box_filter=1.0, ind_width=20.0, n_rms=20,
                       n_laws=None, fix_wn=False, metric='bic', lower_bg=None, upper_bg=None, functions=None):
        """Background parameters
    
        Gets parameters used during the automated background-fitting analysis 

        Attributes
            params : Dict[str,Dict[,]]
                the updated parameters

        """
        self.params.update({
            'background' : True,
            'basis' : 'tau_sigma',
            'box_filter' : 1.0,
            'ind_width' : 20.0,
            'n_rms' : 20,
            'n_laws' : None,
            'fix_wn' : False,
            'metric' : 'bic',
            'lower_bg' : None,
            'upper_bg' : None,
            'functions' : get_dict(type='functions'),
        })


    def get_global(self, globe=True, numax=None, lower_ps=None, upper_ps=None, ex_width=1.0, 
                   sm_par=None, smooth_ps=2.5, fft=True, threshold=1.0, n_peaks=5):
        """Global fitting parameters
    
        Get default parameters that are relevant for deriving global asteroseismic parameters 
        :math:`\\rm \\nu_{max}` and :math:`\\Delta\\nu`

        Attributes
            params : Dict[str,Dict[,]]
                the updated parameters

        """
        self.params.update({
            'globe' : True,
            'numax' : None,
            'lower_ps' : None,
            'upper_ps' : None,
            'ex_width' : 1.0,
            'sm_par' : None,
            'smooth_ps' : 2.5,
            'fft' : True,
            'threshold' : 1.0,
            'n_peaks' : 5,
        })


    def get_sampling(self, mc_iter=1, seed=None, samples=False, n_threads=0):
        """Sampling parameters
    
        Get parameters relevant for the sampling steps i.e. estimating uncertainties

        Attributes
            params : Dict[str,Dict[,]]
                the updated parameters

        """
        self.params.update({
            'mc_iter' : 1,
            'seed' : None,
            'samples' : False,
            'n_threads' : 0,
        })


    def get_plot(self, show_all=False, show=False, cmap='binary', hey=False, clip_value=3.0,
                 interp_ech=False, nox=None, noy='0+0', npb=10, ridges=False, smooth_ech=None):
        """Get plot parser
    
        Save all parameters related to any of the output figures

        Attributes
            params : Dict[str,Dict[,]]
                the updated parameters

        """
        self.params.update({
            'show_all' : False,
            'show' : False,
            'cmap' : 'binary',
            'hey' : False,
            'clip_value' : 3.0,
            'interp_ech' : False,
            'nox' : None,
            'noy' : '0+0',
            'npb' : 10,
            'ridges' : False,
            'smooth_ech' : None,
        })


    def add_cli(self, args):
        """Add CLI

        Save any non-default parameters provided via command line but skips over any keys
        in the override columns since those are star specific and have a given length --
        it will come back to this

        Parameters
            args : argparse.Namespace
                the command line arguments

        """
        self.check_cli(args)
        # CLI options overwrite defaults
        for key, value in args.__dict__.items():
            # Make sure it is not a variable with a >1 length
            if key not in self.override:
                self.params[key] = value


    def check_cli(self, args, max_laws=3, override=['numax','dnu','lower_ex','upper_ex','lower_bg','upper_bg','lower_ps','upper_ps','lower_ech','upper_ech']):
        """Check CLI
    
        Make sure that any command-line inputs are the proper lengths, types, etc.

        Parameters
            args : argparse.Namespace
                the command line arguments
            max_laws : int
                maximum number of Harvey laws to be fit

        Asserts
            - the length of each array provided (in override) is equal
            - the oversampling factor is an integer (if applicable)
            - the number of Harvey laws to "force" is an integer (if applicable)

        """
        self.override = {
            'numax': args.numax,
            'dnu': args.dnu,
            'lower_ex': args.lower_ex,
            'upper_ex': args.upper_ex,
            'lower_bg': args.lower_bg,
            'upper_bg': args.upper_bg,
            'lower_ps': args.lower_ps,
            'upper_ps': args.upper_ps,
            'lower_ech': args.lower_ech,
            'upper_ech': args.upper_ech,
        }
        for each in self.override:
            if self.override[each] and len(self.override[each]) != len(args.stars):
                raise InputError("\nThe number \n of values provided for %s MUST equal the number of stars\n" % each)
        if args.oversampling_factor is not None and not isinstance(args.oversampling_factor, int):
            raise InputError("\nOversampling factor for input PS must be an integer\n")
        if args.n_laws is not None and args.n_laws > max_laws:
            args.n_laws = max_laws
            raise InputWarning("\nWe probs cannot resolve %d Harvey components. \nnlaws changed to %d\n" % max_laws)


    def add_targets(self, stars=None):
        """Add targets

        This was mostly added for non-command-line users, since this makes
        API usage easier.

        """
        if 'stars' not in self.params or self.params['stars'] is None:
            if stars is not None:
                if isinstance(stars, list):
                    self.params['stars'] = stars
                else:
                    self.params['stars'] = [stars]
            else:
                try:
                    loadstars = self._load_starlist()
                except InputError as error:
                    print(error.msg)
                    return
                else:
                    self.params['stars'] = loadstars
        self._make_dicts()


    def _load_starlist(self):
        """Load star list

        If no stars have been provided yet, it will read in the default text file
        (and if that does not exist, it will raise an error)

        """
        if not os.path.exists(self.params['todo']):
            raise InputError("\nERROR: no stars or star list were provided for the software to run -- please try again\n       alternatively, you can run 'pysyd setup' to easily download example files\n")
            return None
        else:
            with open(self.params['todo'], "r") as f:
                stars = [line.strip().split()[0] for line in f.readlines()]
            return stars


    def _make_dicts(self):
        """Add star dicts

        This routine will load in target stars, sets up "groups" (relevant for parallel
        processing) and then load in all relevant information

        """
        # Set file paths and make directories if they don't yet exist
        for star in self.params['stars']:
            self.params[star] = {}
            self.params[star]['path'] = os.path.join(self.params['outdir'], str(star))
            if self.params['save'] and not os.path.exists(self.params[star]['path']):
                os.makedirs(self.params[star]['path'])
        self._get_groups()
        self._load_starinfo()
        self._load_clinfo()
        self._add_derived()


    def _get_groups(self):
        """Get star groups
    
        Mostly relevant for parallel processing -- which sets up star groups to run 
        in parallel based on the number of threads

        """
        if hasattr(self, 'mode') and self.params['mode'] == 'parallel':
            todo = np.array(self.params['stars'])
            if self.params['n_threads'] == 0:
                self.params['n_threads'] = mp.cpu_count()
            if len(todo) < self.params['n_threads']:
                self.params['n_threads'] = len(todo)
            # divide stars into groups set by number of cpus/nthreads available
            digitized = np.digitize(np.arange(len(todo)) % self.params['n_threads'], np.arange(self.params['n_threads']))
            self.params['groups'] = np.array([todo[digitized == i] for i in range(1, self.params['n_threads']+1)], dtype=object)
        else:
            self.params['groups'] = np.array(self.params['stars'])


    def _load_starinfo(self):
        """Load star info csv

        """
        if os.path.exists(self.params['info']):
            columns = get_dict(type='columns')
            df = pd.read_csv(self.params['info'])
            stars = [str(star) for star in df.star.values.tolist()]
            for star in self.params['stars']:
                if str(star) in stars:
                    idx = stars.index(str(star))
                    for column in df.columns.values.tolist():
                        if not np.isnan(df.loc[idx,column]) and column in columns['int']:
                            self.params[star][column] = int(df.loc[idx,column])
                        elif not np.isnan(df.loc[idx,column]) and column in columns['float']:
                            self.params[star][column] = float(df.loc[idx,column])
                        elif not np.isnan(df.loc[idx,column]) and column in columns['bool']:
                            self.params[star][column] = df.loc[idx,column]
                        elif not np.isnan(df.loc[idx,column]) and column in columns['str']:
                            self.params[star][column] = str(df.loc[idx,column])
                        else:
                            pass


    def _load_clinfo(self):
        """Load command-line values

        """
        columns = get_dict(type='columns')
        # make sure all keys exist, even if they are None
        for star in self.params['stars']:
            for column in columns['all']:
                if column in list(self.params.keys()) and column not in columns['override']:
                    self.params[star][column] = self.params[column]
                else:
                    if column not in self.params[star]:
                        self.params[star][column] = None
        # CLI override parameters
        if hasattr(self, 'override'):
            for column in columns['override']:
                if self.override[column] is not None:
                    for i, star in enumerate(self.params['stars']):
                        self.params[star][column] = self.override[column][i]


    def _add_derived(self):
        """Add derived properties

        """
        for star in self.params['stars']:
            if self.params[star]['numax'] is not None:
                self.params[star]['estimate'] = False
                if self.params[star]['dnu'] is None:
                    self.params[star]['dnu'] = delta_nu(self.params[star]['numax'])
            else:
                if 'rs' in self.params[star] and self.params[star]['rs'] is not None and 'logg' in self.params[star] and self.params[star]['logg'] is not None:
                    self.params[star]['ms'] = ((((self.params[star]['rs']*self.constants['r_sun'])**(2.0))*10**(self.params[star]['logg'])/self.constants['G'])/self.constants['m_sun'])
                    self.params[star]['numax'] = self.constants['numax_sun']*self.params[star]['ms']*(self.params[star]['rs']**(-2.0))*((self.params[star]['teff']/self.constants['teff_sun'])**(-0.5))
                    self.params[star]['dnu'] = self.constants['dnu_sun']*(self.params[star]['ms']**(0.5))*(self.params[star]['rs']**(-1.5))  
            if self.params[star]['lower_ech'] is not None and self.params[star]['upper_ech'] is not None:
                self.params[star]['ech_mask'] = [self.params[star]['lower_ech'], self.params[star]['upper_ech']]
            else:
                self.params[star]['ech_mask'] = None


class Question:

    # QUESTIONS CLASS
    def __init__(self, max_attempts=10):
        """Questions 

        Parameters
            max_attempts : int, default=100
                the maximum number of tries a user has before breaking

        """
        self.max_attempts = max_attempts

    def __repr__(self):
        pass

    # ASK BOOLEAN/BINARY
    def ask_boolean(self, question, count=1):
        """Boolean user question
        
        Asks for a boolean (/binary) input 
        
        Parameters
            question : str
                the statement and/or question that needs to be answered
            max_attempts : int, default=100
                the maximum number of tries a user has before breaking
            count : int
                the attempt number
        
        Returns
            answer : bool
                the provided user input as a `True``/`False` boolean (and `False` if number of 
                attempts exceeded `max_attempts`)
        
        """
        print()
        while count < self.max_attempts:
            answer = input('\n%s' % question)
            if answer in ["y", "Y", "1", "yes", "Yes", "YES", "yy", "YY", "T", "True", "TRUE", "t", "true", 1, 1.0]:
                return True
            elif answer in ["n", "N", "0", "no", "NO", "nn", "No", "NN", "F", "False", "FALSE", "f", "false", 0, 0.0]:
                return False
            else:
                count += 1
                print("ERROR: not a valid response \nPlease try again (%d attempts remaining)" % (self.max_attempts-count))
        print('Exceeded maximum number of attempts.\nPlease check your input and try again.')
        return None

    # ASK INTEGER
    def ask_integer(self, question, count=1, special=False, n_trials=None): 
        """Integer user input
        
        Asks for an integer input -- this is specifically formatted for the module that searches
        and identifies power excess due to solar-like oscillations and then estimates its properties.
        Therefore it forces a selection that matches with one of the (n_)trials or accepts zero to 
        provide your own estimate for :math:`\\rm \\nu_{max}` directly
        
        Parameters
            question : str
                the statement and/or question that needs to be answered
            count : int
                current attempt number
            special : bool, default=False
                changes to `True` if the input is zero, changing the integer requirement to
                a float to provide :term:`numax`
        
        Returns
            answer : int
                the provided user input (either an integer corresponding to the trial number or
                a numax estimate) *or* `None` if number of attemps exceeded `max_attempts`
        
        """  
        while count < self.max_attempts:
            answer = input('\n%s'%question)
            try:
                if float(answer).is_integer():
                    if special:
                        if int(answer) == 0:
                            return self.ask_float(question='\nWhat is your value for numax? ', count=count)
                        elif int(answer) >= 1 and int(answer) <= n_trials:
                            return int(answer)
                        else:
                            print('ERROR: please select an integer between 1 and %d \n       (or 0 to provide your own value for numax)\n'%n_trials)
                    else:
                        return int(answer)
            except ValueError:
                print("ERROR: not a valid response \n")
            count += 1
        print('Exceeded maximum number of attempts.\nPlease check your input and try again.')
        return None

    # ASK FLOAT
    def ask_float(self, question, count=1): 
        """Float user input
        
        Asks for a float input -- this is specifically formatted for the module that searches
        and identifies power excess due to solar-like oscillations and then estimates its properties.
        Therefore it forces a selection that matches with one of the (n_)trials or accepts zero to 
        provide your own estimate for :math:`\\rm \\nu_{max}` directly
        
        Parameters
            question : str
                the statement and/or question that needs to be answered
            count : int
                current attempt number
        
        Returns
            answer : float
                the provided user input
        
        """  
        print() 
        while count < self.max_attempts:
            answer = input('\n%s'%question)
            try:
                if float(answer):
                    return float(answer)
            except ValueError:
                print("ERROR: not a valid response \n")
            count += 1
        print('Exceeded maximum number of attempts.\nPlease check your input and try again.')
        return None


def get_dict(type='params'):
    """Get dictionary
    
    Quick+convenient utility function to read in longer dictionaries that are used throughout
    the software

    Parameters
        type : {'columns','params','plots','tests','functions'}, default='params'
            which dictionary to load in -- which *MUST* match their relevant filenames

    Returns
        result : Dict[str,Dict[,]]
            the relevant (type) dictionary

    .. important:: 
       `'functions'` cannot be saved and loaded in like the other dictionarie because it
       points to modules loaded in from another file 

    """
    if type == 'functions':
        return {
                0: lambda white_noise : (lambda frequency : models.harvey_none(frequency, white_noise)),
                1: lambda frequency, white_noise : models.harvey_none(frequency, white_noise), 
                2: lambda white_noise : (lambda frequency, tau_1, sigma_1 : models.harvey_one(frequency, tau_1, sigma_1, white_noise)), 
                3: lambda frequency, tau_1, sigma_1, white_noise : models.harvey_one(frequency, tau_1, sigma_1, white_noise), 
                4: lambda white_noise : (lambda frequency, tau_1, sigma_1, tau_2, sigma_2 : models.harvey_two(frequency, tau_1, sigma_1, tau_2, sigma_2, white_noise)), 
                5: lambda frequency, tau_1, sigma_1, tau_2, sigma_2, white_noise : models.harvey_two(frequency, tau_1, sigma_1, tau_2, sigma_2, white_noise),
                6: lambda white_noise : (lambda frequency, tau_1, sigma_1, tau_2, sigma_2, tau_3, sigma_3 : models.harvey_three(frequency, tau_1, sigma_1, tau_2, sigma_2, tau_3, sigma_3, white_noise)),
                7: lambda frequency, tau_1, sigma_1, tau_2, sigma_2, tau_3, sigma_3, white_noise : models.harvey_three(frequency, tau_1, sigma_1, tau_2, sigma_2, tau_3, sigma_3, white_noise),
               }
    path = os.path.join(PACKAGEDIR, 'data', 'dicts', '%s.dict'%type)
    with open(path, 'r') as f:
        return ast.literal_eval(f.read())


def _save_file(x, y, path, overwrite=False, formats=[">15.8f", ">18.10e"]):
    """Saves basic text files
    
    After determining the best-fit stellar background model, this module
    saved the background-subtracted power spectrum

    Parameters
        x : numpy.ndarray
            the independent variable i.e. the time or frequency array 
        y : numpy.ndarray
            the dependent variable, in this case either the flux or power array
        path : str
            absolute path to save file to
        overwrite : bool, optional
            whether to overwrite existing files or not
        formats : List[str]
            2x1 list of formats to save arrays as


    """
    if os.path.exists(path) and not overwrite:
        path = _get_next(path)
    with open(path, "w") as f:
        for xx, yy in zip(x, y):
            values = [xx, yy]
            text = '{:{}}'*len(values) + '\n'
            fmt = sum(zip(values, formats), ())
            f.write(text.format(*fmt))


def _get_next(path, count=1):
    """
    Get next integer
    
    When the overwriting of files is disabled, this module determines what
    the last saved file was 

    Parameters
        path : str
            absolute path to file name that already exists
        count : int
            starting count, which is incremented by 1 until a new path is determined

    Returns
        new_path : str
            unused path name


    """
    path, fname = os.path.split(path)[0], os.path.split(path)[-1]
    new_fname = '%s_%d.%s'%(fname.split('.')[0], count, fname.split('.')[-1])
    new_path = os.path.join(path, new_fname)
    if os.path.exists(new_path):
        while os.path.exists(new_path):
            count += 1
            new_fname = '%s_%d.%s'%(fname.split('.')[0], count, fname.split('.')[-1])
            new_path = os.path.join(path, new_fname)
    return new_path


def _save_estimates(star, variables=['star','numax','dnu','snr']):
    """
    
    Saves the parameter estimates (i.e. results from first module)

    Parameters
        star : pysyd.target.Target
            processed pipeline target
        variables : List[str]
            list of estimated variables to save (e.g., :math:`\\rm \\nu_{max}`, :math:`\\Delta\\nu`)

    Returns
        star : pysyd.target.Target
            updated pipeline target

    """
    if 'best' in star.params:
        best = star.params['best']
        results = [star.name, star.params['results']['estimates'][best]['value'], delta_nu(star.params['results']['estimates'][best]['value']), star.params['results']['estimates'][best]['snr']]
        if star.params['save']:
            save_path = os.path.join(star.params['path'], 'estimates.csv')
            if not star.params['overwrite']:
                save_path = _get_next(save_path)
            ascii.write(np.array(results), save_path, names=variables, delimiter=',', overwrite=True)
        star.params['numax'], star.params['dnu'], star.params['snr'] = results[1], results[2], results[3]
    return star


def _save_parameters(star):
    """
    
    Saves the derived global parameters 

    Parameters
        star : pysyd.target.Target
            pipeline target with the results of the global fit
        results : Dict[str,float]
            parameter estimates from the pipeline
        cols : List[str]
            columns used to construct a pandas dataframe

    """
    if star.params['save']:
        results = star.params['results']['parameters']
        df = pd.DataFrame(results)
        star.df = df.copy()
        new_df = pd.DataFrame(columns=['parameter','value'])
        for c, col in enumerate(df.columns.values.tolist()):
            new_df.loc[c, 'parameter'] = col
            new_df.loc[c, 'value'] = df.loc[0,col]
            if star.params['mc_iter'] > 1:
                new_df.loc[c, 'uncertainty'] = mad_std(df[col].values)
        save_path = os.path.join(star.params['path'], 'global.csv')
        if not star.params['overwrite']:
            save_path = _get_next(save_path)
        new_df.to_csv(save_path, index=False)
        if star.params['mc_iter'] > 1:
            star.params['plotting']['samples'] = {'df':star.df.copy()}
        if star.params['samples']:
            save_path = os.path.join(star.params['path'], 'samples.csv')
            if not star.params['overwrite']:
                save_path = _get_next(save_path)
            df.to_csv(save_path, index=False)


def _verbose_output(star, note=''):
    """

    Prints verbose output from the global fit 


    """
    params = get_dict()
    if not star.params['overwrite']:
        list_of_files = glob.glob(os.path.join(star.params['path'], 'global*'))
        file = max(list_of_files, key=os.path.getctime)
    else:
        file = os.path.join(star.params['path'], 'global.csv')
    df = pd.read_csv(file)
    note += '-----------------------------------------------------------\nOutput parameters\n-----------------------------------------------------------'
    if star.params['mc_iter'] > 1:
        line = '\n%s: %.2f +/- %.2f %s'
        for idx in df.index.values.tolist():
            note += line%(df.loc[idx,'parameter'], df.loc[idx,'value'], df.loc[idx,'uncertainty'], params[df.loc[idx,'parameter']]['unit'])
    else:
        line = '\n%s: %.2f %s'
        for idx in df.index.values.tolist():
            note += line%(df.loc[idx,'parameter'], df.loc[idx,'value'], params[df.loc[idx,'parameter']]['unit'])
    note+='\n-----------------------------------------------------------'
    print(note)


def _scrape_output(args, columns=['star','numax','dnu','snr']):
    """Concatenate results
    
    Automatically concatenates and summarizes the results for all processed stars in each
    of the two submodules

    Methods
        :mod:`pysyd.utils._sort_table`
    

    """
    # Estimate outputs
    if glob.glob(os.path.join(args.params['outdir'],'**','estimates*csv')):
        df = pd.DataFrame(columns=columns)
        files_estimates = glob.glob(os.path.join(args.params['outdir'],'**','estimates*csv'))
        # get which stars have this output
        dirs = list(set([os.path.split(os.path.split(file)[0])[-1] for file in files_estimates]))
        # iterate through stars
        for i, dir in enumerate(dirs):
            list_of_files = glob.glob(os.path.join(args.params['outdir'],dir,'estimates*csv'))
            file = max(list_of_files, key=os.path.getctime)
            df_new = pd.read_csv(file)
            for column in columns:
                df.loc[i,column] = df_new.loc[0,column]
        _sort_table(df, args.params['outdir'], type='estimates')

    # Parameter outputs
    if glob.glob(os.path.join(args.params['outdir'], '**', 'global*csv')):
        df = pd.DataFrame(columns=['star'])
        files_globals = glob.glob(os.path.join(args.params['outdir'], '**', 'global*csv'))
        dirs = list(set([os.path.split(os.path.split(file)[0])[-1] for file in files_globals]))
        for i, dir in enumerate(dirs):
            df.loc[i, 'star'] = dir
            list_of_files = glob.glob(os.path.join(args.params['outdir'], dir, 'global*csv'))
            file = max(list_of_files, key=os.path.getctime)
            df_new = pd.read_csv(file)
            for p, param in enumerate(df_new.parameter.values.tolist()):
                for col, label in zip(['value','uncertainty'],['%s','%s_err']):
                    if col in df_new.columns.values.tolist():
                        df.loc[i, label%param] = df_new.loc[p, col]
        _sort_table(df, args.params['outdir'], type='global')


def _sort_table(df, outdir, type='global'):
    """Sort stars
    
    Sort full star table in ascending order for easy reading/comparisons

    Parameters
        df : pandas.DataFrame
            dataframe to fix
        type : str, ['estimates','global']
            which dataframe is being fixed and saved

    """
    df.to_csv(os.path.join(outdir,'%s.csv'%type), index=False)
    df_new = pd.read_csv(os.path.join(outdir,'%s.csv'%type))
    df_new.sort_values(by=['star'], ascending=[True], inplace=True)
    df_new.to_csv(os.path.join(outdir,'%s.csv'%type), index=False)


def _max_elements(x, y, npeaks, distance=None, exp_dnu=None):
    """N highest peaks
    
    Module to obtain the x and y values for the n highest peaks in a 2D array 

    Parameters
        x, y : numpy.ndarray, numpy.ndarray
            the series of data to identify peaks in
        npeaks : int, default=5
            the first n peaks
        distance : float, default=None
            used in :mod:`scipy.find_peaks` if not `None`
        exp_dnu : float, default=None
            if not `None`, multiplies y array by Gaussian weighting centered on `exp_dnu`

    Returns
        peaks_x, peaks_y : numpy.ndarray, numpy.ndarray
            the x & y coordinates of the first n peaks `npeaks`


    """
    xx, yy = np.copy(x), np.copy(y)
    weights = np.ones_like(yy)
    # if exp_dnu is not None, use weighting technique
    if exp_dnu is not None:
        sig = 0.35*exp_dnu/2.35482 
        weights *= np.exp(-(xx-exp_dnu)**2./(2.*sig**2))*((sig*np.sqrt(2.*np.pi))**-1.)
    yy *= weights
    # provide threshold for finding peaks
    if distance is not None and distance >= 1.0:
        peaks_idx, _ = find_peaks(yy, distance=distance)
    else:
        peaks_idx, _ = find_peaks(yy)
    # take from original (unweighted) arrays 
    px, py = x[peaks_idx], y[peaks_idx]
    # sort by n highest peaks
    s = np.argsort(py)
    peaks_y = py[s][-int(npeaks):][::-1]
    peaks_x = px[s][-int(npeaks):][::-1]
    return list(peaks_x), list(peaks_y), weights


def _return_max(x, y, exp_dnu=None,):
    """Return max
    
    Return the peak (and/or the index of the peak) in a given 2D array

    Parameters
        x, y : numpy.ndarray, numpy.ndarray
            the independent and dependent axis, respectively
        exp_dnu : Required[float]
            the expected dnu. Default value is `None`.

    Returns
        idx : int
            **New**: *always* returns the index first, followed by the corresponding peak from the x+y arrays
        xx[idx], yy[idx] : Union[int, float], Union[int, float]
            corresponding peak in the x and y arrays

    """
    xx, yy = np.copy(x), np.copy(y)
    if list(yy) != []:
        if exp_dnu is not None:
            lst = list(np.absolute(xx-exp_dnu))
            idx = lst.index(min(lst))
        else:
            lst = list(yy)
            idx = lst.index(max(lst))
    else:
        return None, np.nan, np.nan
    return idx, xx[idx], yy[idx]


def _bin_data(x, y, width, log=False, mode='mean'):
    """Bin data
    
    Bins 2D series of data

    Parameters
        x, y : numpy.ndarray, numpy.ndarray
            the x and y values of the data
        width : float
            bin width (typically in :math:`\\rm \\mu Hz`)
        log : bool, default=False
            creates equal bin sizes in logarithmic space when `True`

    Returns
        bin_x, bin_y, bin_yerr : numpy.ndarray, numpy.ndarray, numpy.ndarray
            binned arrays (and error is computed using the standard deviation)

    """
    if log:
        mi = np.log10(min(x))
        ma = np.log10(max(x))
        no = int(np.ceil((ma-mi)/width))
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
    """:math:`\\Delta\\nu`
    
    Estimates the large frequency separation using the numax scaling relation (add citation?)

    Parameters
        numax : float
            the frequency corresponding to maximum power or numax (:math:`\\rm \\nu_{max}`)

    Returns
        dnu : float
            the approximated frequency spacing or dnu (:math:`\\Delta\\nu`)

    """

    return 0.22*(numax**0.797)


def _get_results(suffixes=['_idl', '_py'], max_numax=3200.,):
    """

    Load and compare results between `SYD` and `pySYD` pipelines

    Parameters
        file_idlsyd : str
            path to ``SYD`` ensemble results
        file_pysyd : str
            path to ``pySYD`` ensemble results
        suffixes : List[str]
            extensions to use when merging the two pipeline results
        max_numax : float, optional
            maximum values to use for numax comparison

    Returns
        df : pandas.DataFrame
            pandas dataframe with merged results

    """
    pathidl = os.path.join(PACKAGEDIR,'data','syd_results.txt')
    # load in both pipeline results
    idlsyd = pd.read_csv(pathidl, skiprows=20, delimiter='|', names=get_dict('columns')['syd'])
    pathpy =  os.path.join(PACKAGEDIR,'data','pysyd_results.csv')
    pysyd = pd.read_csv(pathpy)
    # make sure they can crossmatch
    idlsyd.KIC = idlsyd.KIC.astype(str)
    pysyd.star = pysyd.star.astype(str)
    # merge measurements from syd & pySYD catalogs
    df = pd.merge(idlsyd, pysyd, left_on='KIC', right_on='star', how='inner', suffixes=suffixes)
    df = df[df['numax_smooth'] <= max_numax]
    return df


def setup_dirs(args, note='', dl_dict={}):
    """Setup pySYD directories
    
    Primarily most of pipeline.setup functionality to keep the pipeline script from
    getting too long. Still calls/downloads things in the same way: 1) info directory,
    2) input + data directory and 3) results directory.

    Parameters
        args : argparse.NameSpace
            command-line arguments
        note : str
            verbose output
        dl_dict : Dict[str,str]
            dictionary to keep track of files that need to be downloaded

    Returns
        dl_dict : Dict[str,str]
            dictionary of files to download for setup
        note : str
            updated verbose output

    Calls
        - :mod:`pysyd.utils.get_infdir`
        - :mod:`pysyd.utils.get_inpdir`
        - :mod:`pysyd.utils.get_outdir`

    """
    dl_dict, note = get_infdir(args, dl_dict, note)
    dl_dict, note = get_inpdir(args, dl_dict, note)
    note = get_outdir(args, note)
    # Download files that do not already exist
    if dl_dict:
        # downloading example data will generate output in terminal, so always include this regardless
        print('\nDownloading example data from source:')
        for infile, outfile in dl_dict.items():
            subprocess.call(['curl %s > %s' % (infile, outfile)], shell=True)
    # option to get ALL columns since only subset is included in the example
    if args.makeall:
        df_temp = pd.read_csv(args.info)
        df = pd.DataFrame(columns=get_dict('columns')['setup'])
        for col in df_temp.columns.values.tolist():
            if col in df.columns.values.tolist():
                df[col] = df_temp[col]
        df.to_csv(args.info, index=False)
        note += ' - ALL columns saved to the star info file\n'
    # verbose output
    if args.verbose:
        if note == '':
            print("\nLooks like you've probably done this\nbefore since you already have everything!\n")
        else:
            print('\nNote(s):\n%s' % note)


def get_infdir(args, dl_dict, note, source='https://raw.githubusercontent.com/ashleychontos/pySYD/master/dev/'):
    """Create info directory
    
    Parameters
        args : argparse.NameSpace
            command-line arguments
        note : str
            verbose output
        dl_dict : Dict[str,str]
            dictionary to keep track of files that need to be downloaded
        source : str
            path to pysyd source directory on github

    Returns
        dl_dict : Dict[str,str]
            dictionary of files to download for setup
        note : str
            updated verbose output

    """
    # INFO DIRECTORY
    # create info directory (INFDIR)
    if not os.path.exists(args.infdir):
        os.mkdir(args.infdir)
        note += ' - created input file directory at %s \n' % args.infdir
    # Example input files  
    # 'todo.txt' aka basic text file with list of stars to process 
    if not os.path.exists(args.todo):
        dl_dict.update({'%sinfo/todo.txt' % source : args.todo})
        note += ' - saved an example of a star list\n'
    # 'star_info.csv' aka star information file                             
    if not os.path.exists(args.info):
        dl_dict.update({'%sinfo/star_info.csv' % source : args.info})
        note += ' - saved an example for the star information file\n'
    return dl_dict, note


def get_inpdir(args, dl_dict, note, save=False, examples=['1435467','2309595','11618103'], exts=['LC','PS'],
               source='https://raw.githubusercontent.com/ashleychontos/pySYD/master/dev/'):
    """Create data (i.e. input) directory
    
    Parameters
        args : argparse.NameSpace
            command-line arguments
        note : str
            verbose output
        dl_dict : Dict[str,str]
            dictionary to keep track of files that need to be downloaded
        source : str
            path to pysyd source directory on github
        examples : List[str]
            KIC IDs for 3 example stars
        exts : List[str]
            data types to download for each star

    Returns
        dl_dict : Dict[str,str]
            dictionary of files to download for setup
        note : str
            updated verbose output

    """
    # DATA DIRECTORY
    # create data directory (INPDIR)
    if not os.path.exists(args.inpdir):
        os.mkdir(args.inpdir)
        note += ' - created data directory at %s \n' % args.inpdir
    # Example data for 3 (Kepler) stars
    for target in examples:
        for ext in exts:
            infile = '%sdata/%s_%s.txt' % (source, target, ext)
            outfile = os.path.join(args.inpdir, '%s_%s.txt' % (target, ext))
            if not os.path.exists(outfile):
                save = True
                dl_dict.update({infile : outfile})
    if save:
        note+=' - example data saved to data directory\n'
    return dl_dict, note


def get_outdir(args, note):
    """Create results directory
    
    Parameters
        args : argparse.Namespace
            command-line arguments
        note : str
            verbose output

    Returns
        note : str
            updated verbose output

    """
    # RESULTS DIRECTORY
    # create results directory (OUTDIR)
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        note += ' - results will be saved to %s\n' % args.outdir
    return note


def get_output(fun=False):
    """Print logo output

    Used within test mode when current installation is successfully tested.

    Parameters
        fun : bool, False
            if calling module for 'fun', only prints logo but doesn't test software

    """
    with open(os.path.join(PACKAGEDIR, 'data', 'test.txt'), "r") as f:
        lines = [line[:-2] for line in f.readlines()]
    if fun:
        lines = lines[:-2]
    counts = [len(line) for line in lines]
    width, height = os.get_terminal_size()[0], os.get_terminal_size()[-1]
    if height < len(lines):
        lines = lines[len(lines)-height:]
    for line in lines:
        sentence = line
        if len(sentence) > width:
            value = int(np.ceil((counts[0]-width)/2.))
            sentence = sentence[value:-value]
        print(sentence.center(width,' '))