# Import external modules
import os
import subprocess
import pandas as pd

# Import relevant, local pySYD modules
import pysyd
from pysyd import utils
from pysyd import plots
from pysyd.target import Target


def load(args=None, star=None):
    """
    
    Module to load in all the relevant information and dictionaries
    that is required to run the pipeline on a given target.
    
    Note:
        this does *not* load in a target or target data, this is purely
        information that is required to run any ``pySYD`` mode successfully
        (with the exception of ``pysyd.pipeline.setup``)

    Args:
        args : argparse.Namespace
            the command line arguments
        star : object, optional
            pretty sure this is only used from jupyter notebook
        verbose : bool, optional
            again, this is only used if not using command line
        command : str, optional
            which of the 5 ``pysyd.pipeline`` modes to execute from
            the notebook

    Returns:
        single : target.Target
            current data available for the provided target

    """
    # If running from something other than command line
    args = utils.Parameters(args)
    if star is not None:
        args.params['stars'] = [star]
        args.add_stars()
        # TODO: ALSO ASSERT THAT LENGTH OF STAR OR STARLIST IS ONE
    assert len(args.params['stars']) == 1, "Only one star can be loaded at a time this way.\nPlease try again."
    # add star
    star = Target(args.params['stars'][0], args)
    return star


def run(args):
    """
    
    Main function to initiate the pySYD pipeline (consecutively, not
    in parallel)

    Args:
        args : argparse.Namespace
            the command line arguments

    """
    # Load in relevant information and data
    args = utils.get_info(args)
    # Run single batch of stars
    count = pipe(args.params['stars'], args)
    # check to make sure that at least one star was successfully run (i.e. there are results)  
    if count != 0:
        if args.verbose:
            print(' - combining results into single csv file')
            print('------------------------------------------------------')
            print()
        # Concatenates output into two files
        utils.scrape_output(args)


def pipe(group, args, count=0):
    """
    
    This function is called by both ``pysyd.pipeline.run`` and ``pysyd.pipeline.parallel``
    modes to initiate the ``pySYD`` pipeline for a group of stars

    Args:
        group : List[object]
            list of stars to be processed as a group
        args : argparse.Namespace
            the command line arguments
        count : int
            the number of successful stars processed by the pipeline for a given group (default = `0`)

    Returns:
        count : int
            the number of successful stars processed by ``pySYD`` for a given group of stars

    """
    # Iterate through and run stars in a given star 'group'
    for each in group:
        star = Target()
        if star.assign_star(each):
        # Makes sure target data successfully loaded in
            count+=1
            single.run_syd()
    # Number of successfully processed stars
    return count


def parallel(args):
    """
    
    Run ``pySYD`` in parallel for a large number of stars

    Parameters:
        args : argparse.Namespace
            the command line arguments
    

    """
    # Load in relevant information and data
    args = utils.get_info(args)
    # Import relevant (external) python modules
    import numpy as np
    import multiprocessing as mp
    # Creates the separate, asyncrhonous (nthread) processes
    pool = mp.Pool(args.n_threads)
    result_objects = [pool.apply_async(pipe, args=(group, args)) for group in args.params['groups']]
    results = [r.get() for r in result_objects]
    pool.close()
    pool.join()               # postpones execution of the next line until all processes finish
    count = np.sum(results)
      
    # check to make sure that at least one star was successful (count == the number of successfully processed stars)   
    if count != 0:
        if args.verbose:
            print('Combining results into single csv file.')
            print()
        # Concatenates output into two files
        utils.scrape_output(args)


#####################################################################
# Testing new pySYD functionalities
# GOAL: unittest +/- testing developments 
# NOT CURRENTLY IMPLEMENTED 
# -> use the hacky, boolean flag -t or --test instead (for now)

def Test(args):
    """
    
    This is experimental and meant to be helpful for developers or anyone
    wanting to contribute to ``pySYD``. Ideally this will test new ``pySYD``
    functions.
    
    Parameters:
        args : argparse.Namespace
            the command line arguments
    
    Warning:
        NOT CURRENTLY IMPLEMENTED
        
    Note:
        use the hacky, boolean flag ``-t`` or ``--test`` instead (for now)
    
    """
    # Load in relevant information and data
    args = load(args)
    dnu_comparison()


def setup(args):
    pass


class Setup:
    """
    
    Running this after installation will create the appropriate directories in the current working
    directory as well as download example data and files to test your pySYD installation

    Attributes:
        dir : str
            path to save setup files and directories to
        new : bool
            setup 'new' path in pysyd init file
        note : str, optional
            suppressed verbose output
        raw : str
            path to ``pySYD`` source directory to download package data and examples


    """
    def __init__(self, args, note='', raw='https://raw.githubusercontent.com/ashleychontos/pySYD/master/examples/'):
        """


        Args:
            args : argparse.Namespace
                the command line arguments
            note : str, optional
                suppressed (optional) verbose output
            raw : str
                path to download "raw" package data and examples from the ``pySYD`` source directory

        """
        self.dir, self.note, self.raw = args.dir, note, raw
        if args.new:
            self.change_init()
        self.create_dirs()
        self.retrieve_info()
        self.retrieve_data()
        if args.verbose:
            print(self.note)

    def create_dirs(self, dirs=['info', 'data', 'results']):
        """

        Creates the three main directories for pySYD to operate smoothly:
         #. info
         #. data
         #. results

        Args:
            dirs : List[str]
                list of directories to create (if they do not already exist)

        """
        for dir in dirs:
            if not os.path.exists(os.path.join(self.dir,dir)):
                os.mkdir(os.path.join(self.dir,dir))
                self.note += ' - created %s directory\n'%dir
            else:
                self.note += ' - %s directory already exists (skipping)\n'%dir


    def retrieve_info(self, files=['info/todo.txt', 'info/star_info.csv']):
    
        # get example input files
        for file in files:
            infile = '%s%s'%(self.raw, file)
            outfile = os.path.join(self.dir, os.path.split(file)[0], os.path.split(file)[-1])
            subprocess.call(['curl %s > %s'%(infile, outfile)], shell=True)
        # if not successful, make empty input files for reference
        if not os.path.exists(outfile1):
            f = open(args.todo, "w")
            f.close()
        if not os.path.exists(outfile2):
            df = pd.DataFrame(columns=utils.get_dict(type='columns')['csv'])
            df.to_csv(args.info, index=False)


    def retrieve_data(self, targets=['1435467', '2309595', '11618103'], extensions=['LC', 'PS']):

        # downloading data will generate output in terminal, so include this statement regardless
        print('\n\nDownloading relevant data from source directory:')
        # get example data
        for target in targets:
            for ext in extensions:
                infile = '%sdata/%s_%s.txt'%(self.raw, target, ext)
                outfile = os.path.join(self.dir, '%s_%s.txt'%(target, ext))
                subprocess.call(['curl %s > %s'%(infile, outfile)], shell=True)
        print('\n')
        self.note+=' - example data saved\n'


    def change_init(self, match='_ROOT', idx=None):

        fname = os.path.abspath(pysyd.__file__)
        with open(fname, "r") as f:
            lines = f.readlines()
        for l, line in enumerate(lines):
            if line.startswith(match):
                idx=l
        if idx is not None:
            new_lines = lines[:idx] + ['%s = os.path.abspath(%s)'%(match, self.dir)] + lines[idx+1:]
            with open(fname, "w") as f:
                for line in new_lines:
                    f.write(line)
