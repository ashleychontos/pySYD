# Import relevant, local pySYD modules
import pysyd
from pysyd import utils
from pysyd import plots
from pysyd.target import Target


#####################################################################
# Main function that assigns functions for different pySYD modes 
#

def main(args=None):
    """
    Main script to run the pySYD pipeline

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments

    """
    # Setup operates quite differently from the other pySYD modes
    if args.command == 'setup':
        setup(args)
    else:
        # Load in relevant information and data
        args = load(args)
        if args.command == 'run':
            run(args)
        elif args.command == 'load':
            pass
        elif args.command == 'parallel':
            parallel(args)
        elif args.command == 'test':
            test(args)
        else:
            pass


#####################################################################
# Loads in information and parameters relevant to all pySYD modes
# (except for setup)
# NOTE: this does not load in a target or target data, this is
#       purely information for running any star successfully
#


def load(args, star=None, verbose=False, command='run'):
    """
    A Target class is initialized and processed for each star in the stargroup.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments

    Returns
    -------
    single : target.Target
        current data available for the provided target

    """
    if args is None:
        args = utils.Constants()
        args.cli = False
        if star is None:
            print("If using this method, please provide the 'star' keyword argument")
            print("(i.e. the star to be processed) and try again.")
            return
        else:
            args.stars = [star]
    # Load relevant pySYD parameters
    args = utils.get_info(args)
    return args


#####################################################################
# Run main pySYD pipeline (consecutively, not in parallel)
#
    

def run(args):
    """
    Main script to run the pySYD pipeline

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments

    """
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


#####################################################################
# Pipe is called by both 'run' and 'parallel' modes to initiate the
# pySYD pipeline for a group of stars
# i.e. consecutively for the 'run' command
#      and multiprocessing in 'parallel' mode
#


def pipe(group, args, count=0):
    """
    A Target class is initialized and processed for each star in the stargroup.

    Parameters
    ----------
    group : 
        list of stars to be processed as a group
    args : argparse.Namespace
        the command line arguments

    Returns
    -------
    count : int
        the number of successful stars processed by pySYD for a given group of stars

    """
    # Iterate through and run stars in a given star 'group'
    for star in group:
        single = Target(star, args)
        # Makes sure a target has a power spectrum before processing
        if hasattr(single, 'ps'):
            count+=1
            single.run_syd()
        else:
            # Only print data warnings when running pySYD in regular mode (i.e. not in parallel)
            if args.command != 'parallel':
                print(' - cannot find data for %s'%single.name)
    # Number of successfully processed stars
    return count


#####################################################################
# Run pySYD pipeline in parallel for a large number of stars
#


def parallel(args):
    """
    Uses multiprocessing to run the pySYD pipeline in parallel. Stars are assigned
    evenly to `groups`, which is set by the number of threads or CPUs available.
    Stars will then run one-by-one per group

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments

    """
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


def test(args):

    dnu_comparison()


#####################################################################
# Downloads examples and sets up local directories for a quickstart
#


def setup(args, note='', raw='https://raw.githubusercontent.com/ashleychontos/pySYD/master/examples/'):
    """
    Running this after installation will create the appropriate directories in the current working
    directory as well as download example data and files to test your pySYD installation

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    note : Optional[str]
        suppressed (optional verbose output
    path_info : str
        path where star list and information are saved. Default is 'info/'
    raw : str
        path to download package data and examples from (source directory)

    """
    # Import relevant (external) python modules
    import pandas as pd
    import os, subprocess
    # downloading data will generate output in terminal, so include this statement regardless
    print('\n\nDownloading relevant data from source directory:\n')
    # create info directory
    if len(os.path.split(args.todo)) != 1:
        path_info = os.path.split(args.todo)[0]
        if not os.path.exists(path_info):
            os.mkdir(path_info)
            note+=' - created input file directory: %s \n'%path_info
    print(path_info)
    # get example input files
    infile1='%sinfo/todo.txt'%raw
    outfile1=os.path.join(path_info,'info.txt')
    subprocess.call(['curl %s > %s'%(infile1, outfile1)], shell=True)
    infile2='%sinfo/star_info.csv'%raw
    outfile2=os.path.join(path_info,os.path.split(args.info)[-1])
    subprocess.call(['curl %s > %s'%(infile2, outfile2)], shell=True)
    # if not successful, make empty input files for reference
    if not os.path.exists(outfile1):
        f = open(args.todo, "w")
        f.close()
    if not os.path.exists(outfile2):
        df = pd.DataFrame(columns=utils.get_dict(type='columns')['csv'])
        df.to_csv(args.info, index=False)
    # create data directory
    if not os.path.exists(args.inpdir):
        os.mkdir(args.inpdir)
        note+=' - created data directory at %s \n'%args.inpdir
    # get example data
    for target in ['1435467', '2309595', '11618103']:
        for ext in ['LC', 'PS']:
            infile='%sdata/%s_%s.txt'%(raw, target, ext)
            outfile=os.path.join(args.inpdir, '%s_%s.txt'%(target, ext))
            subprocess.call(['curl %s > %s'%(infile, outfile)], shell=True)
    print('\n')
    note+=' - example data saved\n'
    # create results directory
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    note+=' - results will be saved to %s \n\n'%args.outdir
    if args.verbose:
        print(note)