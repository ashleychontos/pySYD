import os
import shutil
import unittest
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp

import pysyd
from pysyd import utils
from pysyd.target import Target



def run(args, count=0, CLI=True):
    """
    Main script to run the pySYD pipeline

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments

    """

    single = load(args, CLI=CLI)
    if single.ps:
        count+=1
        single.run_syd()
    # check to make sure that at least one star was successful (count == the number of successfully processed stars)   
    if count != 0:
        if args.verbose:
            print('Combining results into single csv file.')
            print()
        # Concatenates output into two files
        utils.scrape_output(args)


def load(args, star=None, CLI=True):
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
    if star is None:
        assert len(args.stars) == 1, "You can only load in data for one star at a time."
    else:
        args.stars = [star]

    args = utils.get_info_all(args, parallel=False, CLI=CLI)
    single = Target(args.stars[0], args)

    return single


def parallel(args, CLI=True):
    """
    Uses multiprocessing to run the pySYD pipeline in parallel. Stars are assigned
    evenly to `groups`, which is set by the number of threads or CPUs available.
    Stars will then run one-by-one per group

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments

    """

    args = utils.get_info_all(args, parallel=True, CLI=CLI)

    # create the separate, asyncrhonous (nthread) processes
    pool = mp.Pool(args.n_threads)
    result_objects = [pool.apply_async(pipe, args=(group, args)) for group in args.params['groups']]
    results = [r.get() for r in result_objects]
    pool.close()
    pool.join()    # postpones execution of the next line until all processes finish
    count = np.sum(results)
      
    # check to make sure that at least one star was successful (count == the number of successfully processed stars)   
    if count != 0:
        if args.verbose:
            print('Combining results into single csv file.')
            print()
        # Concatenates output into two files
        utils.scrape_output(args)


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

    for star in group:
        load(args, star=star, CLI=CLI)
        single = Target(star, args)
        if single.ps:
            count+=1
            single.run_syd()
    return count


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
        df = pd.DataFrame(columns=utils.get_data_columns(type='csv'))
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