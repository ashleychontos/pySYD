import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp

import pysyd
from pysyd import utils
from pysyd.target import Target


def main(args):
    """
    Runs the pySYD pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    """

    args = utils.get_info(args)

    if args.parallel:
        # create the separate, asyncrhonous (nthread) processes
        pool = mp.Pool(args.nthreads)
        result_objects = [pool.apply_async(run, args=(group, args)) for group in args.params['stars']]
        results = [r.get() for r in result_objects]
        pool.close()
        pool.join()    # postpones execution of the next line until all processes finish
        count = np.sum(results)
    else:
        count = run(args.params['todo'], args)
      
    # check to make sure that at least one star was successful (count == number of successful star executions)   
    if count != 0:
        if args.verbose:
            print('Combining results into single csv file.')
            print()
        # Concatenates output into two files
        utils.scrape_output(args)


def run(stargroup, args, count=0):
    """
    A Target class is initialized and processed for each star in the stargroup.

    Parameters
    ----------
    stargroup : 
        list of stars to be processed as a group
    args : argparse.Namespace
        the command line arguments
    count : int
        the number of successful stars processed by pySYD for a given group of stars
    """

    for star in stargroup:
        single = Target(star, args)
        if single.run:
            count+=1
            single.run_syd()
    return count


def setup(args, note='', path_info='', raw='https://raw.githubusercontent.com/ashleychontos/pySYD/master/'):
    """
    Running this after installation will create the appropriate directories in the current working
    directory as well as download example data and files to test your installation.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    note : str
        optional verbose output
    path_info : str
        path where star list and information are saved. Default is 'info/'
    raw : str
        path to download package data and examples from (source directory)
    """

    print('\n\nDownloading relevant data from https://github.com/ashleychontos/pySYD:\n')
    # create info directory
    if len(args.file.split('/')) != 1:
        path_info += '%s/'%'/'.join(args.file.split('/')[:-1])
        if not os.path.exists(path_info):
            os.mkdir(path_info)
            note+=' - created input file directory: %s \n'%path_info

    # get example input files
    infile1='%sinfo/todo.txt'%raw
    outfile1='%s%s'%(path_info,args.file.split('/')[-1])
    subprocess.call(['curl %s > %s'%(infile1, outfile1)], shell=True)
    infile2='%sinfo/star_info.csv'%raw
    outfile2='%s%s'%(path_info,args.info.split('/')[-1])
    subprocess.call(['curl %s > %s'%(infile2, outfile2)], shell=True)

    # if not successful, make empty input files for reference
    if not os.path.exists(outfile1):
        f = open(args.file, "w")
        f.close()
    if not os.path.exists(outfile2):
        df = pd.DataFrame(columns=['stars','rad','teff','logg','numax','lowerx','upperx','lowerb','upperb','seed'])
        df.to_csv(args.info, index=False)

    # create data directory
    if not os.path.exists(args.inpdir):
        os.mkdir(args.inpdir)
        note+=' - created data directory at %s \n'%args.inpdir

    # get example data
    for target in ['1435467', '2309595', '11618103']:
        for ext in ['LC', 'PS']:
            infile='%sdata/%s_%s.txt'%(raw, target, ext)
            outfile='%s/%s_%s.txt'%(args.inpdir, target, ext)
            subprocess.call(['curl %s > %s'%(infile, outfile)], shell=True)
    print('\n')
    note+=' - example data saved\n'

    # create results directory
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    note+=' - results will be saved to %s \n\n'%args.outdir
    
    if args.verbose:
        print(note)