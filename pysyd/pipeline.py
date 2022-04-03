# Import relevant, local pySYD modules
from pysyd import utils
from pysyd import plots
from pysyd.target import Target


def display(args):
    """
    
    This is experimental and meant to be helpful for developers or anyone
    wanting to contribute to ``pySYD``. Ideally this will test new ``pySYD``
    functions.
    
    Parameters
        args : argparse.Namespace
            the command line arguments
    
    .. warning::
        NOT CURRENTLY IMPLEMENTED
        
    .. note::
        use the hacky, boolean flag ``-t`` or ``--test`` instead (for now)
    
    """
    # If running from something other than command line
    args = utils.Parameters(args)


def load(args):
    """
    
    Module to load in all the relevant information and dictionaries
    that is required to run the pipeline on a given target.
    
    .. note::
        this does *not* load in a target or target data, this is purely
        information that is required to run any ``pySYD`` mode successfully
        (with the exception of ``pysyd.pipeline.setup``)

    Parameters
        args : argparse.Namespace
            the command line arguments
        star : object, optional
            pretty sure this is only used from jupyter notebook
        verbose : bool, optional
            again, this is only used if not using command line
        command : str, optional
            which of the 5 ``pysyd.pipeline`` modes to execute from
            the notebook

    Returns
        single : target.Target
            current data available for the provided target

    """
    # Load relevant pySYD parameters
    args = utils.Parameters(args)
    return args


def run(args):
    """
    
    Main function to initiate the pySYD pipeline (consecutively, not
    in parallel)

    Parameters
        args : argparse.Namespace
            the command line arguments


    """
    # Load relevant pySYD parameters
    args = utils.Parameters(args)
    # Run single batch of stars
    pipe(args.params['stars'], args)
    # check to make sure that at least one star was successfully run (i.e. there are results)  
    if args.params['verbose']:
        print(' - combining results into single csv file\n-----------------------------------------------------------\n')
    # Concatenates output into two files
    utils.scrape_output(args)


def pipe(group, args):
    """
    
    This function is called by both ``pysyd.pipeline.run`` and ``pysyd.pipeline.parallel``
    modes to initiate the ``pySYD`` pipeline for a group of stars

    Parameters
        group : List[object]
            list of stars to be processed as a group
        args : argparse.Namespace
            the command line arguments
        count : int
            the number of successful stars processed by the pipeline for a given group (default = `0`)

    Returns
        count : int
            the number of successful stars processed by ``pySYD`` for a given group of stars
    """
    # Iterate through and run stars in a given star 'group'
    for name in group:
        star = Target(name, args)
        # Makes sure a target is 'ok' before processing
        if star.ok:
            star.process_star()
        else:
            # Only print data warnings when running pySYD in regular mode (i.e. not in parallel)
            if star.mode != 'parallel':
                print(' - cannot find data for %s'%star.name)


def parallel(args):
    """
    
    Run ``pySYD`` in parallel for a large number of stars

    Parameters
        args : argparse.Namespace
            the command line arguments
    
    """
    # Import relevant (external) python modules
    import numpy as np
    import multiprocessing as mp
    # Load relevant pySYD parameters
    args = utils.Parameters(args)
    # Creates the separate, asyncrhonous (nthread) processes
    pool = mp.Pool(args.n_threads)
    result_objects = [pool.apply_async(pipe, args=(group, args)) for group in args.params['groups']]
    results = [r.get() for r in result_objects]
    pool.close()
    pool.join()               # postpones execution of the next line until all processes finish
      
    if args.params['verbose']:
        print('Combining results into single csv file.\n')
    # Concatenates output into two files
    utils.scrape_output(args)


def test(args):
    """
    
    This is experimental and meant to be helpful for developers or anyone
    wanting to contribute to ``pySYD``. Ideally this will test new ``pySYD``
    functions.
    
    Parameters
        args : argparse.Namespace
            the command line arguments
    
    .. important::
        NOT CURRENTLY IMPLEMENTED
        
    Note:
        use the hacky, boolean flag ``-t`` or ``--test`` instead (for now)
    
    """
    # Load relevant pySYD parameters
    args = utils.Parameters(args)


def setup(args, note='', raw='https://raw.githubusercontent.com/ashleychontos/pySYD/master/examples/'):
    """
    
    Running this after installation will create the appropriate directories in the current working
    directory as well as download example data and files to test your pySYD installation

    Parameters
        args : argparse.Namespace
            the command line arguments
        note : str, optional
            suppressed (optional) verbose output
        raw : str
            path to download "raw" package data and examples from the ``pySYD`` source directory

    """
    # Import relevant (external) python modules
    import os
    import subprocess
    import pandas as pd
    # downloading data will generate output in terminal, so include this statement regardless
    print('\n\nDownloading relevant data from source directory:\n')

    # create info directory
    if not os.path.exists(args.infdir):
        os.mkdir(args.infdir)
        note+=' - created input file directory: %s \n'%args.infdir

    # get example input files
    infile1 = '%sinfo/todo.txt'%raw
    outfile1 = os.path.join(args.infdir, args.todo)
    subprocess.call(['curl %s > %s'%(infile1, outfile1)], shell=True)
    infile2 = '%sinfo/star_info.csv'%raw
    outfile2 = os.path.join(args.infdir, args.info)
    subprocess.call(['curl %s > %s'%(infile2, outfile2)], shell=True)
    
    # if not successful, make empty input files for reference
    if not os.path.exists(outfile1):
        f = open(args.todo, "w")
        f.close()
    if not os.path.exists(outfile2):
        df = pd.DataFrame(columns=utils.get_dict(type='columns')['all'])
        df.to_csv(outfile2, index=False)
        
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