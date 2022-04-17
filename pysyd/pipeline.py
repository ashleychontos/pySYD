import os
import subprocess
import pandas as pd









# Package mode
from . import utils
from . import plots
from .target import Target



def check(args):
    """
    
    This is intended to be a way to check a target before running it by plotting the
    times series data and/or power spectrum. This works in the most basic way  but has
    not been tested otherwise
    
    Parameters
        args : argparse.Namespace
            the command line arguments
    
    .. important::
        has not been extensively tested

    
    """
    star, args = load(args)
    plots._check_data(star, args)


def load(args):
    """
    
    Module to load in all relevant information and dictionaries
    required to run the pipeline
    
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
            which of the 5 ``pysyd.pipeline`` modes to execute from the notebook

    Returns
        single : target.Target
            current data available for the provided target

    Deprecated
        single : target.Target
            current data available for the provided target

    """
    if args.data:
        if args.stars is None:
            print('\nTrying to check data but no target provided.\nPlease provide a star via --star and try again.')
            return
        else:
            assert len(args.stars) == 1, "No more than one star can be checked at a time."
        if args.verbose:
            print('\n\nChecking data for target %s:'%args.stars[0])
    display, verbose = args.plot, args.verbose
    # Load in data for a given star
    new_args = utils.Parameters(args)
    star = Target(args.stars[0], new_args)
    star.params['show'], star.params['verbose'] = args.plot, args.verbose
    return star, args


def parallel(args):
    """
    
    Run ``pySYD`` in parallel for a large number of stars

    Parameters
        args : argparse.Namespace
            the command line arguments

    Methods
        pipe

    .. seealso:: :mod:`pysyd.pipeline.run`
    
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


def pipe(group, args, progress=False):
    """

    This function is called by both :mod:`pysyd.pipeline.run` and :mod:`pysyd.pipeline.parallel`
    to initialize the pipeline for a `'group'` of stars

    Parameters
        group : List[str]
            list of stars to be processed as a group
        args : argparse.Namespace
            the command line arguments

    """
    if len(group) > 1 and args.params['mode'] == 'run' and not args.params['verbose']:
        progress=True
        print("\nProcessing %d stars:"%len(group))
        from tqdm import tqdm 
        pbar = tqdm(total=len(group))
    # Iterate through and run stars in a given star 'group'
    for name in group:
        star = Target(name, args)
        star.process_star()
        if progress:
            pbar.update(1)
    if progress:
        pbar.close()
        print("\n -- process complete --\n\n")


def plot(args):
    """
    
    Module to load in all relevant information and dictionaries
    required to run the pipeline
    
    .. note::
        this does *not* load in a target or target data, this is purely
        information that is required to run any ``pySYD`` mode successfully
        (with the exception of ``pysyd.pipeline.setup``)

    Parameters
        args : argparse.Namespace
            the command line arguments


    """
    if args.compare:
        plots._create_comparison_plot(show=args.show, save=args.save, overwrite=args.overwrite,)
    if args.results:
        if args.stars is None:
            raise utils.PySYDInputError("Please provide a star to plot results for")
        else:
            assert len(args.stars) == 1, "No more than one star can be checked at a time."
        assert os.path.exists(os.path.join(args.params['outdir'],args.stars[0]))
        if args.verbose:
            print('\n\nPlotting results for target %s:'%args.stars[0])


def run(args):
    """
    
    Main function to initiate the pySYD pipeline (consecutively, not
    in parallel)

    Parameters
        args : argparse.Namespace
            the command line arguments

    Methods
        pipe

    .. seealso:: :mod:`pysyd.pipeline.parallel`


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


def setup(args, note='', raw='https://raw.githubusercontent.com/ashleychontos/pySYD/master/dev/'):
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
    # downloading data will generate output in terminal, so include this statement regardless
    if args.example:
        print('\n\nDownloading relevant data from source directory:\n')

    # INFDIR
    # create info directory
    if not os.path.exists(args.infdir):
        os.mkdir(args.infdir)
        note+=' - created input file directory: %s \n'%args.infdir
    if args.files:
        # get example input files
        infile1 = '%sinfo/todo.txt'%raw
        outfile1 = os.path.join(args.infdir, args.todo)
        subprocess.call(['curl %s > %s'%(infile1, outfile1)], shell=True)
        infile2 = '%sinfo/star_info.csv'%raw
        outfile2 = os.path.join(args.infdir, args.info)
        subprocess.call(['curl %s > %s'%(infile2, outfile2)], shell=True)
        if args.makeall:
            df_temp = pd.read_csv(outfile2)
            df = pd.DataFrame(columns=utils.get_dict('columns')['all'])
            for col in df_temp.columns.values.tolist():
                if col in df.columns.values.tolist():
                    df[col] = df_temp[col]
            df.to_csv(outfile2, index=False)

    # INPDIR
    # create data directory
    if not os.path.exists(args.inpdir):
        os.mkdir(args.inpdir)
        note+=' - created data directory at %s \n'%args.inpdir
    if args.examples:
        # get example data
        for target in ['1435467', '2309595', '11618103']:
            for ext in ['LC', 'PS']:
                infile='%sdata/%s_%s.txt'%(raw, target, ext)
                outfile=os.path.join(args.inpdir, '%s_%s.txt'%(target, ext))
                subprocess.call(['curl %s > %s'%(infile, outfile)], shell=True)
        print('\n')
        note+=' - example data saved\n'
    
    # OUTDIR
    # create results directory
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    note+=' - results will be saved to %s \n\n'%args.outdir
    if args.verbose:
        print(note)


def test(args):
    """
    
    This is experimental and meant to be helpful for developers or anyone
    wanting to contribute to ``pySYD``. Ideally this will test new ``pySYD``
    functions.
    
    Parameters
        args : argparse.Namespace
            the command line arguments
    
    .. important::
        has not been extensively tested
        
    
    """
    print('####################################################################\n#                                                                  #\n#                   Testing pySYD functionality                    #\n#                                                                  #\n####################################################################\n')
    subprocess.call(['pysyd run --star %s --mc 200'%star], shell=True)
    # Load relevant pySYD parameters
    examples = utils.get_dict(type='tests')
    for star in ['1435467', '2309595', '11618103']:
        print('KIC %s\n%s\n'%(star, '-'*(len(star)+4)))        
        answers = examples[star]['results']
    print('####################################################################\n#                                                                  #\n#                   TESTING SUCCESSFULLY COMPLETED                 #\n#                                                                  #\n####################################################################\n')
