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


def setup(args, raw='https://raw.githubusercontent.com/ashleychontos/pySYD/master/dev/'):
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
    note, save, dl = '', False, {}

    # INFO DIRECTORY
    # create info directory (INFDIR)
    if not os.path.exists(args.infdir):
        os.mkdir(args.infdir)
        note+=' - created input file directory at %s \n'%args.infdir
    # example input files   
    outfile1 = os.path.join(args.infdir, args.todo)               # example star list file
    if not os.path.exists(outfile1):
        dl.update({'%sinfo/todo.txt'%raw:outfile1})
        note+=' - saved an example of a star list\n'                               
    outfile2 = os.path.join(args.infdir, args.info)               # example star info file
    if not os.path.exists(outfile2):
        dl.update({'%sinfo/star_info.csv'%raw:outfile2})
        note+=' - saved an example for the star information file\n'

    # DATA DIRECTORY
    # create data directory (INPDIR)
    if not os.path.exists(args.inpdir):
        os.mkdir(args.inpdir)
        note+=' - created data directory at %s \n'%args.inpdir
    # example data
    for target in ['1435467', '2309595', '11618103']:
        for ext in ['LC', 'PS']:
            infile='%sdata/%s_%s.txt'%(raw, target, ext)
            outfile=os.path.join(args.inpdir, '%s_%s.txt'%(target, ext))
            if not os.path.exists(outfile):
                save=True
                dl.update({infile:outfile})
    if save:
        note+=' - example data saved to data directory\n'

    # RESULTS DIRECTORY
    # create results directory (OUTDIR)
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        note+=' - results will be saved to %s\n'%args.outdir

    # Download files that do not already exist
    if dl:
        # downloading example data will generate output in terminal, so always include this regardless
        print('\nDownloading relevant data from source directory:')
        for infile, outfile in dl.items():
            subprocess.call(['curl %s > %s'%(infile, outfile)], shell=True)

    # option to get ALL columns since only subset is included in the example
    if args.makeall:
        df_temp = pd.read_csv(outfile2)
        df = pd.DataFrame(columns=utils.get_dict('columns')['setup'])
        for col in df_temp.columns.values.tolist():
            if col in df.columns.values.tolist():
                df[col] = df_temp[col]
        df.to_csv(outfile2, index=False)
        note+=' - ALL columns saved to the star info file\n'

    if args.verbose:
        if note == '':
            print("\nLooks like you've probably done this\nbefore since you already have everything!\n")
        else:
            print('\nNote(s):\n%s'%note)


def test(args, stars=['1435467', '2309595', '11618103'], answers={}):
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
    # Load in example configurations + answers to compare to
    defaults = utils.get_dict(type='tests')
    # Save defaults to file to reproduce identical results
    df = pd.read_csv(os.path.join(args.infdir, args.info))
    targets = [str(each) for each in df.stars.values.tolist()]
    for star in stars:
        answers.update({star:defaults[star].pop('results')})
        idx = targets.index(star)
        for key in defaults[star]:
            df.loc[idx,key] = defaults[star][key]
    df.to_csv(os.path.join(args.infdir, args.info), index=False)
    # Run pysyd on 3 examples with sampling
    subprocess.call(['pysyd run --mc 200'], shell=True)
    # Compare results
    final_df = pd.read_csv(os.path.join(args.outdir,'global.csv'))
    print('KIC %s\n%s\n'%(star, '-'*(len(star)+4)))        

    print('####################################################################\n#                                                                  #\n#                   TESTING SUCCESSFULLY COMPLETED                 #\n#                                                                  #\n####################################################################\n')
