# Global packages
import os
import argparse

# Local packages
from target import Target
from plots import set_plot_params
from utils import get_info, scrape_output


def main(args, parallel=False, nthreads=None):
    """Runs the SYD-PYpline.

    Parameters
    ----------
    args : argparse.Namespace
        the command line arguments
    parallel : bool
        if true will run the pipeline on multiple threads. Default value is `False`. TODO: Currently not supported!
    nthreads : Optional[int]
        the number of threads to run the pipeline on if parallel processing is enabled. Default value is `None`.
    """

    args = get_info(args)
    set_plot_params()

    for target in args.params['todo']:
        args.target = target
        Target(args)

    if args.verbose:
        print('Combining results into single csv file.')
        print()

    # Concatenates output into a two files
    scrape_output()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Python version of asteroseismic 'SYD'
                    pipeline ( Huber+2009). This script will initialize
                    the SYD-PY pipeline, which is broken up into two main modules:
                    1) find excess (findex)
                    2) fit background (fitbg).
                    By default, both modules will run unless otherwise specified.
                    See -excess and -fitbg for more details.
                    SYD-PY is actively being developed at
                    https://github.com/ashleychontos/SYD-PYpline
                    by: Ashley Chontos (achontos@hawaii.edu)
                        Daniel Huber (huberd@hawaii.edu)
                        Maryum Sayeed
                    Please contact Ashley for more details or new ideas for
                    implementations within the package.
                    [The ReadTheDocs page is currently under construction.]"""
    )
    parser.add_argument('-t', '--t', '-target', '--target', '-targets', '--targets',
                        dest='target',
                        help='Option to specify certain targets outside of todo.txt',
                        nargs='*',
                        type=int,
                        default=None,
                        )
    parser.add_argument('-f', '--f', '-file', '--file', '-list', '--list',
                        dest='file',
                        help='Path to txt file with list of targets to run the pipeline on.',
                        type=str,
                        default='Files/todo.txt',
                        )
    parser.add_argument('-ex', '--ex', '-findex', '--findex', '-excess', '--excess',
                        dest='excess',
                        help="""Turn off the find excess module. This is only recommended when a list
                             of numaxes or a list of stellar parameters (to estimate the numaxes)
                             are provided. Otherwise the second module, which fits the background
                            will not be able to run properly.""",
                        default=True, 
                        action='store_false',
    )
    parser.add_argument('-bg', '--bg', '-fitbg', '--fitbg', '-background', '--background',
                        dest='background',
                        help="""Turn off the background fitting process (although this is not recommended).
                             Asteroseismic estimates are typically unreliable without properly removing
                             stellar contributions from granulation processes. Since this is the money
                             maker, fitbg is set to 'True' by default.""",
                        default=True, 
                        action='store_false',
    )
    parser.add_argument('-filter', '--filter', '-smooth', '--smooth',
                        dest='filter',
                        help='Box filter width [muHz] for the power spectrum (Default = 2.5 muHz)',
                        default=2.5,
    )
    parser.add_argument('-kc', '--kc', '-keplercorr', '--keplercorr',
                        dest='keplercorr',
                        help='Turn on Kepler short-cadence artefact corrections',
                        default=False, 
                        action='store_true',
    )
    parser.add_argument('-v', '--v', '-verbose', '--verbose',
                        dest='verbose',
                        help="""Turn on the verbose output. Please note: the defaults is 'False'.""",
                        default=False, 
                        action='store_true',
    )
    parser.add_argument('-show', '--show', '-plot', '--plot', '-plots', '--plots',
                        dest='show',
                        help="""Shows the appropriate output figures in real time. If the findex module is
                             run, this will show one figure at the end of findex. If the fitbg module is
                             run, a figure will appear at the end of the first iteration. If the monte
                             carlo sampling is turned on, this will provide another figure at the end of
                             the MC iterations. Regardless of this option, the figures will be saved to
                             the output directory. If running more than one target, this is not
                             recommended. """,
                        default=False, 
                        action='store_true',
    )
    parser.add_argument('-mc', '--mc', '-mciter', '--mciter', 
                        dest='mciter', 
                        help='Number of MC iterations (Default = 1)',
                        default=1, 
                        type=int,
    )

    main(parser.parse_args())
