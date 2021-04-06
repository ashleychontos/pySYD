import os
import sys
import argparse

# for packaging purposes
#from syd.target import Target
#from syd.plots import set_plot_params
#from syd.utils import get_info, scrape_output
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

    for target in args.params['todo']:
        args.target = target
        Target(args)

    if args.verbose and len(args.params['todo']) > 1:
        print('Combining results into single csv file.')
        print()

    # Concatenates output into a two files
    scrape_output()


if __name__ == '__main__':

    # Properties inherent to both modules
    parser = argparse.ArgumentParser( 
                                     description='SYD-PYpline',
    )
    parser.add_argument('-bg', '--bg', '-fitbg', '--fitbg', '-background', '--background',
                        dest='background',
                        help='Turn off the background fitting process (although this is not recommended)',
                        default=True, 
                        action='store_false',
    )
    parser.add_argument('-ex', '--ex', '-findex', '--findex', '-excess', '--excess',
                        dest='excess',
                        help='Turn off the find excess module. This is only recommended when a list of numaxes or a list of stellar parameters (to estimate the numaxes) are provided.',
                        default=True, 
                        action='store_false',
    )
    parser.add_argument('-file', '--file', '-list', '--list',
                        dest='file',
                        help="""Path to txt file that contains the list of targets to process (default='Files/todo.txt')""",
                        type=str,
                        default='%s/info/todo.txt'%os.path.abspath('/'.join(__file__.split('/')[:-1])),
                        )
    parser.add_argument('-info', '--info', '-information', '--information',
                        dest='info',
                        help='Path to csv containing star information',
                        type=str,
                        default='%s/info/star_info.csv'%os.path.abspath('/'.join(__file__.split('/')[:-1])),
    )
    parser.add_argument('-kc', '--kc', '-keplercorr', '--keplercorr',
                        dest='keplercorr',
                        help='Turn on Kepler short-cadence artefact corrections',
                        default=False, 
                        action='store_true',
    )
    parser.add_argument('-save', '--save', 
                        dest='save',
                        help='Save output files and figures (default=True)',
                        default=True, 
                        action='store_false',
    )
    parser.add_argument('-show', '--show',
                        dest='show',
                        help="""Shows output figures (default=False) Please note: If running multiple targets, this is not recommended! """,
                        default=False, 
                        action='store_true',
    )
    parser.add_argument('-target', '--target', '-targets', '--targets',
                        dest='target',
                        help="""List of targets to process (default=None). If this is not provided, it will default to read targets in from the default 'file' argument ('Files/todo.txt')""",
                        nargs='*',
                        type=int,
                        default=None,
                        )
    parser.add_argument('-v', '--v', '-verbose', '--verbose',
                        dest='verbose',
                        help='Turn on verbose output (default=False)',
                        default=False, 
                        action='store_true',
    )

    # CLI relevant for finding power excess
    excess = parser.add_argument_group('excess')

    excess.add_argument('-bin', '--bin', '-binning', '--binning', 
                        dest='binning', 
                        help='Look up to be sure (default=0.005)',
                        default=0.005, 
                        type=float,
    )
    excess.add_argument('-sw', '--sw', '-smoothwidth', '--smoothwidth',
                        dest='smooth_width',
                        help='Box filter width [muHz] for the power spectrum (default=1.5muHz)',
                        default=1.5,
                        type=float,
    )
    excess.add_argument('-lx', '--lx', '-lowerx', '--lowerx', 
                        dest='lowerx',
                        help='Lower limit of power spectrum to use in findex module (default=10.0muHz)',
                        nargs='*',
                        default=None,
                        type=float,
    )
    excess.add_argument('-ux', '--ux', '-upperx', '--upperx', 
                        dest='upperx',
                        help='Upper limit of power spectrum to use in findex module (default=4000.0muHz)',
                        nargs='*',
                        default=None,
                        type=float,
    )
    excess.add_argument('-step', '--step', '-steps', '--steps', 
                        dest='step', 
                        help='Look up to be sure (default=0.25)',
                        default=0.25,
                        type=float, 
    )
    excess.add_argument('-trials', '--trials', '-ntrials', '--ntrials',
                        dest='ntrials',
                        help='Number of trials to estimate numax (default=3)',
                        default=3, 
                        type=int,
    )

    # CLI relevant for background fitting
    background = parser.add_argument_group('background')

    background.add_argument('-iw', '--iw', '-width', '--width', '-indwidth', '--indwidth',
                            dest='ind_width', 
                            help='Number of independent points to use for binning of power spectrum (default=50)',
                            default=50, 
                            type=int,
    )
    background.add_argument('-bf', '--bf', '-box', '--box', '-boxfilter', '--boxfilter',
                            dest='box_filter',
                            help='Box filter width [in muHz] for plotting the power spectrum (default=2.5muHz) Please note: this is **NOT** used in any subsequent analyses',
                            default=2.5,
                            type=float,
    )
    background.add_argument('-clip', '--clip',
                            dest='clip',
                            help='Disable the automatic clipping of high peaks in the echelle diagram',
                            default=True, 
                            action='store_false',
    )
    background.add_argument('-dnu', '--dnu',
                            dest='dnu',
                            help='Brute force method to provide value for dnu.',
                            nargs='*',
                            type=float,
                            default=None, 
    )
    background.add_argument('-numax', '--numax',
                            dest='numax',
                            help='Brute force method to bypass findex and provide value for numax. Please note: len(args.numax) == len(args.targets) for this to work! This is mostly intended for single star runs.',
                            nargs='*',
                            type=float,
                            default=None,
    )
    background.add_argument('-lb', '--lb', '-lowerb', '--lowerb', 
                            dest='lowerb',
                            help='Lower limit of power spectrum to use in fitbg module (default=None) Please note: unless numax is known, it is not suggested to fix this beforehand.',
                            nargs='*',
                            default=None,
                            type=float,
    )
    background.add_argument('-mc', '--mc', '-iter', '--iter', '-mciter', '--mciter', 
                            dest='mciter', 
                            help='Number of MC iterations (default=1)',
                            default=1, 
                            type=int,
    )
    background.add_argument('-peak', '--peak', '-peaks', '--peaks', '-npeaks', '--npeaks', 
                            dest='npeaks', 
                            help='Number of peaks to fit in ACF (default=5)',
                            default=5, 
                            type=int,
    )
    background.add_argument('-rms', '--rms', '-nrms', '--nrms', 
                            dest='nrms', 
                            help='Number of points used to estimate amplitudes of individual background components (default=20).',
                            default=20, 
                            type=int,
    )
    background.add_argument('-samples', '--samples',
                            dest='samples',
                            help='Save samples from monte carlo sampling (i.e. if mciter > 1)',
                            default=False, 
                            action='store_true',
    )
    background.add_argument('-se', '--se', '-smoothech', '--smoothech',
                            dest='smooth_ech',
                            help='Option to smooth the echelle diagram output [muHz]',
                            default=None,
                            type=float,
    )
    background.add_argument('-slope', '--slope',
                            dest='slope',
                            help='When true, this will correct for residual slope in a smoothed power spectrum before estimating numax',
                            default=False, 
                            action='store_true',
    )
    background.add_argument('-sp', '--sp', '-smoothps', '--smoothps',
                            dest='smooth_ps',
                            help='Box filter width [muHz] for the power spectrum (Default = 2.5 muHz)',
                            default=2.5,
                            type=float,
    )
    background.add_argument('-ub', '--ub', '-upperb', '--upperb', 
                            dest='upperb',
                            help='Upper limit of power spectrum to use in fitbg module (default=None) Please note: unless numax is known, it is not suggested to fix this beforehand.',
                            nargs='*',
                            default=None,
                            type=float,
    )
    background.add_argument('-value', '--value',
                            dest='value',
                            help='Clip value for echelle diagram (i.e. if clip=True). If none is provided, it will cut at 3x the median value of the folded power spectrum.',
                            default=None, 
                            type=float,
    )

    main(parser.parse_args())
