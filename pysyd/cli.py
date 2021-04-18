import os
import argparse

import pysyd
from pysyd import pipeline
from pysyd import TODODIR, INFODIR, INPDIR, OUTDIR

def main():
    # Properties inherent to both modules
    parser = argparse.ArgumentParser(
                                     description="pySYD: Automated Extraction of Global Asteroseismic Parameters", 
                                     prog='pySYD',
    )
    parser.add_argument('-version', '--version',
                        action='version',
                        version="%(prog)s {}".format(pysyd.__version__),
                        help="Print version number and exit."
    )

    sub_parser = parser.add_subparsers(title='subcommands', dest='subcommand')

    # In the parent parser, we define arguments and options common to all subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-file', '--file', '-list', '--list', '-todo', '--todo',
                               dest='file',
                               help="""Path to txt file that contains the list of targets to process (default='info/todo.txt')""",
                               type=str,
                               default=TODODIR,
    )
    parent_parser.add_argument('-in', '--in', '-input', '--input', '-inpdir', '--inpdir', 
                               dest='inpdir',
                               help='Path to input data',
                               default=INPDIR,
    )
    parent_parser.add_argument('-info', '--info', '-information', '--information',
                               dest='info',
                               help='Path to csv containing star information',
                               type=str,
                               default=INFODIR,
    )
    parent_parser.add_argument('-verbose', '--verbose',
                               dest='verbose',
                               help='Turn on verbose output (default=False)',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('-out', '--out', '-outdir', '--outdir', '-output', '--output',
                               dest='outdir',
                               help='Path to save results to',
                               default=OUTDIR,
    )

    # Setting up
    parser_setup = sub_parser.add_parser('setup', parents=[parent_parser],
                                         description='Easy setup for directories and files')
    parser_setup.set_defaults(func=pipeline.setup)

    # Run pySYD
    parser_run = sub_parser.add_parser('run', parents=[parent_parser],
                                       description='Run pySYD')
    parser_run.add_argument('-bg', '--bg', '-fitbg', '--fitbg', '-background', '--background',
                            dest='background',
                            help='Turn off the background fitting process (although this is not recommended)',
                            default=True, 
                            action='store_false',
    )
    parser_run.add_argument('-ex', '--ex', '-findex', '--findex', '-excess', '--excess',
                            dest='excess',
                            help='Turn off the find excess module. This is only recommended when a list of numaxes or a list of stellar parameters (to estimate the numaxes) are provided.',
                            default=True, 
                            action='store_false',
    )
    parser_run.add_argument('-kc', '--kc', '-keplercorr', '--keplercorr',
                            dest='keplercorr',
                            help='Turn on Kepler short-cadence artefact corrections',
                            default=False, 
                            action='store_true',
    )
    parser_run.add_argument('-nt', '--nt', '-nthread', '--nthread', '-nthreads', '--nthreads',
                            dest='n_threads',
                            help='Number of processes to run in parallel',
                            type=int,
                            default=0,
    )
    parser_run.add_argument('-par', '--par', '-parallel', '--parallel',
                            dest='parallel',
                            help='Run batch of stars in parallel',
                            default=False,
                            action='store_true',
    )
    parser_run.add_argument('-save', '--save', 
                            dest='save',
                            help='Save output files and figures (default=True)',
                            default=True, 
                            action='store_false',
    )
    parser_run.add_argument('-show', '--show',
                            dest='show',
                            help="""Shows output figures (default=False) Please note: If running multiple targets, this is not recommended! """,
                            default=False, 
                            action='store_true',
    )
    parser_run.add_argument('-star', '--star', '-stars', '--stars',
                            dest='star',
                            help="""List of targets to process (default=None). If this is not provided, it will default to read targets in from the default 'file' argument.""",
                            nargs='*',
                            type=int,
                            default=None,
    )

    # CLI relevant for finding power excess
    excess = parser_run.add_argument_group('excess')

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
                        dest='lower_x',
                        help='Lower limit of power spectrum to use in findex module (default=10.0muHz)',
                        nargs='*',
                        default=None,
                        type=float,
    )
    excess.add_argument('-ux', '--ux', '-upperx', '--upperx', 
                        dest='upper_x',
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
                        dest='n_trials',
                        help='Number of trials to estimate numax (default=3)',
                        default=3, 
                        type=int,
    )

    # CLI relevant for background fitting
    background = parser_run.add_argument_group('background')

    background.add_argument('-iw', '--iw', '-width', '--width', '-indwidth', '--indwidth',
                            dest='ind_width', 
                            help='Number of independent points to use for binning of power spectrum (default=50)',
                            default=50, 
                            type=int,
    )
    background.add_argument('-bf', '--bf', '-box', '--box', '-boxfilter', '--boxfilter',
                            dest='box_filter',
                            help='Box filter width [in muHz] for plotting the power spectrum (default=2.5muHz).',
                            default=2.5,
                            type=float,
    )
    background.add_argument('-dnu', '--dnu',
                            dest='dnu',
                            help='Brute force method to provide value for dnu',
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
                            dest='lower_b',
                            help='Lower limit of power spectrum to use in fitbg module (default=None). Please note: unless numax is known, it is not suggested to fix this beforehand.',
                            nargs='*',
                            default=None,
                            type=float,
    )
    background.add_argument('-ub', '--ub', '-upperb', '--upperb', 
                            dest='upper_b',
                            help='Upper limit of power spectrum to use in fitbg module (default=None). Please note: unless numax is known, it is not suggested to fix this beforehand.',
                            nargs='*',
                            default=None,
                            type=float,
    )
    background.add_argument('-mc', '--mc', '-iter', '--iter', '-mciter', '--mciter', 
                            dest='mc_iter', 
                            help='Number of MC iterations (default=1)',
                            default=1, 
                            type=int,
    )
    background.add_argument('-peak', '--peak', '-peaks', '--peaks', '-npeaks', '--npeaks', 
                            dest='n_peaks', 
                            help='Number of peaks to fit in ACF (default=5)',
                            default=5, 
                            type=int,
    )
    background.add_argument('-rms', '--rms', '-nrms', '--nrms', 
                            dest='n_rms', 
                            help='Number of points used to estimate amplitudes of individual background components (default=20).',
                            default=20, 
                            type=int,
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
    background.add_argument('-samples', '--samples',
                            dest='samples',
                            help='Save samples from monte carlo sampling (i.e. if mciter > 1)',
                            default=False, 
                            action='store_true',
    )
    background.add_argument('-ce', '--ce', '-clipech', '--clipech',
                            dest='clip_ech',
                            help='Disable the automatic clipping of high peaks in the echelle diagram',
                            default=True, 
                            action='store_false',
    )
    background.add_argument('-cv', '--cv', '-value', '--value',
                            dest='clip_value',
                            help='Clip value for echelle diagram (i.e. if clip=True). If none is provided, it will cut at 3x the median value of the folded power spectrum.',
                            default=None, 
                            type=float,
    )
    background.add_argument('-se', '--se', '-smoothech', '--smoothech',
                            dest='smooth_ech',
                            help='Option to smooth the echelle diagram output using a box filter [muHz]',
                            default=None,
                            type=float,
    )
    background.add_argument('-ie', '--ie', '-interpech', '--interpech',
                            dest='interp_ech',
                            help='Turn on the bilinear interpolation for the echelle diagram (default=False).',
                            default=False,
                            action='store_true',
    )

    parser_run.set_defaults(func=pipeline.main)

    args = parser.parse_args()
    args.func(args)



if __name__ == '__main__':

    main()