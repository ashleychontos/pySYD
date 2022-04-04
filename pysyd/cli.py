import os
import argparse

import pysyd
from pysyd import pipeline
from pysyd import INFDIR, INPDIR, OUTDIR



def main():

    parser = argparse.ArgumentParser(
                                     description="pySYD: automated measurements of global asteroseismic parameters", 
                                     prog='pySYD',
    )
    parser.add_argument('--version',
                        action='version',
                        version="%(prog)s {}".format(pysyd.__version__),
                        help="Print version number and exit.",
    )

    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('--file', '--list', '--todo',
                               metavar='str',
                               dest='todo',
                               help='List of stars to process',
                               type=str,
                               default='todo.txt',
    )
    parent_parser.add_argument('--in', '--input', '--inpdir', 
                               metavar='str',
                               dest='inpdir',
                               help='Input directory',
                               type=str,
                               default=INPDIR,
    )
    parent_parser.add_argument('--cli',
                               dest='cli',
                               help='Running from command line (this should not be touched)',
                               default=True,
                               action='store_true',
    )
    parent_parser.add_argument('--infdir',
                               metavar='str',
                               dest='infdir',
                               help='Path to relevant pySYD information',
                               type=str,
                               default=INFDIR,
    )
    parent_parser.add_argument('--info', '--information', 
                               metavar='str',
                               dest='info',
                               help='List of stellar parameters and options',
                               type=str,
                               default='star_info.csv',
    )
    parent_parser.add_argument('--out', '--outdir', '--output',
                               metavar='str',
                               dest='outdir',
                               help='Output directory',
                               type=str,
                               default=OUTDIR,
    )

    main_parser = argparse.ArgumentParser(add_help=False)

    main_parser.add_argument('-b', '--bg', '--background',
                             dest='background',
                             help='Turn off the routine that determines the stellar background contribution',
                             default=True,
                             action='store_false',
    )
    main_parser.add_argument('-e', '--est', '--excess',
                             dest='excess',
                             help='Turn off the optional module that estimates numax',
                             default=True,
                             action='store_false',
    )
    main_parser.add_argument('-g', '--globe', '--global',
                             dest='globe',
                             help='Turn off the main module that estimates global properties',
                             default=True,
                             action='store_false',
    )
    main_parser.add_argument('--gap', '--gaps', 
                             metavar='int',
                             dest='gap',
                             help="What constitutes a time series 'gap' (i.e. n x the cadence)",
                             type=int,
                             default=20, 
    )
    main_parser.add_argument('-k', '--kc', '--kepcorr', 
                             dest='kep_corr',
                             help='Turn on the Kepler short-cadence artefact correction routine',
                             default=False, 
                             action='store_true',
    )
    main_parser.add_argument('--of', '--over', '--oversample',
                             metavar='int',
                             dest='oversampling_factor',
                             help='The oversampling factor (OF) of the input power spectrum',
                             type=int,
                             default=None,
    )
    main_parser.add_argument('-o', '--overwrite',
                             dest='overwrite',
                             help='Overwrite existing files with the same name/path',
                             default=False, 
                             action='store_true',
    )
    main_parser.add_argument('-s', '--save',
                             dest='save',
                             help='Do not save output figures and results.',
                             default=True, 
                             action='store_false',
    )
    main_parser.add_argument('--star', '--stars',
                             metavar='str',
                             dest='stars',
                             help='List of stars to process',
                             type=str,
                             nargs='*',
                             default=None,
    )
    main_parser.add_argument('-x', '--stitch', '--stitching',
                             dest='stitch',
                             help="Correct for large gaps in time series data by 'stitching' the light curve",
                             default=False,
                             action='store_true',
    )


    excess = main_parser.add_argument_group('Estimate numax')

    excess.add_argument('-a', '--ask',
                        dest='ask',
                        help='Ask which trial to use',
                        default=False, 
                        action='store_true',
    )
    excess.add_argument('--bin', '--binning',
                        metavar='float',  
                        dest='binning', 
                        help='Binning interval for PS (in muHz)',
                        default=0.005, 
                        type=float,
    )
    excess.add_argument('--bm', '--mode', '--bmode',
                        metavar='str',
                        choices=["mean", "median", "gaussian"],
                        dest='bin_mode',
                        help='Binning mode',
                        default='mean',
                        type=str,
    )
    excess.add_argument('--lx', '--lowerx', 
                        metavar='float', 
                        dest='lower_ex',
                        help='Lower frequency limit of PS',
                        nargs='*',
                        default=None,
                        type=float,
    )
    excess.add_argument('--step', '--steps', 
                        metavar='float', 
                        dest='step', 
                        default=0.25,
                        type=float, 
    )
    excess.add_argument('--trials', '--ntrials',
                        metavar='int', 
                        dest='n_trials',
                        default=3, 
                        type=int,
    )
    excess.add_argument('--sw', '--smoothwidth',
                        metavar='float', 
                        dest='smooth_width',
                        help='Box filter width (in muHz) for smoothing the PS',
                        default=10.0,
                        type=float,
    )
    excess.add_argument('--ux', '--upperx', 
                        metavar='float', 
                        dest='upper_ex',
                        help='Upper frequency limit of PS',
                        nargs='*',
                        default=None,
                        type=float,
    )



    background = main_parser.add_argument_group('Background Fit')

    background.add_argument('--all', '--showall',
                            dest='showall',
                            help='Plot background comparison figure',
                            default=False,
                            action='store_true',
    )
    background.add_argument('--basis', 
                            metavar='str',
                            dest='basis',
                            help="Which basis to use for background fit (i.e. 'a_b', 'pgran_tau', 'tau_sigma'), *** NOT operational yet ***",
                            default='tau_sigma', 
                            type=str,
    )
    background.add_argument('--bf', '--box', '--boxfilter',
                            metavar='float', 
                            dest='box_filter',
                            help='Box filter width [in muHz] for plotting the PS',
                            default=1.0,
                            type=float,
    )
    background.add_argument('-f', '--fix', '--fixwn',
                            dest='fix_wn',
                            help='Fix the white noise level',
                            default=False,
                            action='store_true',
    )
    background.add_argument('--iw', '--indwidth',
                            metavar='float', 
                            dest='ind_width', 
                            help='Width of binning for PS [in muHz]',
                            default=20.0, 
                            type=float,
    )
    background.add_argument('--laws', '--nlaws', 
                            metavar='int', 
                            dest='n_laws', 
                            help='Force number of red-noise component(s)',
                            default=None, 
                            type=int,
    )
    background.add_argument('--lb', '--lowerb', 
                            metavar='float', 
                            dest='lower_bg',
                            help='Lower frequency limit of PS',
                            nargs='*',
                            default=None,
                            type=float,
    )
    background.add_argument('--metric', 
                            metavar='str', 
                            dest='metric', 
                            help="Which model metric to use, choices=['bic','aic']",
                            default='aic', 
                            type=str,
    )
    background.add_argument('--rms', '--nrms', 
                            metavar='int', 
                            dest='n_rms', 
                            help='Number of points to estimate the amplitude of red-noise component(s)',
                            default=20, 
                            type=int,
    )
    background.add_argument('--ub', '--upperb', 
                            metavar='float', 
                            dest='upper_bg',
                            help='Upper frequency limit of PS',
                            nargs='*',
                            default=None,
                            type=float,
    )

#####################################################################
# CLI options related to estimating numax
#

    numax = main_parser.add_argument_group('Deriving numax')

    numax.add_argument('--ew', '--exwidth',
                       metavar='float', 
                       dest='ex_width',
                       help='Fractional value of width to use for power excess, where width is computed using a solar scaling relation.',
                       default=1.0,
                       type=float,
    )
    numax.add_argument('--lp', '--lowerp', 
                       metavar='float', 
                       dest='lower_ps',
                       help='Lower frequency limit for zoomed in PS',
                       nargs='*',
                       default=None,
                       type=float,
    )
    numax.add_argument('--numax',
                       metavar='float',
                       dest='numax',
                       help='Skip find excess module and force numax',
                       nargs='*',
                       default=None,
                       type=float,
    )
    numax.add_argument('--sm', '--smpar',
                       metavar='float', 
                       dest='sm_par',
                       help='Value of smoothing parameter to estimate smoothed numax (typically between 1-4).',
                       default=None, 
                       type=float,
    )
    numax.add_argument('--up', '--upperp', 
                       metavar='float', 
                       dest='upper_ps',
                       help='Upper frequency limit for zoomed in PS',
                       nargs='*',
                       default=None,
                       type=float,
    )


    dnu = main_parser.add_argument_group('Deriving dnu')

    dnu.add_argument('--dnu',
                     metavar='float',
                     dest='dnu',
                     help='Brute force method to provide value for dnu',
                     nargs='*',
                     type=float,
                     default=None, 
    )
    dnu.add_argument('--method',
                     metavar='str',
                     dest='method',
                     help='Method to use to determine dnu, ~[M, A, D]',
                     default='D',
                     type=str,
    )
    dnu.add_argument('--peak', '--peaks', '--npeaks',
                     metavar='int', 
                     dest='n_peaks', 
                     help='Number of peaks to fit in the ACF',
                     default=5, 
                     type=int,
    )
    dnu.add_argument('--sp', '--smoothps',
                     metavar='float', 
                     dest='smooth_ps',
                     help='Box filter width [in muHz] of PS for ACF', 
                     type=float,
                     default=2.5,
    )
    dnu.add_argument('--thresh', '--threshold',
                     metavar='float', 
                     dest='threshold',
                     help='Fractional value of FWHM to use for ACF',
                     default=1.0,
                     type=float,
    )


    echelle = main_parser.add_argument_group('Echelle diagram')

    echelle.add_argument('--ce', '--cm', '--color', 
                         metavar='str',
                         dest='cmap',
                         help='Change colormap of ED, which is `binary` by default.',
                         default='binary', 
                         type=str,
    )
    echelle.add_argument('--cv', '--value',
                         metavar='float', 
                         dest='clip_value',
                         help='Clip value multiplier to use for echelle diagram (ED). Default is 3x the median, where clip_value == `3`.',
                         default=3.0, 
                         type=float,
    )
    echelle.add_argument('-y', '--hey',
                         dest='hey', 
                         help="Use Daniel Hey's plugin for echelle",
                         default=False, 
                         action='store_true',
    )
    echelle.add_argument('-i', '--ie', '--interpech',
                         dest='interp_ech',
                         help='Turn on the interpolation of the output ED',
                         default=False,
                         action='store_true',
    )
    echelle.add_argument('--le', '--lowere', 
                         metavar='float', 
                         dest='lower_ech',
                         help='Lower frequency limit of folded PS to whiten mixed modes',
                         nargs='*',
                         default=None,
                         type=float,
    )
    echelle.add_argument('-n', '--notch', 
                         dest='notching',
                         help='Use notching technique to reduce effects from mixed modes (not fully functional, creates weirds effects for higher SNR cases)',
                         default=False, 
                         action='store_true',
    )
    echelle.add_argument('--nox', '--nacross',
                         metavar='int', 
                         dest='nox',
                         help='Resolution for the x-axis of the ED',
                         default=50,
                         type=int, 
    )
    echelle.add_argument('--noy', '--ndown', '--norders',
                         metavar='int', 
                         dest='noy',
                         help='The number of orders to plot on the ED y-axis',
                         default=0,
                         type=int,
    )
    echelle.add_argument('--se', '--smoothech',
                         metavar='float', 
                         dest='smooth_ech',
                         help='Smooth ED using a box filter [in muHz]',
                         default=None,
                         type=float,
    )
    echelle.add_argument('--ue', '--uppere', 
                         metavar='float', 
                         dest='upper_ech',
                         help='Upper frequency limit of folded PS to whiten mixed modes',
                         nargs='*',
                         default=None,
                         type=float,
    )


    mcmc = main_parser.add_argument_group('Sampling')

    mcmc.add_argument('--mc', '--iter', '--mciter', 
                      metavar='int', 
                      dest='mc_iter', 
                      help='Number of Monte-Carlo iterations',
                      default=1, 
                      type=int,
    )
    mcmc.add_argument('-m', '--samples', 
                      dest='samples',
                      help='Save samples from the Monte-Carlo sampling',
                      default=False, 
                      action='store_true',
    )

# Different parsers

    sub_parser = parser.add_subparsers(title='pySYD modes', dest='mode')

    parser_display = sub_parser.add_parser('display',
                                           parents=[parent_parser, main_parser],
                                           formatter_class=argparse.MetavarTypeHelpFormatter,
                                           help='Display relevant information',
                                        )

    parser_display.add_argument('-c', '--cols', '--columns',
                                dest='columns',
                                help='Show columns of interest in a condensed format',
                                default=False,
                                action='store_true',
    )
    parser_display.add_argument('-d', '--show', '--display',
                                dest='show',
                                help='Show output figures',
                                default=True, 
                                action='store_false',
    )
    parser_display.add_argument('-v', '--verbose', 
                                dest='verbose',
                                help='Turn off verbose output',
                                default=True, 
                                action='store_false',
    )

    parser_display.set_defaults(func=pipeline.display)


    parser_load = sub_parser.add_parser('load',
                                        parents=[parent_parser, main_parser], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        help='Load in data for a given target',  
                                        )

    parser_load.add_argument('-d', '--show', '--display',
                             dest='show',
                             help='Do not show output figures',
                             default=False, 
                             action='store_true',
    )
    parser_load.add_argument('-v', '--verbose', 
                             dest='verbose',
                             help='Turn off verbose output',
                             default=True, 
                             action='store_false',
    )

    parser_load.set_defaults(func=pipeline.load)


    parser_parallel = sub_parser.add_parser('parallel', 
                                            help='Run pySYD in parallel',
                                            parents=[parent_parser, main_parser],
                                            formatter_class=argparse.MetavarTypeHelpFormatter,
                                            )

    parser_parallel.add_argument('-d', '--show', '--display',
                                 dest='show',
                                 help='Show output figures (not recommended for this mode)',
                                 default=False, 
                                 action='store_true',
    )
    parser_parallel.add_argument('--nt', '--nthread', '--nthreads',
                                 metavar='int', 
                                 dest='n_threads',
                                 help='Number of processes to run in parallel',
                                 type=int,
                                 default=0,
    )
    parser_parallel.add_argument('-v', '--verbose', 
                                 dest='verbose',
                                 help='Turn on verbose output (not recommended in this mode)',
                                 default=False, 
                                 action='store_true',
    )

    parser_parallel.set_defaults(func=pipeline.parallel)



    parser_run = sub_parser.add_parser('run',
                                       help='Run the main pySYD pipeline',
                                       parents=[parent_parser, main_parser], 
                                       formatter_class=argparse.MetavarTypeHelpFormatter,
                                       )

    parser_run.add_argument('-d', '--show', '--display',
                            dest='show',
                            help='Show output figures',
                            default=False, 
                            action='store_true',
    )
    parser_run.add_argument('-v', '--verbose', 
                            dest='verbose',
                            help='Turn off verbose output',
                            default=False, 
                            action='store_true',
    )

    parser_run.set_defaults(func=pipeline.run)


    parser_setup = sub_parser.add_parser('setup', 
                                         parents=[parent_parser], 
                                         formatter_class=argparse.MetavarTypeHelpFormatter,
                                         help='Easy setup of relevant directories and files',
                                         )
    parser_setup.add_argument('--dir', '--directory',
                              metavar='str',
                              dest='dir',
                              help='Path to save setup files to (default=os.getcwd())',
                              type=str,
                              default=os.path.abspath(os.getcwd()),
    )
    parser_setup.add_argument('-n', '--newpath', '--path',
                              dest='new',
                              help='Set up new path in pysyd init file',
                              default=False,
                              action='store_true',
    )
    parser_setup.add_argument('-v', '--verbose', 
                              dest='verbose',
                              help='Turn off verbose output',
                              default=True, 
                              action='store_false',
    )

    parser_setup.set_defaults(func=pipeline.setup)


    parser_test = sub_parser.add_parser('test',
                                        parents=[parent_parser, main_parser], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        help='Test different utilities (currently under development)',  
                                        )
    parser_test.add_argument('-d', '--show', '--display',
                             dest='show',
                             help='Show output figures',
                             default=True, 
                             action='store_false',
    )

    parser_test.add_argument('--methods', 
                             dest='methods',
                             help='Compare different dnu methods',
                             default=False,
                             action='store_true',
    )
    parser_test.add_argument('--models', 
                             dest='models',
                             help='Include different model fits',
                             default=False,
                             action='store_true',
    )
    parser_test.add_argument('-v', '--verbose', 
                             dest='verbose',
                             help='Turn off verbose output',
                             default=True, 
                             action='store_false',
    )

    parser_test.set_defaults(func=pipeline.test)

    args = parser.parse_args()
    args.func(args)



if __name__ == '__main__':

    main()
