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

    # In the parent parser, we define arguments and options common to all subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--file', '--list', '--todo',
                               metavar='path',
                               dest='todo',
                               help='List of stars to process',
                               type=str,
                               default=TODODIR,
    )
    parent_parser.add_argument('--in', '--input', '--inpdir', 
                               metavar='path',
                               dest='inpdir',
                               help='Input directory',
                               type=str,
                               default=INPDIR,
    )
    parent_parser.add_argument('--info', '--information',
                               metavar='path',
                               dest='info',
                               help='Path to star info',
                               type=str,
                               default=INFODIR,
    )
    parent_parser.add_argument('--out', '--outdir', '--output',
                               metavar='path',
                               dest='outdir',
                               help='Output directory',
                               type=str,
                               default=OUTDIR,
    )
    parent_parser.add_argument('--verbose', '-v', 
                               dest='verbose',
                               help='Turn on verbose output',
                               default=False, 
                               action='store_true',
    )

    # Main options 
    main_parser = argparse.ArgumentParser(add_help=False)

    main_parser.add_argument('--bg', '--fitbg', '--background', '-b', 
                             dest='background',
                             help='Turn off background fitting routine',
                             default=True, 
                             action='store_false',
    )
    main_parser.add_argument('--ex', '--findex', '--excess', '-x', 
                             dest='excess',
                             help='Turn off the find excess module',
                             default=True, 
                             action='store_false',
    )
    main_parser.add_argument('--kc', '--kepcorr', '-k', 
                             dest='kepcorr',
                             help='Turn on the Kepler short-cadence artefact correction routine',
                             default=False, 
                             action='store_true',
    )
    main_parser.add_argument('--ofa', '--of_actual',
                             metavar='n',
                             dest='of_actual',
                             help='The oversampling factor (OF) of the input PS',
                             type=int,
                             default=None,
    )
    main_parser.add_argument('--ofn', '--of_new',
                             metavar='n',
                             dest='of_new',
                             help='The OF to be used for the first iteration',
                             type=int,
                             default=None,
    )
    main_parser.add_argument('--save',
                             dest='save',
                             help='Do not save output files/figures',
                             default=True, 
                             action='store_false',
    )
    main_parser.add_argument('--show', '-s', 
                             dest='show',
                             help='Display figures',
                             default=False, 
                             action='store_true',
    )
    main_parser.add_argument('--star', '--stars',
                             metavar='star',
                             dest='stars',
                             help='List of stars to process',
                             type=str,
                             nargs='*',
                             default=None,
    )

    # CLI relevant for finding power excess
    excess = main_parser.add_argument_group('(CRUDE) EXCESS FIT')

    excess.add_argument('--bin', '--binning',
                        metavar='value',  
                        dest='binning', 
                        help='Binning interval for PS (in muHz)',
                        default=0.005, 
                        type=float,
    )
    excess.add_argument('--bm', '--mode', '--bmode',
                        metavar='mode',
                        choices=["mean", "median", "gaussian"],
                        dest='mode',
                        help='Binning mode',
                        default='mean',
                        type=str,
    )
    excess.add_argument('--sw', '--smoothwidth',
                        metavar='value', 
                        dest='smooth_width',
                        help='Box filter width (in muHz) for smoothing the PS',
                        default=50.0,
                        type=float,
    )
    excess.add_argument('--lx', '--lowerx', 
                        metavar='freq', 
                        dest='lower_ex',
                        help='Lower frequency limit of PS',
                        nargs='*',
                        default=None,
                        type=float,
    )
    excess.add_argument('--ux', '--upperx', 
                        metavar='freq', 
                        dest='upper_ex',
                        help='Upper frequency limit of PS',
                        nargs='*',
                        default=None,
                        type=float,
    )
    excess.add_argument('--step', '--steps', 
                        metavar='value', 
                        dest='step', 
                        default=0.25,
                        type=float, 
    )
    excess.add_argument('--trials', '--ntrials',
                        metavar='n', 
                        dest='n_trials',
                        default=3, 
                        type=int,
    )

    # CLI relevant for background fitting
    background = main_parser.add_argument_group('BACKGROUND-RELATED')

    background.add_argument('--lb', '--lowerb', 
                            metavar='freq', 
                            dest='lower_bg',
                            help='Lower frequency limit of PS',
                            nargs='*',
                            default=None,
                            type=float,
    )
    background.add_argument('--ub', '--upperb', 
                            metavar='freq', 
                            dest='upper_bg',
                            help='Upper frequency limit of PS',
                            nargs='*',
                            default=None,
                            type=float,
    )
    background.add_argument('--iw', '--indwidth',
                            metavar='value', 
                            dest='ind_width', 
                            help='Width of binning for PS [in muHz]',
                            default=20., 
                            type=float,
    )
    background.add_argument('--bf', '--box', '--boxfilter',
                            metavar='value', 
                            dest='box_filter',
                            help='Box filter width [in muHz] for plotting the PS',
                            default=1.0,
                            type=float,
    )
    background.add_argument('--rms', '--nrms', 
                            metavar='n', 
                            dest='n_rms', 
                            help='Number of points to estimate the amplitude of red-noise component(s)',
                            default=20, 
                            type=int,
    )
    background.add_argument('--laws', '--nlaws', 
                            metavar='n', 
                            dest='n_laws', 
                            help='Force number of red-noise component(s)',
                            default=None, 
                            type=int,
    )
    background.add_argument('--use', 
                            metavar='metric', 
                            dest='use', 
                            help="Which model metric to use, choices=['bic','aic']",
                            default='bic', 
                            type=str,
    )

    # CLI relevant for the global parameters
    numax = main_parser.add_argument_group('NUMAX-RELATED')
    numax.add_argument('--sm', '--smpar',
                       metavar='value', 
                       dest='sm_par',
                       help='Value of smoothing parameter to estimate smoothed numax (typically between 1-4).',
                       default=None, 
                       type=float,
    )
    numax.add_argument('--numax',
                       metavar='value',
                       dest='numax',
                       help='Skip find excess module and force numax',
                       nargs='*',
                       default=None,
                       type=float,
    )
    numax.add_argument('--lp', '--lowerp', 
                       metavar='freq', 
                       dest='lower_ps',
                       help='Lower frequency limit for zoomed in PS',
                       nargs='*',
                       default=None,
                       type=float,
    )
    numax.add_argument('--up', '--upperp', 
                       metavar='freq', 
                       dest='upper_ps',
                       help='Upper frequency limit for zoomed in PS',
                       nargs='*',
                       default=None,
                       type=float,
    )
    numax.add_argument('--ew', '--exwidth',
                       metavar='value', 
                       dest='width',
                       help='Fractional value of width to use for power excess, where width is computed using a solar scaling relation.',
                       default=1.0,
                       type=float,
    )

    dnu = main_parser.add_argument_group('DNU-RELATED')
    dnu.add_argument('--dnu',
                     metavar='value',
                     dest='dnu',
                     help='Brute force method to provide value for dnu',
                     nargs='*',
                     type=float,
                     default=None, 
    )
    dnu.add_argument('--sp', '--smoothps',
                     metavar='value', 
                     dest='smooth_ps',
                     help='Box filter width [in muHz] of PS for ACF', 
                     default=2.5,
                     type=float,
    )
    dnu.add_argument('--peak', '--peaks', '--npeaks',
                     metavar='n', 
                     dest='n_peaks', 
                     help='Number of peaks to fit in the ACF',
                     default=5, 
                     type=int,
    )
    dnu.add_argument('--thresh', '--threshold',
                     metavar='value', 
                     dest='threshold',
                     help='Fractional value of FWHM to use for ACF',
                     default=1.0,
                     type=float,
    )

    echelle = main_parser.add_argument_group('ECHELLE-RELATED')
    echelle.add_argument('--cv', '--value',
                         metavar='value', 
                         dest='clip_value',
                         help='Clip value to use for echelle diagram (ED)',
                         default=None, 
                         type=float,
    )
    echelle.add_argument('--clipech', 
                         dest='clip_ech',
                         help='Disable the auto-clipping of high peaks in the ED',
                         default=True, 
                         action='store_false',
    )
    echelle.add_argument('--le', '--lowere', 
                         metavar='freq', 
                         dest='lower_ech',
                         help='Lower frequency limit of folded PS to whiten mixed modes',
                         nargs='*',
                         default=None,
                         type=float,
    )
    echelle.add_argument('--ue', '--uppere', 
                         metavar='freq', 
                         dest='upper_ech',
                         help='Upper frequency limit of folded PS to whiten mixed modes',
                         nargs='*',
                         default=None,
                         type=float,
    )
    echelle.add_argument('--hey',
                         dest='hey', 
                         help="Use Daniel Hey's plugin for echelle",
                         default=False, 
                         action='store_true',
    )
    echelle.add_argument('--ie', '-interpech', '--interpech',
                         dest='interp_ech',
                         help='Turn on the interpolation of the output ED',
                         default=False,
                         action='store_true',
    )
    echelle.add_argument('--se', '--smoothech',
                         metavar='value', 
                         dest='smooth_ech',
                         help='Smooth ED using a box filter [in muHz]',
                         default=None,
                         type=float,
    )
    echelle.add_argument('--xe', '--xech', '--nacross',
                         metavar='n', 
                         dest='n_across',
                         help='Resolution for the x-axis of the ED',
                         default=50,
                         type=int, 
    )
    echelle.add_argument('--ye', '--yech', '--ndown',
                         metavar='n', 
                         dest='n_down',
                         help='The number of orders to plot on the ED y-axis',
                         default=5,
                         type=int,
    )

    mcmc = main_parser.add_argument_group('MCMC PARAMETERS')
    mcmc.add_argument('--mc', '--iter', '--mciter', 
                      metavar='n', 
                      dest='mc_iter', 
                      help='Number of Monte-Carlo iterations',
                      default=1, 
                      type=int,
    )
    mcmc.add_argument('--samples', 
                      dest='samples',
                      help='Save samples from the Monte-Carlo sampling',
                      default=False, 
                      action='store_true',
    )

    sub_parser = parser.add_subparsers(title='pySYD modes', dest='modes')

    # Setting up
    parser_setup = sub_parser.add_parser('setup', help='Easy setup of relevant directories and files',
                                         parents=[parent_parser], formatter_class=argparse.MetavarTypeHelpFormatter)
    parser_setup.set_defaults(func=pipeline.setup)

    # Load data in for a target
    parser_load = sub_parser.add_parser('load', help='Load in data for a given target',
                                       parents=[parent_parser, main_parser], formatter_class=argparse.MetavarTypeHelpFormatter)

    parser_load.set_defaults(func=pipeline.load)

    # Running pySYD in regular mode
    parser_run = sub_parser.add_parser('run', help='Run the main pySYD pipeline',
                                       parents=[parent_parser, main_parser], formatter_class=argparse.MetavarTypeHelpFormatter)

    parser_run.set_defaults(func=pipeline.run)

    # Run pySYD in parallel
    parser_parallel = sub_parser.add_parser('parallel', help='Run pySYD in parallel',
                                            parents=[parent_parser, main_parser], formatter_class=argparse.MetavarTypeHelpFormatter)
    parser_parallel.add_argument('-nt', '--nt', '-nthread', '--nthread', '-nthreads', '--nthreads',
                                 metavar='n', 
                                 dest='n_threads',
                                 help='Number of processes to run in parallel',
                                 type=int,
                                 default=0,
    )

    parser_parallel.set_defaults(func=pipeline.parallel)

    args = parser.parse_args()
    args.func(args)



if __name__ == '__main__':

    main()