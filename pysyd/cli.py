import os
import argparse

import pysyd
from pysyd import pipeline
from pysyd import INFDIR, INPDIR, OUTDIR


def main():

####

    parser = argparse.ArgumentParser(
                                     description="pySYD: automated measurements of global asteroseismic parameters", 
                                     prog='pySYD',
    )
    parser.add_argument('--version',
                        action='version',
                        version="%(prog)s {}".format(pysyd.__version__),
                        help="Print version number and exit.",
    )

####

    high_level_function = argparse.ArgumentParser(add_help=False)
    high_level_function.add_argument('--in', '--input', '--inpdir', 
                                     metavar='str',
                                     dest='inpdir',
                                     help='Input directory',
                                     type=str,
                                     default=INPDIR,
    )
    high_level_function.add_argument('--infdir',
                                     metavar='str',
                                     dest='infdir',
                                     help='Path to relevant pySYD information',
                                     type=str,
                                     default=INFDIR,
    )
    high_level_function.add_argument('--out', '--outdir', '--output',
                                     metavar='str',
                                     dest='outdir',
                                     help='Output directory',
                                     type=str,
                                     default=OUTDIR,
    )
    high_level_function.add_argument('-s', '--save',
                                     dest='save',
                                     help='Do not save output figures and results.',
                                     default=True, 
                                     action='store_false',
    )
    high_level_function.add_argument('-o', '--overwrite',
                                     dest='overwrite',
                                     help='Overwrite existing files with the same name/path',
                                     default=False, 
                                     action='store_true',
    )
    high_level_function.add_argument('-v', '--verbose', 
                                     dest='verbose',
                                     help='turn on verbose output',
                                     default=False, 
                                     action='store_true',
    )
    high_level_function.add_argument('--cli',
                                     dest='cli',
                                     help='Running from command line (this should not be touched)',
                                     default=True,
                                     action='store_true',
    )

####

    data_analyses = argparse.ArgumentParser(add_help=False)
    data_analyses.add_argument('--star', '--stars',
                               metavar='str',
                               dest='stars',
                               help='List of stars to process',
                               type=str,
                               nargs='*',
                               default=None,
    )
    data_analyses.add_argument('--file', '--list', '--todo',
                               metavar='str',
                               dest='todo',
                               help='list of stars to process',
                               type=str,
                               default='todo.txt',
    )
    data_analyses.add_argument('--info', '--information', 
                               metavar='str',
                               dest='info',
                               help='list of stellar parameters and options',
                               type=str,
                               default='star_info.csv',
    )
    data_analyses.add_argument('--gap', '--gaps', 
                               metavar='int',
                               dest='gap',
                               help="What constitutes a time series 'gap' (i.e. n x the cadence)",
                               type=int,
                               default=20, 
    )
    data_analyses.add_argument('-x', '--stitch', '--stitching',
                               dest='stitch',
                               help="Correct for large gaps in time series data by 'stitching' the light curve",
                               default=False,
                               action='store_true',
    )
    data_analyses.add_argument('--of', '--over', '--oversample',
                               metavar='int',
                               dest='oversampling_factor',
                               help='The oversampling factor (OF) of the input power spectrum',
                               type=int,
                               default=None,
    )
    data_analyses.add_argument('-k', '--kc', '--kepcorr', 
                               dest='kep_corr',
                               help='Turn on the Kepler short-cadence artefact correction routine',
                               default=False, 
                               action='store_true',
    )
    data_analyses.add_argument('--dnu',
                               metavar='float',
                               dest='dnu',
                               help='spacing to fold PS for mitigating mixed modes',
                               nargs='*',
                               type=float,
                               default=None, 
    )
    data_analyses.add_argument('--le', '--lowere', 
                               metavar='float', 
                               dest='lower_ech',
                               help='lower frequency limit of folded PS to whiten mixed modes',
                               nargs='*',
                               default=None,
                               type=float,
    )
    data_analyses.add_argument('--ue', '--uppere', 
                               metavar='float', 
                               dest='upper_ech',
                               help='upper frequency limit of folded PS to whiten mixed modes',
                               nargs='*',
                               default=None,
                               type=float,
    )
    data_analyses.add_argument('-n', '--notch', 
                               dest='notching',
                               help='another technique to mitigate effects from mixed modes (not fully functional, creates weirds effects for higher SNR cases??)',
                               default=False, 
                               action='store_true',
    )

####

    main_parser = argparse.ArgumentParser(add_help=False)

####

    estimate = main_parser.add_argument_group('Estimate parameters')
    estimate.add_argument('-e', '--est', '--excess',
                         dest='excess',
                         help='Turn off the optional module that estimates numax',
                         default=True,
                         action='store_false',
    ) 
    estimate.add_argument('--sw', '--smoothwidth',
                         metavar='float', 
                         dest='smooth_width',
                         help='Box filter width (in muHz) for smoothing the PS',
                         default=10.0,
                         type=float,
    )
    estimate.add_argument('--bin', '--binning',
                         metavar='float',  
                         dest='binning', 
                         help='Binning interval for PS (in muHz)',
                         default=0.005, 
                         type=float,
    )
    estimate.add_argument('--bm', '--mode', '--bmode',
                         metavar='str',
                         choices=["mean", "median", "gaussian"],
                         dest='bin_mode',
                         help='Binning mode',
                         default='mean',
                         type=str,
    )
    estimate.add_argument('--step', '--steps', 
                         metavar='float', 
                         dest='step', 
                         default=0.25,
                         type=float, 
    )
    estimate.add_argument('--trials', '--ntrials',
                         metavar='int', 
                         dest='n_trials',
                         default=3, 
                         type=int,
    )
    estimate.add_argument('-a', '--ask',
                         dest='ask',
                         help='Ask which trial to use',
                         default=False, 
                         action='store_true',
    )
    estimate.add_argument('--lx', '--lowerx', 
                         metavar='float', 
                         dest='lower_ex',
                         help='Lower frequency limit of PS',
                         nargs='*',
                         default=None,
                         type=float,
    )
    estimate.add_argument('--ux', '--upperx', 
                         metavar='float', 
                         dest='upper_ex',
                         help='Upper frequency limit of PS',
                         nargs='*',
                         default=None,
                         type=float,
    )

####

    background = main_parser.add_argument_group('Background fits')
    background.add_argument('--basis', 
                            metavar='str',
                            dest='basis',
                            help="Which basis to use for background fit (i.e. 'a_b', 'pgran_tau', 'tau_sigma'), *** NOT operational yet ***",
                            default='tau_sigma', 
                            type=str,
    )
    background.add_argument('-b', '--bg', '--background',
                            dest='background',
                            help='Turn off the routine that determines the stellar background contribution',
                            default=True,
                            action='store_false',
    )
    background.add_argument('--bf', '--box', '--boxfilter',
                            metavar='float', 
                            dest='box_filter',
                            help='Box filter width [in muHz] for plotting the PS',
                            default=1.0,
                            type=float,
    )
    background.add_argument('--iw', '--indwidth',
                            metavar='float', 
                            dest='ind_width', 
                            help='Width of binning for PS [in muHz]',
                            default=20.0, 
                            type=float,
    )
    background.add_argument('--rms', '--nrms', 
                            metavar='int', 
                            dest='n_rms', 
                            help='Number of points to estimate the amplitude of red-noise component(s)',
                            default=20, 
                            type=int,
    )
    background.add_argument('--laws', '--nlaws', 
                            metavar='int', 
                            dest='n_laws', 
                            help='Force number of red-noise component(s)',
                            default=None, 
                            type=int,
    )
    background.add_argument('-f', '--fix', '--fixwn',
                            dest='fix_wn',
                            help='Fix the white noise level',
                            default=False,
                            action='store_true',
    )
    background.add_argument('--metric', 
                            metavar='str', 
                            dest='metric', 
                            help="Which model metric to use, choices=['bic','aic']",
                            default='bic', 
                            type=str,
    )
    background.add_argument('--lb', '--lowerb', 
                            metavar='float', 
                            dest='lower_bg',
                            help='Lower frequency limit of PS',
                            nargs='*',
                            default=None,
                            type=float,
    )
    background.add_argument('--ub', '--upperb', 
                            metavar='float', 
                            dest='upper_bg',
                            help='Upper frequency limit of PS',
                            nargs='*',
                            default=None,
                            type=float,
    )

####

    globe = main_parser.add_argument_group('Global parameters')
    globe.add_argument('-g', '--globe', '--global',
                       dest='globe',
                       help='Disable the main global-fitting routine',
                       default=True,
                       action='store_false',
    )
    globe.add_argument('--numax',
                       metavar='float',
                       dest='numax',
                       help='initial estimate for numax to bypass the forst module',
                       nargs='*',
                       default=None,
                       type=float,
    )
    globe.add_argument('--lp', '--lowerp', 
                       metavar='float', 
                       dest='lower_ps',
                       help='lower frequency limit for the envelope of oscillations',
                       nargs='*',
                       default=None,
                       type=float,
    )
    globe.add_argument('--up', '--upperp', 
                       metavar='float', 
                       dest='upper_ps',
                       help='upper frequency limit for the envelope of oscillations',
                       nargs='*',
                       default=None,
                       type=float,
    )
    globe.add_argument('--ew', '--exwidth',
                       metavar='float', 
                       dest='ex_width',
                       help='fractional value of width to use for power excess, where width is computed using a solar scaling relation.',
                       default=1.0,
                       type=float,
    )
    globe.add_argument('--sm', '--smpar',
                       metavar='float', 
                       dest='sm_par',
                       help='smoothing parameter used to estimate the smoothed numax (typically before 1-4 through experience -- **development purposes only**)',
                       default=None, 
                       type=float,
    )
    globe.add_argument('--sp', '--smoothps',
                       metavar='float', 
                       dest='smooth_ps',
                       help='box filter width [in muHz] of PS for ACF', 
                       type=float,
                       default=2.5,
    )
    globe.add_argument('--thresh', '--threshold',
                       metavar='float', 
                       dest='threshold',
                       help='fractional value of FWHM to use for ACF',
                       default=1.0,
                       type=float,
    )
    globe.add_argument('--method',
                       metavar='str',
                       dest='method',
                       help='method to use to determine dnu, ~[M, A, D] **development purposes only**',
                       default='D',
                       type=str,
    )
    globe.add_argument('--peak', '--peaks', '--npeaks',
                       metavar='int', 
                       dest='n_peaks', 
                       help='number of peaks to fit in the ACF',
                       default=10, 
                       type=int,
    )

####

    mcmc = main_parser.add_argument_group('Estimate uncertainties')
    mcmc.add_argument('--mc', '--iter', '--mciter', 
                      metavar='int', 
                      dest='mc_iter', 
                      help='number of Monte-Carlo iterations to run for estimating uncertainties (typically 200 is sufficient)',
                      default=1, 
                      type=int,
    )
    mcmc.add_argument('-m', '--samples', 
                      dest='samples',
                      help='save samples from the Monte-Carlo sampling',
                      default=False, 
                      action='store_true',
    )

####

    plotting = argparse.ArgumentParser(add_help=False)
    plotting.add_argument('--all', '--showall',
                          dest='showall',
                          help='plot background comparison figure',
                          default=False,
                          action='store_true',
    )
    plotting.add_argument('-d', '--show', '--display',
                          dest='show',
                          help='Do not show output figures',
                          default=False, 
                          action='store_true',
    )
    plotting.add_argument('--ce', '--cm', '--color', 
                         metavar='str',
                         dest='cmap',
                         help='Change colormap of ED, which is `binary` by default.',
                         default='binary', 
                         type=str,
    )
    plotting.add_argument('--cv', '--value',
                         metavar='float', 
                         dest='clip_value',
                         help='Clip value multiplier to use for echelle diagram (ED). Default is 3x the median, where clip_value == `3`.',
                         default=3.0, 
                         type=float,
    )
    plotting.add_argument('-y', '--hey',
                         dest='hey', 
                         help="plugin for Daniel Hey's echelle package **not currently implemented**",
                         default=False, 
                         action='store_true',
    )
    plotting.add_argument('-i', '--ie', '--interpech',
                         dest='interp_ech',
                         help='turn on the interpolation of the output ED',
                         default=False,
                         action='store_true',
    )
    plotting.add_argument('--nox', '--nacross',
                         metavar='int', 
                         dest='nox',
                         help='number of bins to use on the x-axis of the ED',
                         default=50,
                         type=int, 
    )
    plotting.add_argument('--noy', '--ndown', '--norders',
                         metavar='int', 
                         dest='noy',
                         help='The number of orders to plot on the y-axis of the ED',
                         default=0,
                         type=int,
    )
    plotting.add_argument('--se', '--smoothech',
                         metavar='float', 
                         dest='smooth_ech',
                         help='Smooth ED using a box filter [in muHz]',
                         default=None,
                         type=float,
    )

#####################
# Different parsers #
#####################

    sub_parser = parser.add_subparsers(title='pySYD modes', dest='mode')

    parser_check = sub_parser.add_parser('check',
                                         parents=[high_level_function, data_analyses, plotting],
                                         formatter_class=argparse.MetavarTypeHelpFormatter,
                                         help='Check data for a target or other relevant information',
                                        )

    parser_check.add_argument('-c', '--cols', '--columns',
                              dest='columns',
                              help='Show columns of interest in a condensed format',
                              default=False,
                              action='store_true',
    )
    parser_check.add_argument('--data', 
                              dest='data',
                              help='Check data for a target',
                              default=True, 
                              action='store_false',
    )
    parser_check.add_argument('-l', '--log', 
                              dest='log',
                              help='Disable plotting of power spectrum in log-log scale',
                              default=True, 
                              action='store_false',
    )
    parser_check.add_argument('--lp', '--lowerp', '--lowerps',
                              metavar='float', 
                              dest='lower_ps',
                              help='Lower frequency limit to plot for power spectrum',
                              default=None,
                              type=float,
    )
    parser_check.add_argument('--ll', '--lc', '--lowert', '--lowerl', '--lowerlc',
                              metavar='float', 
                              dest='lower_lc',
                              help='Lower limit to plot for time series data',
                              default=None,
                              type=float,
    )
    parser_check.add_argument('-p', '--plot', 
                              dest='plot',
                              help='Disable automatic plotting of data',
                              default=True, 
                              action='store_false',
    )
    parser_check.add_argument('-r', '--ret', '--return',
                              dest='return',
                              help='Disable the returning of any output',
                              default=True, 
                              action='store_false',
    )
    parser_check.add_argument('--up', '--upperp', '--upperps',
                              metavar='float', 
                              dest='upper_ps',
                              help='Upper frequency limit to plot for power spectrum',
                              default=None,
                              type=float,
    )
    parser_check.add_argument('--ul', '--uc', '--uppert', '--upperl', '--upperlc',
                              metavar='float', 
                              dest='upper_lc',
                              help='Upper limit to plot for time series data',
                              default=None,
                              type=float,
    )
    parser_check.set_defaults(func=pipeline.check)

####

    parser_load = sub_parser.add_parser('load',
                                        parents=[high_level_function, data_analyses, plotting], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        help='Load in data for a given target',  
                                        )
    parser_load.add_argument('-r', '--ret', '--return',
                             dest='return',
                             help='Disable the returning of any output',
                             default=True, 
                             action='store_false',
    )
    parser_load.set_defaults(func=pipeline.load)

####

    parser_parallel = sub_parser.add_parser('parallel', 
                                            help='Run pySYD in parallel',
                                            parents=[high_level_function, data_analyses, main_parser, plotting],
                                            formatter_class=argparse.MetavarTypeHelpFormatter,
                                            )
    parser_parallel.add_argument('--nt', '--nthread', '--nthreads',
                                 metavar='int', 
                                 dest='n_threads',
                                 help='Number of processes to run in parallel',
                                 type=int,
                                 default=0,
    )
    parser_parallel.set_defaults(func=pipeline.parallel)

####

    parser_plot = sub_parser.add_parser('plot',
                                        help='Create and show relevant figures',
                                        parents=[high_level_function, data_analyses, plotting], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        )
    parser_plot.add_argument('-c', '--compare', 
                             dest='compare',
                             help='Reproduce the *Kepler* legacy results',
                             default=False,
                             action='store_true',
    )
    parser_plot.add_argument('-r', '--ret', '--return',
                             dest='return',
                             help='Disable the returning of any output',
                             default=True, 
                             action='store_false',
    )
    parser_plot.add_argument('--results', 
                             dest='results',
                             help='Re-plot ``pySYD`` results for a single star',
                             default=False,
                             action='store_true',
    )
    parser_plot.set_defaults(func=pipeline.plot)

####

    parser_run = sub_parser.add_parser('run',
                                       help='Run the main pySYD pipeline',
                                       parents=[high_level_function, data_analyses, main_parser, plotting], 
                                       formatter_class=argparse.MetavarTypeHelpFormatter,
                                       )
    parser_run.set_defaults(func=pipeline.run)

####

    parser_setup = sub_parser.add_parser('setup', 
                                         parents=[high_level_function], 
                                         formatter_class=argparse.MetavarTypeHelpFormatter,
                                         help='Easy setup of relevant directories and files',
                                         )
    parser_setup.add_argument('-a', '--all', 
                              dest='makeall',
                              help='Save all columns',
                              default=False, 
                              action='store_true',
    )
    parser_setup.add_argument('--dir', '--directory',
                              metavar='str',
                              dest='dir',
                              help='Path to save setup files to (default=os.getcwd())',
                              type=str,
                              default=os.path.abspath(os.getcwd()),
    )
    parser_setup.add_argument('-e', '--ex', '--examples',
                              dest='examples',
                              help='Disable auto-saving of example data',
                              default=True, 
                              action='store_false',
    )
    parser_setup.add_argument('-f', '--files',
                              dest='files',
                              help='Disable auto-saving of example input files',
                              default=True, 
                              action='store_false',
    )
    parser_setup.add_argument('-n', '--newpath', '--path',
                              dest='new',
                              help='Set up new path in pysyd init file',
                              default=False,
                              action='store_true',
    )
    parser_setup.set_defaults(func=pipeline.setup)


    parser_test = sub_parser.add_parser('test',
                                        parents=[high_level_function, data_analyses, main_parser, plotting], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        help='Test different utilities (currently under development)',  
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
    parser_test.add_argument('-r', '--ret', '--return',
                             dest='return',
                             help='Disable the returning of any parameters',
                             default=True, 
                             action='store_false',
    )
    parser_test.set_defaults(func=pipeline.test)



    args = parser.parse_args()
    args.func(args)



if __name__ == '__main__':

    main()
