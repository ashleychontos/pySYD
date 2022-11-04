import os
import sys
import argparse







# Package mode
import pysyd.pipeline
from pysyd import _ROOT, __version__



def main():

    parser = argparse.ArgumentParser(prog='pysyd',description="pySYD: automated measurements of global asteroseismic parameters")
    parser.add_argument('--version',
                        action='version',
                        version="%(prog)s {}".format(pysyd.__version__),
                        help="Print version number and exit.",
    )

####

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--in', '--input', '--inpdir', 
                               metavar='str',
                               dest='inpdir',
                               help='Input directory',
                               type=str,
                               default=os.path.join(_ROOT,'data'),
    )
    parent_parser.add_argument('--infdir',
                               metavar='str',
                               dest='infdir',
                               help='Path to relevant pySYD information',
                               type=str,
                               default=os.path.join(_ROOT,'info'),
    )
    parent_parser.add_argument('--out', '--outdir', '--output',
                               metavar='str',
                               dest='outdir',
                               help='Output directory',
                               type=str,
                               default=os.path.join(_ROOT,'results'),
    )
    parent_parser.add_argument('-s', '--save',
                               dest='save',
                               help='Do not save output figures and results',
                               default=True, 
                               action='store_false',
    )
    parent_parser.add_argument('-t', '--test',
                               dest='test',
                               help='Test core functionality of software on example stars',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('-o', '--overwrite',
                               dest='overwrite',
                               help='Overwrite existing files with the same name/path',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('-v', '--verbose', 
                               dest='verbose',
                               help='turn on verbose output',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('-w', '--warnings', 
                               dest='warnings',
                               help='turn on output warnings',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('--cli',
                               dest='cli',
                               help='Running from command line (this should not be touched)',
                               default=True,
                               action='store_false',
    )
    parent_parser.add_argument('--notebook',
                               dest='notebook',
                               help='Running from a jupyter notebook (this should not be touched)',
                               default=False,
                               action='store_true',
    )

####

    data_parser = argparse.ArgumentParser(add_help=False)
    data_parser.add_argument('--star', '--stars',
                             metavar='str',
                             dest='stars',
                             help='List of stars to process',
                             type=str,
                             nargs='*',
                             default=None,
    )
    data_parser.add_argument('--file', '--list', '--todo',
                             metavar='str',
                             dest='todo',
                             help='list of stars to process',
                             type=str,
                             default=os.path.join(_ROOT,'info','todo.txt'),
    )
    data_parser.add_argument('--info', '--information', 
                             metavar='str',
                             dest='info',
                             help='list of stellar parameters and options',
                             type=str,
                             default=os.path.join(_ROOT,'info','star_info.csv'),
    )
    data_parser.add_argument('--gap', '--gaps', 
                             metavar='int',
                             dest='gap',
                             help="What constitutes a time series 'gap' (i.e. n x the cadence)",
                             type=int,
                             default=20, 
    )
    data_parser.add_argument('-x', '--stitch', '--stitching',
                             dest='stitch',
                             help="Correct for large gaps in time series data by 'stitching' the light curve",
                             default=False,
                             action='store_true',
    )
    data_parser.add_argument('--of', '--over', '--oversample',
                             metavar='int',
                             dest='oversampling_factor',
                             help='The oversampling factor (OF) of the input power spectrum',
                             type=int,
                             default=None,
    )
    data_parser.add_argument('-k', '--kc', '--kepcorr', 
                             dest='kep_corr',
                             help='Turn on the Kepler short-cadence artefact correction routine',
                             default=False, 
                             action='store_true',
    )
    data_parser.add_argument('-f', '--force',
                             dest='force',
                             help='hack into ED output and use own value for dnu',
                             default=False,
                             action='store_true',
    )
    data_parser.add_argument('--dnu',
                             metavar='float',
                             dest='dnu',
                             help='spacing to fold PS for mitigating mixed modes',
                             nargs='*',
                             type=float,
                             default=None, 
    )
    data_parser.add_argument('--le', '--lowere', 
                             metavar='float', 
                             dest='lower_ech',
                             help='lower frequency limit of folded PS to whiten mixed modes',
                             nargs='*',
                             default=None,
                             type=float,
    )
    data_parser.add_argument('--ue', '--uppere', 
                             metavar='float', 
                             dest='upper_ech',
                             help='upper frequency limit of folded PS to whiten mixed modes',
                             nargs='*',
                             default=None,
                             type=float,
    )
    data_parser.add_argument('-n', '--notch', 
                             dest='notching',
                             help='another technique to mitigate effects from mixed modes (not fully functional, creates weirds effects for higher SNR cases??)',
                             default=False, 
                             action='store_true',
    )

####

    main_parser = argparse.ArgumentParser(add_help=False)

####

    estimate = main_parser.add_argument_group('Search parameters')
    estimate.add_argument('-e', '--est', '--estimate',
                          dest='estimate',
                          help='Turn off the optional module that estimates numax',
                          default=True,
                          action='store_false',
    )
    estimate.add_argument('-j', '--adjust',
                          dest='adjust',
                          help='Adjusts default parameters based on region of oscillations',
                          default=False, 
                          action='store_true',
    )
    estimate.add_argument('--def', '--defaults',
                          metavar='str',
                          dest='defaults',
                          help="Adjust defaults for low vs. high numax values (e.g., smoothing filters)",
                          type=str,
                          default=None,
    )
    estimate.add_argument('--sw', '--smoothwidth',
                          metavar='float', 
                          dest='smooth_width',
                          help='Box filter width (in muHz) for smoothing the PS',
                          default=20.0,
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

    background = main_parser.add_argument_group('Background parameters')
    background.add_argument('-b', '--bg', '--background',
                            dest='background',
                            help='Turn off the routine that determines the stellar background contribution',
                            default=True,
                            action='store_false',
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
    background.add_argument('--wn', '--fixwn',
                            dest='fix_wn',
                            help='Fix the white noise level',
                            default=False,
                            action='store_true',
    )
    background.add_argument('--metric', 
                            metavar='str', 
                            choices=['aic','bic'],
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
                       help='initial estimate for numax to bypass the first module',
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
    globe.add_argument('--fft',
                       dest='fft',
                       help='Use :mod:`numpy.correlate` instead of fast fourier transforms to compute the ACF',
                       default=True,
                       action='store_false',
    )
    globe.add_argument('--thresh', '--threshold',
                       metavar='float', 
                       dest='threshold',
                       help='fractional value of FWHM to use for ACF',
                       default=1.0,
                       type=float,
    )
    globe.add_argument('--peak', '--peaks', '--npeaks',
                       metavar='int', 
                       dest='n_peaks', 
                       help='number of peaks to fit in the ACF',
                       default=5, 
                       type=int,
    )

####

    mcmc = main_parser.add_argument_group('Sampling parameters')
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

    plot_parser = argparse.ArgumentParser(add_help=False)
    plot_parser.add_argument('--showall',
                             dest='show_all',
                             help='plot background comparison figure',
                             default=False,
                             action='store_true',
    )
    plot_parser.add_argument('-d', '--show', '--display',
                             dest='show',
                             help='Show output figures',
                             default=False, 
                             action='store_true',
    )
    plot_parser.add_argument('--cm', '--color', 
                             metavar='str',
                             dest='cmap',
                             help='Change colormap of ED, which is `binary` by default',
                             default='binary', 
                             type=str,
    )
    plot_parser.add_argument('--cv', '--value',
                             metavar='float', 
                             dest='clip_value',
                             help='Clip value multiplier to use for echelle diagram (ED). Default is 3x the median, where clip_value == `3`.',
                             default=3.0, 
                             type=float,
    )
    plot_parser.add_argument('-y', '--hey',
                             dest='hey', 
                             help="plugin for Daniel Hey's echelle package **not currently implemented**",
                             default=False, 
                             action='store_true',
    )
    plot_parser.add_argument('-i', '--ie', '--interpech',
                             dest='interp_ech',
                             help='turn on the interpolation of the output ED',
                             default=False,
                             action='store_true',
    )
    plot_parser.add_argument('--nox', '--nacross',
                             metavar='int', 
                             dest='nox',
                             help='number of bins to use on the x-axis of the ED (currently being tested)',
                             default=None,
                             type=int, 
    )
    plot_parser.add_argument('--noy', '--ndown', '--norders',
                             metavar='str', 
                             dest='noy',
                             help='NEW!! Number of orders to plot pm how many orders to shift (if ED is not centered)',
                             default='0+0',
                             type=str,
    )
    plot_parser.add_argument('--npb',
                             metavar='int',
                             dest='npb',
                             help='NEW!! npb == "number per bin", which is option instead of nox that uses the frequency resolution and spacing to compute an appropriate bin size for the ED',
                             default=10,
                             type=int,
    )
    plot_parser.add_argument('-r', '--ridges',
                             dest='ridges',
                             help='Run the optimize ridges module for the most excellent spacing (numerically solved so takes a minute or 3)',
                             default=False, 
                             action='store_true',
    )
    plot_parser.add_argument('--se', '--smoothech',
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
                                         parents=[parent_parser, data_parser, plot_parser],
                                         formatter_class=argparse.MetavarTypeHelpFormatter,
                                         help='Check target data',
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
    parser_check.set_defaults(func=pysyd.pipeline.check)

    parser_fun = sub_parser.add_parser('fun',
                                       help='Print logo and exit',
                                       )
    parser_fun.set_defaults(func=pysyd.pipeline.fun)

####

    parser_load = sub_parser.add_parser('load',
                                        parents=[parent_parser, data_parser, plot_parser], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        help='Load in data for a given target',  
                                        )
    parser_load.set_defaults(func=pysyd.pipeline.load)

####

    parser_parallel = sub_parser.add_parser('parallel', 
                                            help='Run pySYD in parallel',
                                            parents=[parent_parser, data_parser, main_parser, plot_parser],
                                            formatter_class=argparse.MetavarTypeHelpFormatter,
                                            )
    parser_parallel.add_argument('--nt', '--nthread', '--nthreads',
                                 metavar='int', 
                                 dest='n_threads',
                                 help='Number of processes to run in parallel',
                                 type=int,
                                 default=0,
    )
    parser_parallel.set_defaults(func=pysyd.pipeline.parallel)

####

    parser_plot = sub_parser.add_parser('plot',
                                        help='Create and show relevant figures',
                                        parents=[parent_parser, data_parser, plot_parser], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        )
    parser_plot.add_argument('-c', '--compare', 
                             dest='compare',
                             help='Reproduce the *Kepler* legacy results',
                             default=False,
                             action='store_true',
    )
    parser_plot.add_argument('--results', 
                             dest='results',
                             help='Re-plot ``pySYD`` results for a single star',
                             default=False,
                             action='store_true',
    )
    parser_plot.set_defaults(func=pysyd.pipeline.plot)

####

    parser_run = sub_parser.add_parser('run',
                                       help='Run the main pySYD pipeline',
                                       parents=[parent_parser, data_parser, main_parser, plot_parser], 
                                       formatter_class=argparse.MetavarTypeHelpFormatter,
                                       )
    parser_run.add_argument('--seed',
                            dest='seed',
                            help='save seed for reproducible results',
                            default=None,
                            type=int,
    )
    parser_run.set_defaults(func=pysyd.pipeline.run)

####

    parser_setup = sub_parser.add_parser('setup', 
                                         parents=[parent_parser, data_parser], 
                                         formatter_class=argparse.MetavarTypeHelpFormatter,
                                         help='Easy setup of relevant directories and files',
                                         )
    parser_setup.add_argument('--all',
                              dest='makeall',
                              help='Save all columns',
                              default=False, 
                              action='store_true',
    )
    parser_setup.add_argument('--path', '--dir', '--directory',
                              metavar='str',
                              dest='dir',
                              help='Path to save setup files to (default=os.getcwd()) **not functional yet',
                              type=str,
                              default=os.path.abspath(os.getcwd()),
    )
    parser_setup.set_defaults(func=pysyd.pipeline.setup)


    args = parser.parse_args()
    args.func(args)



if __name__ == '__main__':

    main()




def parse_args(args):

    parser = argparse.ArgumentParser(prog='pysyd',description="pySYD: automated measurements of global asteroseismic parameters")
    parser.add_argument('--version',
                        action='version',
                        version="%(prog)s {}".format(pysyd.__version__),
                        help="Print version number and exit.",
    )

####

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--in', '--input', '--inpdir', 
                               metavar='str',
                               dest='inpdir',
                               help='Input directory',
                               type=str,
                               default=os.path.join(_ROOT,'data'),
    )
    parent_parser.add_argument('--infdir',
                               metavar='str',
                               dest='infdir',
                               help='Path to relevant pySYD information',
                               type=str,
                               default=os.path.join(_ROOT,'info'),
    )
    parent_parser.add_argument('--out', '--outdir', '--output',
                               metavar='str',
                               dest='outdir',
                               help='Output directory',
                               type=str,
                               default=os.path.join(_ROOT,'results'),
    )
    parent_parser.add_argument('-s', '--save',
                               dest='save',
                               help='Do not save output figures and results',
                               default=True, 
                               action='store_false',
    )
    parent_parser.add_argument('-t', '--test',
                               dest='test',
                               help='Test core functionality of software on example stars',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('-o', '--overwrite',
                               dest='overwrite',
                               help='Overwrite existing files with the same name/path',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('-v', '--verbose', 
                               dest='verbose',
                               help='turn on verbose output',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('-w', '--warnings', 
                               dest='warnings',
                               help='turn on output warnings',
                               default=False, 
                               action='store_true',
    )
    parent_parser.add_argument('--cli',
                               dest='cli',
                               help='Running from command line (this should not be touched)',
                               default=True,
                               action='store_false',
    )
    parent_parser.add_argument('--notebook',
                               dest='notebook',
                               help='Running from a jupyter notebook (this should not be touched)',
                               default=False,
                               action='store_true',
    )

####

    data_parser = argparse.ArgumentParser(add_help=False)
    data_parser.add_argument('--star', '--stars',
                             metavar='str',
                             dest='stars',
                             help='List of stars to process',
                             type=str,
                             nargs='*',
                             default=None,
    )
    data_parser.add_argument('--file', '--list', '--todo',
                             metavar='str',
                             dest='todo',
                             help='list of stars to process',
                             type=str,
                             default=os.path.join(_ROOT,'info','todo.txt'),
    )
    data_parser.add_argument('--info', '--information', 
                             metavar='str',
                             dest='info',
                             help='list of stellar parameters and options',
                             type=str,
                             default=os.path.join(_ROOT,'info','star_info.csv'),
    )
    data_parser.add_argument('--gap', '--gaps', 
                             metavar='int',
                             dest='gap',
                             help="What constitutes a time series 'gap' (i.e. n x the cadence)",
                             type=int,
                             default=20, 
    )
    data_parser.add_argument('-x', '--stitch', '--stitching',
                             dest='stitch',
                             help="Correct for large gaps in time series data by 'stitching' the light curve",
                             default=False,
                             action='store_true',
    )
    data_parser.add_argument('--of', '--over', '--oversample',
                             metavar='int',
                             dest='oversampling_factor',
                             help='The oversampling factor (OF) of the input power spectrum',
                             type=int,
                             default=None,
    )
    data_parser.add_argument('-k', '--kc', '--kepcorr', 
                             dest='kep_corr',
                             help='Turn on the Kepler short-cadence artefact correction routine',
                             default=False, 
                             action='store_true',
    )
    data_parser.add_argument('-f', '--force',
                             dest='force',
                             help='hack into ED output and use own value for dnu',
                             default=False,
                             action='store_true',
    )
    data_parser.add_argument('--dnu',
                             metavar='float',
                             dest='dnu',
                             help='spacing to fold PS for mitigating mixed modes',
                             nargs='*',
                             type=float,
                             default=None, 
    )
    data_parser.add_argument('--le', '--lowere', 
                             metavar='float', 
                             dest='lower_ech',
                             help='lower frequency limit of folded PS to whiten mixed modes',
                             nargs='*',
                             default=None,
                             type=float,
    )
    data_parser.add_argument('--ue', '--uppere', 
                             metavar='float', 
                             dest='upper_ech',
                             help='upper frequency limit of folded PS to whiten mixed modes',
                             nargs='*',
                             default=None,
                             type=float,
    )
    data_parser.add_argument('-n', '--notch', 
                             dest='notching',
                             help='another technique to mitigate effects from mixed modes (not fully functional, creates weirds effects for higher SNR cases??)',
                             default=False, 
                             action='store_true',
    )

####

    main_parser = argparse.ArgumentParser(add_help=False)

####

    estimate = main_parser.add_argument_group('Search parameters')
    estimate.add_argument('-e', '--est', '--estimate',
                          dest='estimate',
                          help='Turn off the optional module that estimates numax',
                          default=True,
                          action='store_false',
    )
    estimate.add_argument('-j', '--adjust',
                          dest='adjust',
                          help='Adjusts default parameters based on region of oscillations',
                          default=False, 
                          action='store_true',
    )
    estimate.add_argument('--def', '--defaults',
                          metavar='str',
                          dest='defaults',
                          help="Adjust defaults for low vs. high numax values (e.g., smoothing filters)",
                          type=str,
                          default=None,
    )
    estimate.add_argument('--sw', '--smoothwidth',
                          metavar='float', 
                          dest='smooth_width',
                          help='Box filter width (in muHz) for smoothing the PS',
                          default=20.0,
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

    background = main_parser.add_argument_group('Background parameters')
    background.add_argument('-b', '--bg', '--background',
                            dest='background',
                            help='Turn off the routine that determines the stellar background contribution',
                            default=True,
                            action='store_false',
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
    background.add_argument('--wn', '--fixwn',
                            dest='fix_wn',
                            help='Fix the white noise level',
                            default=False,
                            action='store_true',
    )
    background.add_argument('--metric', 
                            metavar='str', 
                            choices=['aic','bic'],
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
                       help='initial estimate for numax to bypass the first module',
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
    globe.add_argument('--fft',
                       dest='fft',
                       help='Use :mod:`numpy.correlate` instead of fast fourier transforms to compute the ACF',
                       default=True,
                       action='store_false',
    )
    globe.add_argument('--thresh', '--threshold',
                       metavar='float', 
                       dest='threshold',
                       help='fractional value of FWHM to use for ACF',
                       default=1.0,
                       type=float,
    )
    globe.add_argument('--peak', '--peaks', '--npeaks',
                       metavar='int', 
                       dest='n_peaks', 
                       help='number of peaks to fit in the ACF',
                       default=5, 
                       type=int,
    )

####

    mcmc = main_parser.add_argument_group('Sampling parameters')
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

    plot_parser = argparse.ArgumentParser(add_help=False)
    plot_parser.add_argument('--showall',
                             dest='show_all',
                             help='plot background comparison figure',
                             default=False,
                             action='store_true',
    )
    plot_parser.add_argument('-d', '--show', '--display',
                             dest='show',
                             help='Show output figures',
                             default=False, 
                             action='store_true',
    )
    plot_parser.add_argument('--cm', '--color', 
                             metavar='str',
                             dest='cmap',
                             help='Change colormap of ED, which is `binary` by default',
                             default='binary', 
                             type=str,
    )
    plot_parser.add_argument('--cv', '--value',
                             metavar='float', 
                             dest='clip_value',
                             help='Clip value multiplier to use for echelle diagram (ED). Default is 3x the median, where clip_value == `3`.',
                             default=3.0, 
                             type=float,
    )
    plot_parser.add_argument('-y', '--hey',
                             dest='hey', 
                             help="plugin for Daniel Hey's echelle package **not currently implemented**",
                             default=False, 
                             action='store_true',
    )
    plot_parser.add_argument('-i', '--ie', '--interpech',
                             dest='interp_ech',
                             help='turn on the interpolation of the output ED',
                             default=False,
                             action='store_true',
    )
    plot_parser.add_argument('--nox', '--nacross',
                             metavar='int', 
                             dest='nox',
                             help='number of bins to use on the x-axis of the ED (currently being tested)',
                             default=None,
                             type=int, 
    )
    plot_parser.add_argument('--noy', '--ndown', '--norders',
                             metavar='str', 
                             dest='noy',
                             help='NEW!! Number of orders to plot pm how many orders to shift (if ED is not centered)',
                             default='0+0',
                             type=str,
    )
    plot_parser.add_argument('--npb',
                             metavar='int',
                             dest='npb',
                             help='NEW!! npb == "number per bin", which is option instead of nox that uses the frequency resolution and spacing to compute an appropriate bin size for the ED',
                             default=10,
                             type=int,
    )
    plot_parser.add_argument('-r', '--ridges',
                             dest='ridges',
                             help='Run the optimize ridges module for the most excellent spacing (numerically solved so takes a minute or 3)',
                             default=False, 
                             action='store_true',
    )
    plot_parser.add_argument('--se', '--smoothech',
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
                                         parents=[parent_parser, data_parser, plot_parser],
                                         formatter_class=argparse.MetavarTypeHelpFormatter,
                                         help='Check target data',
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

    parser_fun = sub_parser.add_parser('fun',
                                       help='Print logo and exit',
                                       )

####

    parser_load = sub_parser.add_parser('load',
                                        parents=[parent_parser, data_parser, plot_parser], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        help='Load in data for a given target',  
                                        )
####

    parser_parallel = sub_parser.add_parser('parallel', 
                                            help='Run pySYD in parallel',
                                            parents=[parent_parser, data_parser, main_parser, plot_parser],
                                            formatter_class=argparse.MetavarTypeHelpFormatter,
                                            )
    parser_parallel.add_argument('--nt', '--nthread', '--nthreads',
                                 metavar='int', 
                                 dest='n_threads',
                                 help='Number of processes to run in parallel',
                                 type=int,
                                 default=0,
    )

####

    parser_plot = sub_parser.add_parser('plot',
                                        help='Create and show relevant figures',
                                        parents=[parent_parser, data_parser, plot_parser], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        )
    parser_plot.add_argument('-c', '--compare', 
                             dest='compare',
                             help='Reproduce the *Kepler* legacy results',
                             default=False,
                             action='store_true',
    )
    parser_plot.add_argument('--results', 
                             dest='results',
                             help='Re-plot ``pySYD`` results for a single star',
                             default=False,
                             action='store_true',
    )

####

    parser_run = sub_parser.add_parser('run',
                                       help='Run the main pySYD pipeline',
                                       parents=[parent_parser, data_parser, main_parser, plot_parser], 
                                       formatter_class=argparse.MetavarTypeHelpFormatter,
                                       )
    parser_run.add_argument('--seed',
                            dest='seed',
                            help='save seed for reproducible results',
                            default=None,
                            type=int,
    )

####

    parser_setup = sub_parser.add_parser('setup', 
                                         parents=[parent_parser, data_parser], 
                                         formatter_class=argparse.MetavarTypeHelpFormatter,
                                         help='Easy setup of relevant directories and files',
                                         )
    parser_setup.add_argument('--all',
                              dest='makeall',
                              help='Save all columns',
                              default=False, 
                              action='store_true',
    )
    parser_setup.add_argument('--path', '--dir', '--directory',
                              metavar='str',
                              dest='dir',
                              help='Path to save setup files to (default=os.getcwd()) **not functional yet',
                              type=str,
                              default=os.path.abspath(os.getcwd()),
    )

    return parser.parse_args(args)

