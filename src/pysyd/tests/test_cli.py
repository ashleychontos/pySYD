import os
import sys
import requests
import warnings
import subprocess
import pytest

import pysyd
from pysyd import cli
from ..cli import parse_args
from ..utils import InputError, Parameters

warnings.simplefilter('ignore')



# TEST BASIC ARGUMENT PARSING
def test_cli(args=["pysyd","--version"]):
    sys.argv = args
    try:
        cli.main()
    except SystemExit:
        pass


# TEST COMMAND LINE INSTALL
def test_help(cmd='pysyd --help'):
    proc = subprocess.Popen(cmd.split())
    proc.wait()
    status = proc.poll()
    out, _ = proc.communicate()
    assert status == 0, "{} failed with exit code {}".format(cmd, status)


# MAKE SURE PACKAGE DATA IS INSTALLED
def test_info():
    from .. import PACKAGEDIR
    assert os.path.exists(PACKAGEDIR), "Package directory not found"


# TEST INPUT ERRORS
# PART I: INPDIR (i.e. data)
def test_inpdir(args=["pysyd","run","--star","1435467","--inpdir","DNE"]):
    sys.argv = args
    try:
        cli.main()
    except InputError as error:
        pass

# PART II: INFDIR (via 'todo.txt')
def test_todo(args=["pysyd","run","--todo","bananas"]):
    sys.argv = args
    try:
        cli.main()
    except InputError as error:
        pass


# MAKE SURE PROGRAM DEFAULTS MATCH FOR BOTH API+CLI VERSIONS
def test_parser_defaults():
    args = parse_args(["run"])
    params = Parameters()
    params.params['stars'] = None
    for key in params.params:
        if key != 'functions' and key != 'n_threads':
            assert hasattr(args, key), "Parser is missing %s option" % key
            assert args.__dict__[key] == params.params[key], "Default value is inconsistent for %s option" % key


# MAKE SURE STARS ARE SAVED AS A LIST
def test_multiple_inputs():
    args = parse_args(["run", "--star", "1435467", "2309595", "11618103"])
    assert isinstance(args.stars, list)


# TEST INVALID CLI INPUT
def test_incorrect_input():
    args = parse_args(["run", "--star", "1435467", "2309595", "--numax", "1299"])
    try:
        params = Parameters(args=args)
    except InputError as error:
        pass

"""

def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version="%(prog)s {}".format(pysyd.__version__))

    # PARENT PARSER
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--in','--input','--inpdir', dest='inpdir', type=str, default=INPDIR)
    parent_parser.add_argument('--infdir', dest='infdir', type=str, default=INFDIR)
    parent_parser.add_argument('--out', '--outdir', '--output', dest='outdir', type=str, default=OUTDIR)
    parent_parser.add_argument('-s', '--save', dest='save', default=True, action='store_false')
    parent_parser.add_argument('-t', '--test', dest='test', default=False, action='store_true')
    parent_parser.add_argument('-o', '--overwrite', dest='overwrite', default=False, action='store_true')
    parent_parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true')
    parent_parser.add_argument('-w', '--warnings', dest='warnings', default=False, action='store_true')
    parent_parser.add_argument('--cli', dest='cli', default=True, action='store_false')
    parent_parser.add_argument('--notebook', dest='notebook', default=False, action='store_true')

    # DATA PARSER
    data_parser = argparse.ArgumentParser()
    data_parser.add_argument('--star', '--stars', dest='stars', type=str, nargs='*', default=None)
    data_parser.add_argument('--file', '--list', '--todo', dest='todo', type=str, default=os.path.join(INFDIR,'todo.txt'))
    data_parser.add_argument('--info', '--information', dest='info', type=str, default=os.path.join(INFDIR,'star_info.csv'))
    data_parser.add_argument('--gap', '--gaps', dest='gap', type=int, default=20)
    data_parser.add_argument('-x', '--stitch', '--stitching', dest='stitch', default=False, action='store_true')
    data_parser.add_argument('--of', '--over', '--oversample', dest='oversampling_factor', type=int, default=None)
    data_parser.add_argument('-k', '--kc', '--kepcorr', dest='kep_corr', default=False, action='store_true')
    data_parser.add_argument('-f', '--force', dest='force', default=False, action='store_true')
    data_parser.add_argument('--dnu', dest='dnu', nargs='*', type=float, default=None)
    data_parser.add_argument('--le', '--lowere', dest='lower_ech', nargs='*', default=None, type=float)
    data_parser.add_argument('--ue', '--uppere', dest='upper_ech', nargs='*', default=None, type=float)
    data_parser.add_argument('-n', '--notch', dest='notching', default=False, action='store_true')

    # MAIN PARSER
    main_parser = argparse.ArgumentParser()
    # ESTIMATE PARAMETERS
    estimate = main_parser.add_argument_group()
    estimate.add_argument('-e', '--est', '--estimate', dest='estimate', default=True, action='store_false')
    estimate.add_argument('-j', '--adjust', dest='adjust', default=False, action='store_true')
    estimate.add_argument('--def', '--defaults', dest='defaults', type=str, default=None)
    estimate.add_argument('--sw', '--smoothwidth', dest='smooth_width', default=20.0, type=float)
    estimate.add_argument('--bin', '--binning', dest='binning', default=0.005, type=float)
    estimate.add_argument('--bm', '--mode', '--bmode', choices=["mean", "median", "gaussian"], dest='bin_mode', default='mean', type=str)
    estimate.add_argument('--step', '--steps', dest='step', default=0.25, type=float)
    estimate.add_argument('--trials', '--ntrials', dest='n_trials', default=3, type=int)
    estimate.add_argument('-a', '--ask', dest='ask', default=False, action='store_true')
    estimate.add_argument('--lx', '--lowerx', dest='lower_ex', nargs='*', default=None, type=float)
    estimate.add_argument('--ux', '--upperx', dest='upper_ex', nargs='*', default=None, type=float)

    # STELLAR BACKGROUND FITTING
    background = main_parser.add_argument_group()
    background.add_argument('-b', '--bg', '--background', dest='background', default=True, action='store_false')
    background.add_argument('--basis', dest='basis', default='tau_sigma', type=str)
    background.add_argument('--bf', '--box', '--boxfilter', dest='box_filter', default=1.0, type=float)
    background.add_argument('--iw', '--indwidth', dest='ind_width', default=20.0, type=float)
    background.add_argument('--rms', '--nrms', dest='n_rms', default=20, type=int)
    background.add_argument('--laws', '--nlaws', dest='n_laws', default=None, type=int)
    background.add_argument('--wn', '--fixwn', dest='fix_wn', default=False, action='store_true')
    background.add_argument('--metric', choices=['aic','bic'], dest='metric', default='bic', type=str)
    background.add_argument('--lb', '--lowerb', dest='lower_bg', nargs='*', default=None, type=float)
    background.add_argument('--ub', '--upperb', dest='upper_bg', nargs='*', default=None, type=float)

    # GLOBAL PARAMETERS
    globe = main_parser.add_argument_group()
    globe.add_argument('-g', '--globe', '--global', dest='globe', default=True, action='store_false')
    globe.add_argument('--numax', dest='numax', nargs='*', default=None, type=float)
    globe.add_argument('--lp', '--lowerp', dest='lower_ps', nargs='*', default=None, type=float)
    globe.add_argument('--up', '--upperp', dest='upper_ps', nargs='*', default=None, type=float)
    globe.add_argument('--ew', '--exwidth', dest='ex_width', default=1.0, type=float)
    globe.add_argument('--sm', '--smpar', dest='sm_par', default=None, type=float)
    globe.add_argument('--sp', '--smoothps', dest='smooth_ps', type=float, default=2.5)
    globe.add_argument('--fft', dest='fft', default=True, action='store_false')
    globe.add_argument('--thresh', '--threshold', dest='threshold', default=1.0, type=float)
    globe.add_argument('--peak', '--peaks', '--npeaks', dest='n_peaks', default=5, type=int)

    # SAMPLING
    mcmc = main_parser.add_argument_group()
    mcmc.add_argument('--mc', '--iter', '--mciter', dest='mc_iter', default=1, type=int)
    mcmc.add_argument('-m', '--samples', dest='samples', default=False, action='store_true')
    mcmc.add_argument('--nt', '--nthread', '--nthreads', dest='n_threads', type=int, default=0)
    mcmc.add_argument('--seed', dest='seed', default=None, type=int)

    # PLOTTING
    plot_parser = argparse.ArgumentParser()
    plot_parser.add_argument('--showall', dest='show_all', default=False, action='store_true')
    plot_parser.add_argument('-d', '--show', '--display', dest='show', default=False, action='store_true')
    plot_parser.add_argument('--cm', '--color', dest='cmap', default='binary', type=str)
    plot_parser.add_argument('--cv', '--value', dest='clip_value', default=3.0, type=float)
    plot_parser.add_argument('-y', '--hey', dest='hey', default=False, action='store_true')
    plot_parser.add_argument('-i', '--ie', '--interpech', dest='interp_ech', default=False, action='store_true')
    plot_parser.add_argument('--nox', '--nacross', dest='nox', default=None, type=int)
    plot_parser.add_argument('--noy', '--ndown', '--norders', dest='noy', default='0+0', type=str)
    plot_parser.add_argument('--npb', dest='npb', default=10, type=int)
    plot_parser.add_argument('-r', '--ridges', dest='ridges', default=False, action='store_true')
    plot_parser.add_argument('--se', '--smoothech', dest='smooth_ech', default=None, type=float)

    ###################
    # Different modes #
    ###################
    sub_parser = parser.add_subparsers(dest='mode')
    parser_check = sub_parser.add_parser('check', parents=[parent_parser, data_parser, plot_parser])
    parser_fun = sub_parser.add_parser('fun')
    parser_load = sub_parser.add_parser('load', parents=[parent_parser, data_parser, plot_parser])
    parser_parallel = sub_parser.add_parser('parallel', parents=[parent_parser, data_parser, main_parser, plot_parser])
    parser_plot = sub_parser.add_parser('plot', parents=[parent_parser, data_parser, plot_parser])
    parser_run = sub_parser.add_parser('run', parents=[parent_parser, data_parser, main_parser, plot_parser])
    parser_setup = sub_parser.add_parser('setup', parents=[parent_parser, data_parser])

    return parser


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = create_parser()

    def test_something(self):
        parsed = self.parser.parse_args(['-dv'])
        self.assertEqual(parsed.something, 'test')


"""
