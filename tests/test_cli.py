import sys
import warnings
import subprocess

import pysyd
import pysyd.cli
from pysyd import utils

warnings.simplefilter('ignore')

__all__ = ["test_cli","test_help","test_defaults","test_dictpath","test_run"]


def test_cli():
    """Test basic argument parsing"""
    sys.argv = ["pysyd", "--version"]
    try:
        pysyd.cli.main()
    except SystemExit:
        pass


def test_help():
    """Test command-line install"""
    cmd = 'pysyd --help'
    proc = subprocess.Popen(cmd.split())
    proc.wait()
    status = proc.poll()
    out, _ = proc.communicate()
    assert status == 0, "{} failed with exit code {}".format(cmd, status)


def test_defaults():
    args = utils.Parameters()
    assert has_attr(args, 'constants'), "Loading default parameter class failed"


def test_dictpath():
    from pysyd import DICTDIR
    try:
        os.path.exists(DICTDIR)
    except ImportError:
        print("Could not find path to pysyd dict files")


def test_run(stars=[1435467,2309595,11618103], results={}, answers={}):
    defaults = utils.get_dict(type='tests')              # get known answers + defaults
    for star in stars:
        # Load relevant pySYD parameters
        args = utils.Parameters()
        args.add_stars(stars=[star])
        args.params['test'] = True                       # to return results dict
        answers[star] = defaults[star].pop('results')
        for rest in defaults:                            # make sure to copy seed
            args.params[rest] = defaults[star][rest]
        star = Target(star, args)
        result = star.process_star()
        results[star] = {}
        df = pd.DataFrame(result)
        for param in list(answers[star]):
            results[star].update({'%s'%param:df.loc[0,param],'%s_err'%param:mad_std(df[param].values)})
    # make sure answers are identical
    for star in stars:
        numax, numaxerr = results[star]['numax_smooth'], results[star]['numax_smooth_err']
        dnu, dnuerr = results[star]['dnu'], results[star]['dnu_err']
        assert '%.2f +/- %.2f' % (float(numax), float(numaxerr)) == \
               '%.2f +/- %.2f' % (float(answers[star]['numax_smooth']['value']), float(answers[star]['numax_smooth']['error'])), \
               "Incorrect numax for star %d"%star
        assert '%.2f +/- %.2f' % (float(dnu), float(dnuerr)) == \
               '%.2f +/- %.2f' % (float(answers[star]['dnu']['value']), float(answers[star]['dnu']['error'])), \
               "Incorrect dnu for star %d"%star