import os
import sys
import requests
import warnings
import subprocess
import numpy as np
import pandas as pd
from astropy.stats import mad_std

from .. import cli
from ..utils import Parameters, get_dict
from ..target import Target

warnings.simplefilter('ignore')



# TEST BASIC ARGUMENT PARSING
def test_cli():
    sys.argv = ["pysyd", "--version"]
    try:
        pysyd.cli.main()
    except SystemExit:
        pass

# TEST COMMAND LINE INSTALL
def test_help(
    cmd = 'pysyd --help',
    ):
    proc = subprocess.Popen(cmd.split())
    proc.wait()
    status = proc.poll()
    out, _ = proc.communicate()
    assert status == 0, "{} failed with exit code {}".format(cmd, status)

# TEST PARAMETER INHERITANCE
def test_inheritance():
    args = Parameters()
    assert hasattr(args, 'constants'), "Loading default parameter class failed"

# TEST CLI OPTIONS + DEFAULTS
def test_defaults():
    args = Parameters()
    options = get_dict(type='columns')['all']
    for option in options:
        assert option in args.params, "%s option not available via command line" % option

# TEST ABILITY TO LOCATE PACKAGE DATA
def test_info():
    from pysyd import DICTDIR
    try:
        os.path.exists(DICTDIR)
    except ImportError:
        print("Could not find path to pysyd info files")

# TEST PATH TO EXAMPLE DATA IN SOURCE REPO
def test_input(star=1435467, source='https://raw.githubusercontent.com/ashleychontos/pySYD/master/dev/data/'):
    request = requests.get(os.path.join(source, '%s_%s.txt' % (str(star), 'LC')))
    assert request.status_code == requests.codes.ok, "Example data not found"

# LOAD EXAMPLE DATA FROM GITHUB REBO
def _load_data(target, data={}, source='https://raw.githubusercontent.com/ashleychontos/pySYD/master/dev/data/',
               d={'LC':{'x':'time','y':'flux'},'PS':{'x':'frequency','y':'power'}}):
    for filetype in ['LC','PS']:
        request = requests.get(os.path.join(source, '%s_%s.txt' % (str(target.name), filetype)))
        lines = request.text.split('\n')[:-1]
        x, y = [float(line.split()[0]) for line in lines], [float(line.split()[-1]) for line in lines]
        data.update({d[filetype]['x']:np.array(x),d[filetype]['y']:np.array(y)})
    target.time, target.flux = np.copy(data['time']), np.copy(data['flux'])
    target.frequency, target.power = np.copy(data['frequency']), np.copy(data['power'])
    target = _fix_data(target)
    return target

# CALCULATE PARAMETERS NEEDED FOR MAIN PIPELINE EXECUTION
def _fix_data(target):
    target.time -= min(target.time)
    target.cadence = int(round(np.nanmedian(np.diff(target.time)*24.0*60.0*60.0),0))
    target.nyquist = 10**6./(2.0*target.cadence)
    target.baseline = (max(target.time)-min(target.time))*24.*60.*60.
    target.tau_upper = target.baseline/2.
    target.params['oversampling_factor'] = int((1./((max(target.time)-min(target.time))*0.0864))/(target.frequency[1]-target.frequency[0]))
    target.freq_os, target.pow_os = np.copy(target.frequency), np.copy(target.power)
    target.freq_cs = np.array(target.frequency[target.params['oversampling_factor']-1::target.params['oversampling_factor']])
    target.pow_cs = np.array(target.power[target.params['oversampling_factor']-1::target.params['oversampling_factor']])
    return target

# TEST SINGLE STAR W/ NO SAMPLING
def test_single_run(
    star = 11618103,
    has_lc = False,       # disables autosaving to a dictionary that does not exist
    has_ps = False,
    test = True,
    mc_iter = 1,
    ):
    defaults = get_dict(type='tests')[star]                 # get known answers + defaults
    # Load relevant pySYD parameters
    args = Parameters()
    args.add_targets(stars=[star])
    target = Target(star, args, test=True)
    target = _load_data(target)
    target.lc, target.ps, target.params['test'], target.params['mc_iter'] = has_lc, has_ps, test, mc_iter
    answers = defaults.pop('results')
    for param in list(defaults.keys()):                     # make sure to copy seed
        target.params[param] = defaults[param]
    result = target.process_star()
    df = pd.DataFrame(result)
    numax, dnu = df.loc[0,'numax_smooth'], df.loc[0,'dnu']
    assert '%.2f' % (float(numax)) == '%.2f' % (float(answers['numax_smooth']['value'])), "Incorrect numax for Target %d"%star
    assert '%.2f' % (float(dnu)) == '%.2f' % (float(answers['dnu']['value'])), "Incorrect dnu for Target %d"%star

# TEST SINGLE STAR W/ SAMPLING
def test_sampler_run(
    star = 1435467,
    has_lc = False,       # disables autosaving to a dictionary that does not exist
    has_ps = False,
    test = True,
    mc_iter = 200,
    ):
    defaults = get_dict(type='tests')[star]                 # get known answers + defaults
    # Load relevant pySYD parameters
    args = Parameters()
    args.add_targets(stars=[star])
    target = Target(star, args, test=True)
    target = _load_data(target)
    target.lc, target.ps, target.params['test'], target.params['mc_iter'] = has_lc, has_ps, test, mc_iter
    answers = defaults.pop('results')
    for param in list(defaults.keys()):                     # make sure to copy seed
        target.params[param] = defaults[param]
    result = target.process_star()
    df = pd.DataFrame(result)
    numax, numaxerr = df.loc[0,'numax_smooth'], mad_std(df['numax_smooth'].values)
    dnu, dnuerr = df.loc[0,'dnu'], mad_std(df['dnu'].values)
    assert '%.2f +/- %.2f' % (float(numax), float(numaxerr)) == '%.2f +/- %.2f' % (float(answers['numax_smooth']['value']), float(answers['numax_smooth']['error'])), \
           "Incorrect numax for Target %d"%star
    assert '%.2f +/- %.2f' % (float(dnu), float(dnuerr)) == '%.2f +/- %.2f' % (float(answers['dnu']['value']), float(answers['dnu']['error'])), \
           "Incorrect dnu for Target %d"%star
