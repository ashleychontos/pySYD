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
        cli.main()
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
def test_single_run():
    params = {
              'star' : 11618103,
              'seed' : 3454566, 
              'smooth_width' : 5.0,
              'mc_iter' : 1,
              'test' : True,
              'results' : {'numax_smooth':106.29, 'dnu':9.26},
    }
    star = params.pop('star')
    # Load relevant pySYD parameters
    args = Parameters()
    args.add_targets(stars=[star])
    target = Target(star, args, test=True)
    target = _load_data(target)
    target.lc, target.ps = False, False
    answers = params.pop('results')
    for param in list(params.keys()):       
        target.params[param] = params[param]
    df = pd.DataFrame(target.process_star())
    assert '%.2f' % (float(df.loc[0,'numax_smooth'])) == '%.2f' % (float(answers['numax_smooth'])), "Incorrect numax for single iteration"
    assert '%.2f' % (float(df.loc[0,'dnu'])) == '%.2f' % (float(answers['dnu'])), "Incorrect dnu for single iteration"

# TEST SINGLE STAR W/ SAMPLING
def test_sampler_run():
    params = {
              'star' : 1435467,
              'seed' : 737499, 
              'smooth_width' : 10.0,
              'lower_ex' : 100.0,
              'upper_ex' : 5000.0,
              'lower_bg' : 100.0,
              'mc_iter' : 200,
              'test' : True,
              'results' : {'numax_smooth':{'value':1299.90, 'error':59.81,}, 'dnu':{'value':70.68, 'error':0.75,},},
    }
    star = params.pop('star')
    # Load relevant pySYD parameters
    args = Parameters()
    args.add_targets(stars=[star])
    target = Target(star, args, test=True)
    target = _load_data(target)
    target.lc, target.ps = False, False
    answers = params.pop('results')
    for param in list(params.keys()): 
        target.params[param] = params[param]
    df = pd.DataFrame(target.process_star())
    numax, numaxerr = df.loc[0,'numax_smooth'], mad_std(df['numax_smooth'].values)
    dnu, dnuerr = df.loc[0,'dnu'], mad_std(df['dnu'].values)
    assert '%.2f +/- %.2f' % (float(numax), float(numaxerr)) == '%.2f +/- %.2f' % (float(answers['numax_smooth']['value']), float(answers['numax_smooth']['error'])), "Incorrect numax/error for sampler"
    assert '%.2f +/- %.2f' % (float(dnu), float(dnuerr)) == '%.2f +/- %.2f' % (float(answers['dnu']['value']), float(answers['dnu']['error'])), "Incorrect dnu/error for sampler"
