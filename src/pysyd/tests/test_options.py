import os
import random
import requests
import numpy as np



from ..utils import Parameters
from ..target import Target




# TEST OPTIONAL DATA MANIPULATION METHODS
def _load_data(target, source='https://raw.githubusercontent.com/ashleychontos/pySYD/master/dev/data/'):
    request = requests.get(os.path.join(source, '%s_PS.txt' % str(target.name)))
    lines = request.text.split('\n')[:-1]
    target.frequency, target.power = \
        np.array([float(line.split()[0]) for line in lines]), np.array([float(line.split()[-1]) for line in lines])
    return target

# KEPLER ARTEFACT CORRECTION
def test_kepcorr(star=2309595, kep_corr=True, seed=2904822, lowf=[240.0,380.0], highf=[4530.0,4534.0],):
    args = Parameters()
    args.add_targets(stars=[star])
    args.params[star]['kep_corr'], args.params[star]['seed'] = kep_corr, seed
    target = Target(star, args, test=True)
    target = _load_data(target)
    # generate a handful (n=5) of medium amplitude artefacts at low frequencies
    indices = [i for i, freq in enumerate(target.frequency) if freq > lowf[0] and freq <= lowf[-1]]
    idxs = random.choices(indices, k=5)
    for idx in idxs:
        target.power[idx] *= 5000.0
    # generate a single large amplitude artefact at a high frequency
    indices = [i for i, freq in enumerate(target.frequency) if freq > highf[0] and freq <= highf[-1]]
    idx = random.choice(indices)
    target.power[idx] = 10.0**9
    # remember indices of injected signals
    indices = idxs + [idx]
    freq_before, pow_before = np.copy(target.frequency), np.copy(target.power)
    freq_after, pow_after = target.remove_artefact(freq_before, pow_before)
    # make sure injected signals were removed
    for idx in indices:
        assert pow_after[idx] < pow_before[idx], "Issue with artefact whitening module"

# WHITENING MODULE
def test_whiten(star=2309595, seed=2904822, dnu=36.82, ech_mask=[10.0,25.0], notching=False,):
    args = Parameters()
    args.add_targets(stars=[star])
    args.params[star]['ech_mask'], args.params[star]['dnu'], args.params[star]['seed'], args.params[star]['notching'] = ech_mask, dnu, seed, notching
    target = Target(star, args, test=True)
    target = _load_data(target)
    # Estimate white noise as average
    white = np.mean(target.power[(target.frequency >= max(target.frequency)-100.0)&(target.frequency <= max(target.frequency)-50.0)])
    # Take the provided dnu and "fold" the power spectrum
    folded_freq = np.copy(target.frequency)%target.params['dnu']
    mask = np.ma.getmask(np.ma.masked_inside(folded_freq, target.params['ech_mask'][0], target.params['ech_mask'][1]))
    freq_before, pow_before = np.copy(target.frequency), np.copy(target.power)
    freq_after, pow_after = target.whiten_mixed(freq_before, pow_before)
    # Since the PS has been "whitened" in a region of the folded PS, we expect that the summed
    # power in that region should now be less if it were removing signal
    assert np.sum(pow_after[mask]) < np.sum(pow_before[mask]), "Mixed mode module is not working properly"

# NOTCHING TECHNIQUE
def test_notch(star=2309595, seed=2904822, dnu=36.82, ech_mask=[10.0,25.0], notching=True,):
    args = Parameters()
    args.add_targets(stars=[star])
    args.params[star]['ech_mask'], args.params[star]['dnu'], args.params[star]['seed'], args.params[star]['notching'] = ech_mask, dnu, seed, notching
    target = Target(star, args, test=True)
    target = _load_data(target)
    # Estimate white noise as minimum (for notching)
    white = min(target.power[(target.frequency >= max(target.frequency)-100.0)&(target.frequency <= max(target.frequency)-50.0)])
    # Take the provided dnu and "fold" the power spectrum
    folded_freq = np.copy(target.frequency)%target.params['dnu']
    mask = np.ma.getmask(np.ma.masked_inside(folded_freq, target.params['ech_mask'][0], target.params['ech_mask'][1]))
    freq_before, pow_before = np.copy(target.frequency), np.copy(target.power)
    freq_after, pow_after = target.whiten_mixed(freq_before, pow_before)
    # Notching is more predictable since it sets all values to the minimum (i.e. white defined above)
    assert pow_after[mask][0] == white, "Notching module is not working properly"
