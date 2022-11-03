import os
import random
import requests
import numpy as np
import pytest


from ..utils import Parameters, InputError, get_dict


# TEST PARAMETER INHERITANCE
def test_inheritance():
    args = Parameters()
    assert hasattr(args, 'constants'), "Loading default parameter class failed"

# TEST CLI OPTIONS + DEFAULTS
def test_defaults():
    params = Parameters()
    options = get_dict(type='columns')['all']
    for option in options:
        assert option in params.params, "%s option not available via command line" % option

# TEST ADD TARGETS
def test_targets():
    params = Parameters()
    params.params['save'] = False
    params.add_targets(stars=None)        # this would normally print input error msg
    assert 'stars' in params.params and params.params['stars'] is None, "Where are stars being provided"
    chuck = params.params.pop('stars')
    params.add_targets(stars=[1435467,2309595,11618103])
    assert 'stars' in params.params and params.params['stars'] is not None, "Not loading in target parameters"

