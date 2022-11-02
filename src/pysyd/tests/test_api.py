import numpy as np
import pytest


from ..utils import Parameters


def test_inheritance(convert=90.):
    args = Parameters()
    assert float(args.constants['deg2rad'])*convert == np.pi/2., "Parameter inheritance is not working"
