import numpy as np
import pandas as pd


import pysyd
from pysyd import utils


def test_inheritance(convert=90.):
    args = utils.Parameters()
    assert float(args.constants['deg2rad'])*convert == np.pi/2., "Parameter inheritance is not working"
