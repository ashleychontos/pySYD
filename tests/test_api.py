import numpy as np



from pysyd import utils


def test_inheritance(convert=90.):
    args = utils.Parameters()
    assert float(args.constants['deg2rad'])*convert == np.pi/2., "Parameter inheritance is not working"
