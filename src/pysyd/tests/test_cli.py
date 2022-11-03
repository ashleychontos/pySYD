import os
import sys
import requests
import warnings
import subprocess
import pytest

import pysyd

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


# TEST ABILITY TO LOCATE PACKAGE DATA
def test_info():
    from pysyd import DICTDIR
    try:
        os.path.exists(DICTDIR)
    except ImportError:
        print("Could not find path to pysyd info files")