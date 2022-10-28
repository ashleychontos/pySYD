# DEVELOPMENT BRANCH ---> SOURCE PACKAGE
print('\n\n COPYING FROM DEVELOPMENT BRANCH ---> SOURCE PACKAGE \n\n')
from pysyd import utils
cont = utils._ask_yesno('continue? ')

if cont:
    import os 
    scripts = ['cli','models','pipeline','plots','target','utils']
    rows, rows_cli = 18, 32

    _ROOT, _ = os.path.split(os.path.abspath(os.getcwd()))
    package = os.path.join(os.path.split(_ROOT)[0], 'pysyd')

    # copy scripts from dev -> src 
    for script in scripts:
        if script == 'cli':
            n = rows_cli
        else:
            n = rows
        # keep header from pysyd package
        with open(os.path.join(package, '%s.py'%script), "r") as f:
            lines = [line for line in f.readlines()]
        header = lines[:n]
        # copy body of development branch script
        with open(os.path.join(_ROOT, '%s.py'%script), "r") as f:
            lines = [line for line in f.readlines()]
        body = lines[n:]
        # smash together header & body
        lines = header+body
        with open(os.path.join(package, '%s.py'%script), "w") as f:
            for line in lines:
                f.write(line)

    import shutil
    # version is different
    src = os.path.join(_ROOT, 'version.py')
    dst = os.path.join(package, 'version.py')
    shutil.copy(src, dst)

    import glob
    # make sure data and dicts are up-to-date
    files = glob.glob(os.path.join(_ROOT, 'info', 'data', '*'))
    for file in files:
        dst = os.path.join(package, 'data', os.path.split(file)[-1])
        shutil.copy(file, dst)
    files = glob.glob(os.path.join(_ROOT, 'dicts', '*'))
    for file in files:
        dst = os.path.join(package, 'dicts', os.path.split(file)[-1])
        shutil.copy(file, dst)