import os
import re
import setuptools



def get_property(variable, project='pysyd'):
    fname = os.path.join('src', project, '__init__.py')
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(variable), open(fname).read())
    return result.group(1)


setuptools.setup(
    name="pysyd",
    version=get_property('__version__'),
    author="Ashley Chontos",
    author_email="achontos@hawaii.edu",
    description="automated measurements of global asteroseismic parameters",
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ashleychontos/pysyd",
    project_urls={
        "Documentation": "https://pysyd.readthedocs.io",
        "Source": "https://github.com/ashleychontos/pySYD",
        "Bug Tracker": "https://github.com/ashleychontos/pySYD/issues",
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
    install_requires = [
        'astropy>=4.0.0',
        'numpy>=1.23.0',
        'matplotlib>=1.5.3',
        'pandas>=1.0.5',
        'scipy>=1.9.0',
        'tqdm>=4.64.0',
    ],
    tests_require = [
        'pytest',
        'requests',
    ],
    packages=['pysyd'],
    package_dir={'pysyd':'src/pysyd'},
    package_data={'pysyd': ["data/*.csv", "data/*.mplstyle", "data/*.txt", "data/dicts/*.dict"]},
    entry_points={'console_scripts':['pysyd=pysyd.cli:main']},
    python_requires=">=3.8",
)
