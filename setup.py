import setuptools

exec(open("pysyd/version.py").read())

setuptools.setup(
    name="pysyd",
    version=__version__,
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
        'numpy>=1.14.5',
        'pandas>=1.0.5',
        'scipy',
        'tqdm',
        'matplotlib>=1.5.3',
    ],
    extras_require = {
        'sampling': ['tqdm'],
        'plotting': ['matplotlib>=1.5.3', 'gridspec'],
    },
#    setup_requires = ['pytest-runner', 'flake8'],
#    tests_require = ['pytest'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ["data/*.csv", "data/*.mplstyle", "data/*.txt"]},
    entry_points={'console_scripts':['pysyd=pysyd.cli:main']},
    python_requires=">=3",
)
