import re
import setuptools

exec(open("pysyd/version.py").read())

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

install_reqs = []
for line in open('requirements.txt', 'r').readlines():
    install_reqs.append(line)

setuptools.setup(
    name="pysyd",
    version=__version__,
    author="Ashley Chontos",
    author_email="achontos@hawaii.edu",
    description="automated measurements of global asteroseismic parameters",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashleychontos/pysyd",
    project_urls={
        "Documentation": "https://pysyd.readthedocs.io",
        "Source": "https://github.com/ashleychontos/pySYD",
        "Bug Tracker": "https://github.com/ashleychontos/pySYD/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_reqs,
    packages=setuptools.find_packages(),
#    package_data={"pysyd": ["data/*", "dicts/*"]},
#    data_files=[("",['data/pysyd_results.csv']),
#                ("",['data/idlsyd_results.csv'])],
    entry_points={'console_scripts':['pysyd=pysyd.cli:main']},
    python_requires=">=3.6",
)
