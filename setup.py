import setuptools

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

reqs = []
for line in open('requirements.txt', 'r').readlines():
    reqs.append(line)

setuptools.setup(
    name="pysyd",
    version="0.4.1",
    license="MIT",
    author="Ashley Chontos",
    author_email="achontos@hawaii.edu",
    description="Automated extraction of global asteroseismic parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashleychontos/pysyd",
    project_urls={
        "Documentation": "https://readthedocs.org/projects/syd-pypline",
        "Source": "https://github.com/ashleychontos/pysyd",
        "Bug Tracker": "https://github.com/ashleychontos/pysyd/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=reqs,
    packages=setuptools.find_packages(),
    entry_points={'console_scripts':['pysyd=pysyd.cli:main']},
    python_requires=">=3.6",
)
