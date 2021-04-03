import setuptools

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

reqs = []
for line in open('requirements.txt', 'r').readlines():
    reqs.append(line)

setuptools.setup(
    name="astero",
    version="0.0.3",
    license="MIT",
    author="Ashley Chontos",
    author_email="achontos@hawaii.edu",
    description="Automated extraction of global asteroseismic parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashleychontos/sydpy",
    project_urls={
        "Documentation": "https://readthedocs.org/projects/syd-pypline",
        "Bug Tracker": "https://github.com/ashleychontos/sydpy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=reqs,
    entry_points={'console_scripts':['sydpy=sydpy.sydpy.py:main']},
    package_dir={"": "sydpy"},
    packages=setuptools.find_packages(where="sydpy"),
    python_requires=">=3.6",
)
