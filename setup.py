import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyhydroqc",
    version="0.0.4",
    author="Amber Jones",
    author_email="amber.jones@usu.edu",
    description="A package containing functions for anomaly detection and correction of aquatic sensor data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmberSJones/pyhydroqc",
    packages=setuptools.find_packages(include=['pyhydroqc']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology"
    ],
    python_requires='>=3.6',
    install_requires=[
        'sklearn',
        'tensorflow',
        'matplotlib',
        'scipy',
        'pmdarima',
        'statsmodels',
        'numpy',
        'pandas'
    ],
)


