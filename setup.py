
from setuptools import setup, find_packages

setup(
    name='python-signal-processing-project',
    version='1.0.0',
    author='Thomas Simmer',
    description=("A Python project to understand signal processing basics."),
    license="BSD",
    url="https://gitlab.com/thomas_simmer/python-signal-processing-project",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="python scipy numpy",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
)