
from setuptools import setup

setup(
    name='python-test-project',
    version='1.0.0',
    author='Thomas Simmer',
    description=("A Python project to understand signal processing basics."),
    license="BSD",
    url="https://gitlab.com/thomas_simmer/python-test-project",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="python scipy numpy",
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
)