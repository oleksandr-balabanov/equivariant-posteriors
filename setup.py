#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="equivariant-posteriors",
    version="1.0",
    # Modules to import from other scripts:
    packages=find_packages(),
    # Executables
    scripts=[
        "test.py",
        "experiments/embed_d/spiral_embed_d.py",
        "experiments/mlp_dim/spiral_mlp_dim.py",
    ],
)
