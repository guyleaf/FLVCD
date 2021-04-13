#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='UTWRS',
    version='0.0.0',
    description='Universal Transformer for pytorch',
    author='GuyLeaf',
    author_email='ychhua1@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/guyleaf/UTWRS',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

