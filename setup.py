#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
setup script for data processing package
"""
import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='mg_dataprocessing',
    version='1.0.0',
    
    packages=find_packages(), #automagically include all subfolders as packages
    
    license='MIT',
    long_description=read('README.txt'),
    
    author='Malte Gotz',
    author_email='malte.gotz@oncoray.de',
    url='https://github.com/mgotz/PyDataProcessing',
    
    install_requires=['matplotlib','scipy','numpy']
)