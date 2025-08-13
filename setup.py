
from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-e'):
            reqs.append(line)
    return reqs

setup(
    name='spam_mlops',
    version='0.1.0',
    description='Spam classification project using ML and MLOps',
    author='Vasim-rgb',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.7',
)
