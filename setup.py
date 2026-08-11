"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='scipion-chem-metoken',
    version='0.2.0',
    description='Scipion framework plugin for PTM prediction with MeToken',
    long_description=long_description,
    url='https://github.com/Lvera-code/scipion-chem-metoken',
    author='Enzo Sierra',
    author_email='enzogael57@gmail.com',
    keywords='scipion ptm post-translational-modification metoken',
    packages=find_packages(),
    install_requires=[requirements],
    include_package_data=True,
    entry_points={
        'pyworkflow.plugin': 'metoken = metoken'
    }
)
