from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='nsopy',
    version='1.1',
    description='Non-smooth optimization for Python',
    # url='http://github.com/storborg/funniest',
    author='Robin Vujanic',
    author_email='robin@acfr.usyd.edu.au',
    install_requires = required,
    packages=find_packages(),
    # packages=['dmp'],
)

# Version 2.0: when we have implemented a working version of the transport planners