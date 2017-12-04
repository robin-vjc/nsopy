from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='nsopy',
    version='1.0',
    description='Non-smooth optimization for Python',
    # url='http://github.com/storborg/funniest',
    author='Robin Vujanic',
    author_email='vjc.robin@gmail.com',
    install_requires=required,
    packages=find_packages(),
)
