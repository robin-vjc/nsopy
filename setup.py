from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='nsopy',
    version='1.0',
    description='Non-smooth optimization for Python',
    author='Robin Vujanic',
    author_email='vjc.robin@gmail.com',
    install_requires=required,
    url='https://github.com/robin-vjc/nsopy',  # use the URL to the github repo
    download_url='https://github.com/robin-vjc/nsopy/archive/0.1.tar.gz',
    keywords=['non-smooth', 'distributed', 'optimization', 'python'],
    packages=find_packages(),
)
