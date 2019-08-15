from setuptools import setup, find_packages

# with open('./requirements.txt') as f:
#     required = f.read().splitlines()

REQUIRES = [
    'colorama >= 0.3.7',
    'numpy >= 1.11.2',
    'pandas >= 0.19.0',
    'python-dateutil >= 2.5.3',
    'pytz >= 2016.7',
    'six >= 1.10.0',
]

setup(
    name='nsopy',
    version='1.50',
    description='Non-smooth optimization for Python',
    author='Robin Vujanic',
    author_email='vjc.robin@gmail.com',
    install_requires=REQUIRES,
    url='https://github.com/robin-vjc/nsopy',  # use the URL to the github repo
    download_url='https://github.com/robin-vjc/nsopy/archive/1.21.tar.gz',
    keywords=['non-smooth', 'distributed', 'optimization', 'python'],
    packages=find_packages(),
)

