from setuptools import setup, find_packages

NAME = 'jigsaw'
VERSION = '1.0'
REQUIRED_PACKAGES = ['tensorflow-transform==0.9.0']

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
)
