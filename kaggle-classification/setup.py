from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tflearn>=0.3.2', 'Keras==2.1.5', 'h5py==2.7.1']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='tflearn.'
)

setup(
    name='keras-trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='tflearn.'
)
