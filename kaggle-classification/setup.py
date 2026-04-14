from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tflearn>=0.3.2', 'Keras==3.13.2', 'h5py==2.7.1', 'comet-ml==1.0.8',
    'nltk>=3.3'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='tflearn.')

setup(
    name='keras_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='tflearn.')

setup(
    name='tf_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='tflearn.')
