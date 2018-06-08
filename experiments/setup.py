from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.1.5', 'comet-ml==1.0.8', 'nltk>=3.3']

setup(
    name='tf_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Common Keras model + TF Estimator modelling framework.')
