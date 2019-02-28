from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'nltk>=3.3', 'typed_ast==1.1.0', 'tensorflow-hub==0.1.1'
]

setup(
    name='tf_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='TF Estimator modelling framework.')
