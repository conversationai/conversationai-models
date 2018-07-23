from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.2.0', 'comet-ml==1.0.16', 'nltk>=3.3',
                      'typed_ast==1.1.0', 'tensorflow-gpu==1.5']

setup(
    name='tf_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Common Keras model + TF Estimator modelling framework.')
