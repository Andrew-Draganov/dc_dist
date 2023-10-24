from setuptools import setup

setup(
    name='dc_dist',
    version='0.2.0',
    description='Density-Connected Distance Experiments',
    author='Andrew Draganov',
    author_email='draganovandrew@cs.au.dk',
    install_requires=[
        'GradientDR==0.1.3.4',
        'tqdm',
        'sklearn',
    ],
)
