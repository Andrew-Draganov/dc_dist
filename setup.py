from setuptools import setup

setup(
    name='DensityDR',
    version='0.1.0',
    description='Density-connected dimensionality reduction',
    author='Anonymous',
    author_email='anonym@ous.com',
    install_requires=[
        'GradientDR==0.1.3.4',
        'tqdm',
        'sklearn',
    ],
)
