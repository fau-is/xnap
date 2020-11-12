
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xnap',
    version='1.0.0',
    description='Explainable next activity prediction using LSTMs and LRP (xnap)',
    long_description=readme,
    author='Sven Weinzierl',
    author_email='sven.weinzierl@fau.de',
    url='https://github.com/fau-is/xnap',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

