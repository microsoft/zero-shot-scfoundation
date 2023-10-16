from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='sc_foundation_evals',
    version='0.1',
    packages=find_packages(),
    # install_requires=required_packages,
    author='Kasia Kedzierska',
    author_email='kasia@well.ox.ac.uk',
    description='Evaluations of foundation models in single-cell biology',
)
