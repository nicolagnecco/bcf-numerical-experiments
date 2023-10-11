from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(include=["src", "src.*"]),
    version="0.1.0",
    description="Implementation of stable trees,",
    author="Nicola Gnecco, Sebastian Engelke, Jonas Peters, and Niklas Pfister",
    license="MIT",
)
