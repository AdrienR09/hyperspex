from setuptools import setup, find_packages

setup(
    name="hyperspex",
    version="0.1.0",
    author="Adrien Rousseau",
    description="A Python package to help 3D hyperspectral data exploration through a Graphical User Interface.",
    packages=find_packages(),
    install_requires=[
    "numpy",
    "scipy",
    "matplotlib",
    "lmfit",
    ],
)
