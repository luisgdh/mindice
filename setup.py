from setuptools import setup, find_packages
import os

# Helper to grab the version without importing the whole library
def get_version():
    with open("mindice/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
setup(
    name="mindice",
    version=get_version()
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas'
    ],
)
