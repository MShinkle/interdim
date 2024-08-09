import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
setup(
    name="interdim",
    install_requires=install_requires,
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    author="Matthew Shinkle",
    author_email="matthewshinkle@gmail.com",
    description="InterDim is a streamlined package for interactive exploration of latent data dimensions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MShinkle/interdim.git",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    # python_requires='>=3.6',
)