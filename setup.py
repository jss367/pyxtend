import os

from setuptools import find_packages, setup

version = {}
with open(os.path.join("src", "pyxtend", "_version.py")) as fp:
    exec(fp.read(), version)

setup(
    name="pyxtend",
    version=version["__version__"],
    author="Julius",
    author_email="julius.simonelli@gmail.com",
    description="Some functions for Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jss367/pyxtend",
    project_urls={
        "Bug Tracker": "https://github.com/jss367/pyxtend/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
