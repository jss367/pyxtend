import os

from setuptools import find_packages, setup

# Get the directory where setup.py is located
here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "src", "pyxtend", "_version.py"), encoding="utf-8") as fp:
    exec(fp.read(), version)


def load_requirements(filename):
    filepath = os.path.join(here, filename)
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read().splitlines()


setup(
    name="pyxtend",
    version=version["__version__"],
    author="Julius",
    author_email="julius.simonelli@gmail.com",
    description="Some functions for Python",
    long_description=open(os.path.join(here, "README.md"), encoding="utf-8").read(),
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
    extras_require={
        "test": load_requirements("requirements-test.txt"),
    },
)
