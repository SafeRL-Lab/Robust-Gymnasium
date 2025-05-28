"""Sets up the project."""
import pathlib
from setuptools import setup

from setuptools import setup, find_packages


CWD = pathlib.Path(__file__).absolute().parent
print("CWD-----:", CWD)

"""Robust-Gymnasium: Comprehensive and reliable environments for evaluating the effectiveness of 
robust reinforcement learning baselines."""

__version__ = '0.1.0'
__license__ = 'MIT License'
__author__ = 'Robust-Gymnasium Contributors'
__release__ = False

def get_version():
    """Gets the robust_gymnasium version."""
    path = CWD / "robust_gymnasium" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return long_description


setup(name="robust_gymnasium", version=get_version(), long_description=get_description(), include_package_data=True)
