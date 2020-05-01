from setuptools import setup

try:
    import firedrake
except ImportError:
    raise Exception("Firedrake is needed in order to run this code.")

from firedrake.utils import complex_mode

assert complex_mode, "Firedrake needs to support complex numbers"

setup(
    name = "scattering",
    version = "0.1",
    author = "Dario Bojanjac, Darko Janeković",
    author_email = "dario.bojanjac@fer.hr, darko.janekovic@fer.hr",
    description = "FEM package for solving Maxwell scattering problem",
    packages = ['scattering'],
    zip_safe = False)
