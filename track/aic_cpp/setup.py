from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

module = Pybind11Extension("aic_cpp", sources=["src/aic.cpp"])

setup(
    name="aic_cpp",
    version="1.0",
    description="Compute distance between two lines",
    ext_modules=[module]
)