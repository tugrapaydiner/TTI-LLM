# Setup script for building C++ kernels - run: python setup.py build_ext --inplace
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys

ext_modules = [
    Pybind11Extension(
        "cpp_kernels",
        ["src/kernels/cpp_kernels.cpp"],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"] if sys.platform != "win32" else ["/O2", "/arch:AVX2", "/fp:fast"],
    ),
]

setup(
    name="tti-llm-kernels",
    version="0.1.0",
    description="TTI-LLM C++ Kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)

