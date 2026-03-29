"""Setup configuration for homotopy_regressors package."""

from setuptools import setup, find_packages

setup(
    name="homotopy_regressors",
    version="0.2.0",
    author="Rodolfo H. Rodrigo",
    author_email="rrodrigo@unsj.edu.ar",
    description=(
        "Unified HAM library for nonlinear ODE systems: "
        "discrete numerical regressors + continuous analytic series"
    ),
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/rodrigo1964-ai/ham-mlp-regressor",
    packages=find_packages(),
    package_data={
        '': ['*.md', '*.txt'],
    },
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "sympy>=1.10",
    ],
    extras_require={
        "plot": ["matplotlib>=3.4"],
        "test": ["pytest>=7.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords=[
        "homotopy analysis method", "HAM", "ODE solver",
        "nonlinear dynamics", "control systems", "embedded systems",
        "Padé approximant", "series solution",
    ],
)
