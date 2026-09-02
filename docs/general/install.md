# Installation

Echopop is available for installation via pip, conda-forge, or by cloning the repository for development.

```{danger}
Echopop supports Python 3.12, 3.13, and 3.14. Python 3.14 users should install with
conda-forge because Cartopy does not yet publish Python 3.14 wheels on PyPI. Python 3.15 is not yet
supported.
```

## pip

[![PyPI version](https://img.shields.io/pypi/v/echopop)](https://pypi.org/project/echopop/)

Install Echopop directly from PyPI:

```shell
pip install echopop
```

Python 3.14 users should use the conda-forge installation below until Cartopy publishes Python 3.14
wheels on PyPI.

## conda-forge
[![Conda version](https://img.shields.io/conda/vn/conda-forge/echopop)](https://anaconda.org/conda-forge/echopop)

Install Echopop from conda-forge with either Anaconda or Miniconda:

```shell
conda install -c conda-forge echopop
```

### Python 3.14

Python 3.14 installations must currently use conda-forge because Cartopy does not yet publish
Python 3.14 wheels on PyPI. Create a dedicated environment with:

```shell
conda create -n echopop-py314 -c conda-forge python=3.14 python-gil echopop
conda activate echopop-py314
```

The command above requires a released conda-forge Echopop package that supports Python 3.14. To
install the latest source before that package is available, clone the repository and run:

```shell
conda create -n echopop-py314 -c conda-forge python=3.14 python-gil cartopy geopandas
conda activate echopop-py314
python -m pip install -e .
```

The ``python-gil`` package selects the standard CPython build rather than the experimental
free-threaded build. Installing Cartopy and GeoPandas first ensures that their compiled
dependencies come from conda-forge. The editable pip installation then installs the remaining
dependencies declared in ``pyproject.toml``. Python 3.12 and 3.13 users can continue to use the pip
installation described above.

```{attention}
We recommend using the ``libmamba`` solver instead of the classic solver.
   See instructions [here](https://conda.github.io/conda-libmamba-solver/getting-started/)
   for installation and usage.
```

## Latest source
[![GitHub release](https://img.shields.io/github/v/release/echostack-org/echopop)](https://github.com/echostack-org/echopop/releases)

If you need the latest development version or want to contribute, clone the repository and install from source:

```shell
# Clone the repository
git clone https://github.com/echostack-org/echopop.git
cd echopop

# Create and activate a conda environment
conda create -c conda-forge -n echopop --yes python=3.12
conda activate echopop

# Optional: Install ipykernel for JupyterLab support
conda install -c conda-forge ipykernel

# Install Echopop in editable mode (-e)
pip install -e .
```

For a full development environment that includes testing and documentation tools, use:

```shell
conda create -c conda-forge -n echopop-dev --yes python=3.12
conda activate echopop-dev
pip install -e ".[dev,docs]"
```

:::{admonition} Development mode
:class: example
The `-e` flag means that you are installing Echopop in a development mode, which allows you to not only use but also develop and edit the source code.
:::

## Optional: Using WSL on Windows

For Windows users, consider creating environments in Windows Subsystem for Linux (WSL) for improved compatibility with Linux-based tools. See the [official WSL documentation](https://docs.microsoft.com/en-us/windows/wsl/) for setup instructions.
