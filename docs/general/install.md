# Installation

Echopop is available for installation via pip, conda-forge, or by cloning the repository for development.

```{danger}
Ensure you have Python 3.12 or 3.13 installed, as Python 3.14 has known dependency issues.
```

## pip

[![PyPI version](https://img.shields.io/pypi/v/echopop)](https://pypi.org/project/echopop/)

Install Echopop directly from PyPI:

```shell
pip install echopop
```

## conda-forge
[![Conda version](https://img.shields.io/conda/vn/conda-forge/echopop)](https://anaconda.org/conda-forge/echopop)

Install Echopop from conda-forge with either Anaconda or Miniconda:

```shell
conda install -c conda-forge echopop
```
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
