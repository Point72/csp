CSP is written in Python and C++ with Python and C++ build dependencies. While prebuilt wheels are provided for end users, it is also straightforward to build CSP from either the Python [source distribution](https://packaging.python.org/en/latest/specifications/source-distribution-format/) or the GitHub repository.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Make commands](#make-commands)
- [Prerequisites](#prerequisites)
- [Building with Conda on Linux](#building-with-conda-on-linux)
  - [Install conda](#install-conda)
  - [Clone](#clone)
  - [Install build dependencies](#install-build-dependencies)
  - [Build](#build)
- [Building with a system package manager](#building-with-a-system-package-manager)
  - [Clone](#clone-1)
  - [Install build dependencies](#install-build-dependencies-1)
    - [Linux](#linux)
    - [MacOS](#macos)
  - [Install Python dependencies](#install-python-dependencies)
  - [Build](#build-1)
  - [Building on `aarch64` Linux](#building-on-aarch64-linux)
- [Lint and Autoformat](#lint-and-autoformat)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
  - [MacOS](#macos-1)
    - [vcpkg install failed](#vcpkg-install-failed)
    - [Building thrift:arm64-osx/thrift:x64-osx failed](#building-thriftarm64-osxthriftx64-osx-failed)
    - [CMake was unable to find a build program corresponding to "Unix Makefiles".](#cmake-was-unable-to-find-a-build-program-corresponding-to-unix-makefiles)

## Make commands

As a convenience, CSP uses a `Makefile` for commonly used commands. You can print the main available commands by running `make` with no arguments

```bash
> make

build                          build the library
clean                          clean the repository
fix                            run autofixers
install                        install library
lint                           run lints
test                           run the tests
```

## Prerequisites

CSP has a few system-level dependencies which you can install from your machine package manager. Other package managers like `conda`, `nix`, etc, should also work fine. Currently, CSP relies on the `GNU` compiler toolchain only.

## Building with Conda on Linux

The easiest way to get started on a Linux machine is by installing the necessary dependencies in a self-contained conda environment.

Tweak this script to create a conda environment, install the build dependencies, build, and install a development version of CSP into the environment.

### Install conda

```bash
mkdir ~/github
cd ~/github

# this downloads a Linux x86_64 build, change your architecture to match your development machine
# see https://conda-forge.org/miniforge/ for alternate download links

wget https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Mambaforge-23.3.1-1-Linux-x86_64.sh
chmod 755 Mambaforge-23.3.1-1-Linux-x86_64.sh
./Mambaforge-23.3.1-1-Linux-x86_64.sh -b -f -u -p csp_venv

. ~/github/csp_venv/etc/profile.d/conda.sh

# optionally, run this if you want to set up conda in your .bashrc
# conda init bash

conda config --add channels conda-forge
conda config --set channel_priority strict
conda activate base
```

### Clone

```bash
git clone https://github.com/Point72/csp.git
cd csp
git submodule update --init --recursive
```

### Install build dependencies

```bash
# Note the operating system, change as needed
# Linux and MacOS should use the unix dev environment spec
micromamba create -n csp -f conda/dev-environment-unix.yml
micromamba activate csp
```

### Build

```bash
make build

# on aarch64 linux, comment the above command and use this instead
# VCPKG_FORCE_SYSTEM_BINARIES=1 make build

# finally install into the csp_venv conda environment
make develop
```

If you didn’t do `conda init bash` you’ll need to re-add conda to your shell environment and activate the `csp` environment to use it:

```bash
. ~/github/csp_venv/etc/profile.d/conda.sh
conda activate csp

# make sure everything works
cd ~/github/csp
make test
```

## Building with a system package manager

### Clone

Clone the repo and submodules with:

```bash
git clone https://github.com/Point72/csp.git
cd csp
git submodule update --init --recursive
```

### Install build dependencies

#### Linux

**Debian/Ubuntu/etc**

```bash
# for vcpkg
sudo make dependencies-debian
# or
# sudo apt-get install -y automake bison cmake curl flex ninja-build tar unzip zip

# for g++
sudo apt install build-essential
```

**Fedora/RedHat/Centos/Rocky/Alma/etc**

```bash
# for vcpkg
sudo make dependencies-fedora
# or
# yum install -y automake bison cmake curl flex perl-IPC-Cmd tar unzip zip

# for g++
sudo dnf group install "Development Tools"
```

#### MacOS

**Homebrew**

```bash
# for vcpkg
make dependencies-mac
# or
# brew install bison cmake flex make ninja
```

### Install Python dependencies

Python build and develop dependencies are specified in the `pyproject.toml`, but you can manually install them:

```bash
make requirements

# or
# python -m pip install toml
# python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["build-system"]["requires"]))'`
# python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["develop"]))'`
```

Note that these dependencies would otherwise be installed normally as part of [PEP517](https://peps.python.org/pep-0517/) / [PEP518](https://peps.python.org/pep-0518/).

### Build

Build the python project in the usual manner:

```bash
make build

# or
# python setup.py build build_ext --inplace
```

### Building on `aarch64` Linux

On `aarch64` Linux the VCPKG_FORCE_SYSTEM_BINARIES environment variable must be set before running `make build`:

```bash
VCPKG_FORCE_SYSTEM_BINARIES=1 make build
```

## Lint and Autoformat

CSP has listing and auto formatting.

| Language | Linter | Autoformatter | Description |
| :------- | :----- | :------------ | :---------- |
| C++      | `clang-format` | `clang-format` | Style |
| Python   | `ruff`         | `ruff` | Style |
| Python   | `isort`         | `isort` | Imports |

**C++ Linting**

```bash
make lint-cpp
# or
# clang-format --dry-run -Werror -i -style=file `find ./cpp/ -name "*.*pp"`
```

**C++ Autoformatting**

```bash
make fix-cpp
# or
# clang-format -i -style=file `find ./cpp/ -name "*.*pp"`
```

**Python Linting**

```bash
make lint-py
# or
# python -m isort --check csp/ setup.py
# python -m ruff csp/ setup.py
```

**Python Autoformatting**

```bash
make fix-py
# or
# python -m isort csp/ setup.py
# python -m ruff format csp/ setup.py
```

## Testing

CSP has both Python and C++ tests. The bulk of the functionality is tested in Python, which can be run via `pytest`. First, install the Python development dependencies with

```bash
make develop
```

For full test coverage including certain adapters and integration/regression tests, you will need to install some additional dependencies and spin up some services. These dependencies include:

- [Graphviz](https://graphviz.org/): for generating static diagrams of graph structure
- [Docker Compose](https://docs.docker.com/compose/): for spinning up self-contained services against which to integration-test adapters

On Debian/Ubuntu Linux, run

```bash
sudo apt-get install -y graphviz
```

On Fedora/RedHat/Centos/Rocky/Alma/etc, run

```bash
yum install -y graphviz
```

On MacOS using Homebrew, run

```bash
brew install graphviz
```

**Python**

```bash
make test-py
# or
# python -m pytest -v csp/tests --junitxml=junit.xml

make coverage-py
# or python -m pytest -v csp/tests --junitxml=junit.xml --cov=csp --cov-report xml --cov-report html --cov-branch --cov-fail-under=80 --cov-report term-missing
```

Adapters might rely on external services. For example, the `kafka` adapter requires a cluster against which to test. We rely on [docker compose](https://docs.docker.com/compose/) for these services. We store docker compose files in the `ci` directory, and convenience `Makefile` commands are available:

```bash
make dockerup ADAPTER=kafka
# or
# docker compose -f ci/kafka/docker-compose.yml up -d

make dockerps ADAPTER=kafka
# or
# docker compose -f ci/kafka/docker-compose.yml ps

make dockerdown ADAPTER=kafka
# or
# docker compose -f ci/kafka/docker-compose.yml down
```

Note that tests may be skipped by default, so run with:

```bash
CSP_TEST_KAFKA=1 make test
# or
# CSP_TEST_KAFKA=1 python -m pytest -v csp/tests --junitxml=junit.xml
```

There are a few test flags available:

- **`CSP_TEST_KAFKA`**
- **`CSP_TEST_SKIP_EXAMPLES`**: skip tests of examples folder

## Troubleshooting

### MacOS

#### vcpkg install failed

Check the `vcpkg-manifest-install.log` files, and install the corresponding packages if needed.

For example, you may need to `brew install pkg-config`.

#### Building thrift:arm64-osx/thrift:x64-osx failed

```
Thrift requires bison > 2.5, but the default `/usr/bin/bison` is version 2.3.
```

Ensure the homebrew-installed bison (version >= 3.8) is **before** `/usr/bin/bison` on $PATH

On ARM: `export PATH="/opt/homebrew/opt/bison/bin:$PATH"`

On Intel: `export PATH="/usr/local/opt/bison/bin:$PATH"`

#### CMake was unable to find a build program corresponding to "Unix Makefiles".

Complete error message:

```bash
CMake Error: CMake was unable to find a build program corresponding to "Unix Makefiles".  CMAKE_MAKE_PROGRAM is not set.  You probably need to select a different build tool.
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
```

`vcpkg-manifest-install.log`:

```bash
CMake Error at scripts/cmake/vcpkg_execute_build_process.cmake:134
```

`install-arm64-osx-dbg-out.log`:

```bash
ld: Assertion failed: (resultIndex < sectData.atoms.size()), function findAtom, file Relocations.cpp, line 1336.
collect2: error: ld returned 1 exit status
```

This is a known bug: https://github.com/Homebrew/homebrew-core/issues/145991

Retry the build with `CXXFLAGS='-Wl,-ld_classic'`.

Apple Silicon (ARM):

```bash
CXX=/opt/homebrew/bin/g++-13 CXXFLAGS='-Wl,-ld_classic' make build
```

Intel:

```bash
CXX=/usr/local/opt/gcc/bin/g++-13 CXXFLAGS='-Wl,-ld_classic' make build
```
