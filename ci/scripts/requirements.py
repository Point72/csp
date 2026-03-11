#!/bin/env python
import os

def deps():
    import toml
    c = toml.load("pyproject.toml")
    requires = c["build-system"]["requires"]
    develop = c["project"]["optional-dependencies"]["develop"]
    return requires + develop

def main():
    ret = os.system("python -m pip install --prefer-binary toml")
    if ret != 0:
        raise ValueError("Python requirement install failed: see output for error")
    ret = os.system("python -m pip install --prefer-binary " + " ".join(f'"{_}"' for _ in deps()))
    if ret != 0:
        raise ValueError("Python requirement install failed: see output for error")


if __name__ == "__main__":
    main()
