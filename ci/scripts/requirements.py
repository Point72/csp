#!/bin/env python
import os

def deps():
    import toml
    c = toml.load("pyproject.toml")
    requires = c["build-system"]["requires"]
    develop = c["project"]["optional-dependencies"]["develop"]
    return requires + develop

def main():
    os.system("python -m pip install toml")
    os.system("python -m pip install " + " ".join(f'"{_}"' for _ in deps()))


if __name__ == "__main__":
    main()
