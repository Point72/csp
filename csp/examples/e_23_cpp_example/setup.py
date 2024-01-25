import os
import os.path
import multiprocessing
import subprocess
import sys
import platform
from skbuild import setup
from shutil import which


python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
cmake_args = [f"-DPYTHON_VERSION={python_version}"]

if "CMAKE_ARGS" in os.environ:
    # conda
    cmake_args.extend(os.environ["CMAKE_ARGS"].split(" "))

if "CXX" in os.environ:
    cmake_args.append(f"-DCMAKE_CXX_COMPILER={os.environ['CXX']}")

if "DEBUG" in os.environ:
    cmake_args.append("-DCMAKE_BUILD_TYPE=Debug")

if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(multiprocessing.cpu_count())

if platform.system() == "Darwin":
    os.environ["OSX_DEPLOYMENT_TARGET"] = os.environ.get("OSX_DEPLOYMENT_TARGET", "10.13")
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = os.environ.get("OSX_DEPLOYMENT_TARGET", "10.13")

print(f"CMake Args: {cmake_args}")
setup(
    name="csp-example-piglatin",
    version="0.0.1",
    packages=["piglatin"],
    cmake_install_dir=".",
    cmake_args=cmake_args,
    # cmake_with_sdist=True,
)
