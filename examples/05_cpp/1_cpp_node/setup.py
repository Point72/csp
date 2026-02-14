import os
import os.path
import sys

from skbuild import setup

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
cmake_args = [f"-DPYTHON_VERSION={python_version}"]

if "CXX" in os.environ:
    cmake_args.append(f"-DCMAKE_CXX_COMPILER={os.environ['CXX']}")

print(f"CMake Args: {cmake_args}")
setup(
    name="csp-example-piglatin",
    version="0.0.1",
    packages=["piglatin"],
    cmake_install_dir=".",
    cmake_args=cmake_args,
    # cmake_with_sdist=True,
)
