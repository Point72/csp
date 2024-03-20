import multiprocessing
import os
import os.path
import platform
import subprocess
import sys
from shutil import which
from skbuild import setup

CSP_USE_VCPKG = os.environ.get("CSP_USE_VCPKG", "1") == "1"

# This will be used for e.g. the sdist
if CSP_USE_VCPKG:
    if not os.path.exists("vcpkg"):
        subprocess.call(["git", "clone", "https://github.com/Microsoft/vcpkg.git"])
    if not os.path.exists("vcpkg/ports"):
        subprocess.call(["git", "submodule", "update", "--init", "--recursive"])
    if not os.path.exists("vcpkg/buildtrees"):
        subprocess.call(["git", "pull"], cwd="vcpkg")
        if os.name == "nt":
            subprocess.call(["bootstrap-vcpkg.bat"], cwd="vcpkg")
            subprocess.call(["vcpkg", "install"], cwd="vcpkg")
        else:
            subprocess.call(["./bootstrap-vcpkg.sh"], cwd="vcpkg")
            subprocess.call(["./vcpkg", "install"], cwd="vcpkg")


python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
cmake_args = [f"-DCSP_PYTHON_VERSION={python_version}"]
vcpkg_toolchain_file = os.path.abspath(
    os.environ.get(
        "CSP_VCPKG_PATH",
        os.path.join("vcpkg/scripts/buildsystems/vcpkg.cmake"),
    )
)

if CSP_USE_VCPKG and os.path.exists(vcpkg_toolchain_file):
    cmake_args.extend(
        [
            "-DCMAKE_TOOLCHAIN_FILE={}".format(vcpkg_toolchain_file),
            "-DCSP_USE_VCPKG=ON",
        ]
    )
else:
    cmake_args.append("-DCSP_USE_VCPKG=OFF")

if "CXX" in os.environ:
    cmake_args.append(f"-DCMAKE_CXX_COMPILER={os.environ['CXX']}")

if "DEBUG" in os.environ:
    cmake_args.append("-DCMAKE_BUILD_TYPE=Debug")

if "CSP_MANYLINUX" in os.environ:
    cmake_args.append("-DCSP_MANYLINUX=ON")

if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(multiprocessing.cpu_count())

if platform.system() == "Darwin":
    os.environ["OSX_DEPLOYMENT_TARGET"] = os.environ.get("OSX_DEPLOYMENT_TARGET", "10.13")
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = os.environ.get("OSX_DEPLOYMENT_TARGET", "10.13")

if hasattr(platform, "mac_ver") and platform.mac_ver()[0].startswith("14"):
    cmake_args.append("-DCSP_USE_LD_CLASSIC_MAC=ON")

if which("ccache"):
    cmake_args.append("-DCSP_USE_CCACHE=On")

print(f"CMake Args: {cmake_args}")

setup(
    name="csp",
    version="0.0.2",
    packages=["csp"],
    cmake_install_dir="csp",
    cmake_args=cmake_args,
    # cmake_with_sdist=True,
)
