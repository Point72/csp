import multiprocessing
import os
import os.path
import platform
import subprocess
import sys
from shutil import which

from skbuild import setup

CSP_USE_VCPKG = os.environ.get("CSP_USE_VCPKG", "1").lower() in ("1", "on")
# Allow arg to override default / env
if "--csp-no-vcpkg" in sys.argv:
    CSP_USE_VCPKG = False
    sys.argv.remove("--csp-no-vcpkg")

# CMake Options
CMAKE_OPTIONS = (
    ("CSP_BUILD_NO_CXX_ABI", "0"),
    ("CSP_BUILD_TESTS", "1"),
    ("CSP_MANYLINUX", "0"),
    ("CSP_BUILD_ARROW_ADAPTER", "1"),
    ("CSP_BUILD_KAFKA_ADAPTER", "1"),
    ("CSP_BUILD_PARQUET_ADAPTER", "1"),
    ("CSP_BUILD_WS_CLIENT_ADAPTER", "1"),
    ("CSP_ENABLE_ASAN", "0"),
    ("CSP_ENABLE_UBSAN", "0"),
    # NOTE:
    # - omit vcpkg, need to test for presence
    # - omit ccache, need to test for presence
    # - omit coverage/gprof, not implemented
)

if sys.platform == "linux":
    VCPKG_TRIPLET = "x64-linux"
elif sys.platform == "win32":
    VCPKG_TRIPLET = "x64-windows-static-md"
else:
    VCPKG_TRIPLET = None

VCPKG_SHA = "9c5c2a0ab75aff5bcd08142525f6ff7f6f7ddeee"

# This will be used for e.g. the sdist
if CSP_USE_VCPKG:
    if not os.path.exists("vcpkg"):
        # Clone at the sha we want
        subprocess.call(
            [
                "git",
                "clone",
                "https://github.com/Microsoft/vcpkg.git",
            ],
        )
        subprocess.call(["git", "checkout", VCPKG_SHA], cwd="vcpkg")
    else:
        # Ensure that the sha matches what we expect
        # First get the sha
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd="vcpkg",
            )
            .decode("utf-8")
            .strip()
        )
        if sha != VCPKG_SHA:
            raise RuntimeError(f"vcpkg sha {sha} does not match expected {VCPKG_SHA}")
    if not os.path.exists("vcpkg/ports"):
        subprocess.call(["git", "submodule", "update", "--init", "--recursive"])
    if not os.path.exists("vcpkg/buildtrees"):
        subprocess.call(["git", "pull"], cwd="vcpkg")
        args = ["install"]
        if VCPKG_TRIPLET is not None:
            args.append(f"--triplet={VCPKG_TRIPLET}")

        if os.name == "nt":
            subprocess.call(["bootstrap-vcpkg.bat"], cwd="vcpkg", shell=True)
            subprocess.call(["vcpkg.bat"] + args, cwd="vcpkg", shell=True)
        else:
            subprocess.call(["./bootstrap-vcpkg.sh"], cwd="vcpkg")
            subprocess.call(["./vcpkg"] + args, cwd="vcpkg")


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

    if VCPKG_TRIPLET is not None:
        cmake_args.append(f"-DVCPKG_TARGET_TRIPLET={VCPKG_TRIPLET}")
else:
    cmake_args.append("-DCSP_USE_VCPKG=OFF")

if "CXX" in os.environ:
    cmake_args.append(f"-DCMAKE_CXX_COMPILER={os.environ['CXX']}")

if "DEBUG" in os.environ:
    cmake_args.append("-DCMAKE_BUILD_TYPE=Debug")

if platform.system() == "Windows":
    import distutils.msvccompiler as dm

    # https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_with_Visual_Studio_2015_.28x86.2C_x64.2C_ARM.29
    msvc = {
        "12": "Visual Studio 12 2013",
        "14": "Visual Studio 14 2015",
        "14.0": "Visual Studio 14 2015",
        "14.1": "Visual Studio 15 2017",
        "14.2": "Visual Studio 16 2019",
        "14.3": "Visual Studio 17 2022",
        "14.4": "Visual Studio 17 2022",
    }.get(str(dm.get_build_version()), "Visual Studio 17 2022")
    cmake_args.extend(
        [
            "-G",
            os.environ.get("CSP_GENERATOR", msvc),
        ]
    )

for cmake_option, default in CMAKE_OPTIONS:
    if os.environ.get(cmake_option, default).lower() in ("1", "on"):
        cmake_args.append(f"-D{cmake_option}=ON")
    else:
        cmake_args.append(f"-D{cmake_option}=OFF")

if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(multiprocessing.cpu_count())

if platform.system() == "Darwin":
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = os.environ.get("OSX_DEPLOYMENT_TARGET", "10.15")
    cmake_args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={os.environ.get('OSX_DEPLOYMENT_TARGET', '10.15')}")

if which("ccache") and os.environ.get("CSP_USE_CCACHE", "") != "0":
    cmake_args.append("-DCSP_USE_CCACHE=On")

print(f"CMake Args: {cmake_args}")

setup(
    name="csp",
    version="0.13.2",
    packages=["csp"],
    cmake_install_dir="csp",
    cmake_args=cmake_args,
    # cmake_with_sdist=True,
)
