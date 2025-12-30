###################################################################################################################################################
# Project Configuration #
#########################
cmake_minimum_required(VERSION 3.20.0)
project(csp VERSION "0.13.2")
set(CMAKE_CXX_STANDARD 20)

###################################################################################################################################################
# CMake Dependencies #
######################
include(CheckCCompilerFlag)
include(CheckLinkerFlag)

###################################################################################################################################################
# CMake Policies #
##################

# option() should use new make behavior wrt variable clobbering
cmake_policy (SET CMP0077 NEW)

# Allow dep roots from env vars
cmake_policy (SET CMP0074 NEW)

# Set CMP0094 to NEW - find the first version that matches constraints,
# instead of the latest version installed
cmake_policy(SET CMP0094 NEW)

###################################################################################################################################################
# Environment #
###############
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(WIN32 ON)
    set(MACOS OFF)
    set(LINUX OFF)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(WIN32 OFF)
    set(MACOS ON)
    set(LINUX OFF)
else()
    set(WIN32 OFF)
    set(MACOS OFF)
    set(LINUX ON)
endif()


###################################################################################################################################################
# Paths #
#########
# Custom CMake modules
if(NOT DEFINED CSP_CMAKE_MODULE_PATH)
    set(CSP_CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cpp/cmake/modules")
endif()
list(PREPEND CMAKE_MODULE_PATH "${CSP_CMAKE_MODULE_PATH}")

###################################################################################################################################################
# Build Configuration #
#######################
find_package(Color)

# Build options
option(CMAKE_BUILD_TYPE "Release/Debug build" RELEASE)
option(CSP_BUILD_NO_CXX_ABI "Do not use CXX11 ABI" OFF)
option(CSP_BUILD_TESTS "Build tests" OFF)
option(CSP_BUILD_COVERAGE "Hook into gcov for coverage testing" OFF)
option(CSP_BUILD_GPROF "Hook into gprof for profiling" OFF)
option(CSP_MANYLINUX "Build for python's manylinux setup" OFF)
option(CSP_USE_VCPKG "Build with vcpkg dependencies" ON)
option(CSP_USE_CCACHE "Build with ccache caching" OFF)
option(CSP_USE_LD_CLASSIC_MAC "On macOS, link with ld_classic" OFF)
option(CSP_ENABLE_ASAN "Build with address sanitizer" OFF)
option(CSP_ENABLE_UBSAN "Build with undefined behavior sanitizer" OFF)

# Extension options
option(CSP_BUILD_KAFKA_ADAPTER "Build kafka adapter" ON)
option(CSP_BUILD_ARROW_ADAPTER "Build arrow adapter" ON)
option(CSP_BUILD_PARQUET_ADAPTER "Build parquet adapter" ON)
option(CSP_BUILD_WS_CLIENT_ADAPTER "Build ws client adapter" ON)

# Normalize build type for downstream comparisons
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)

# Python library config
set(BUILD_SHARED_LIBS TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_NAME_DIR "@rpath")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Normalize flags
string(REGEX REPLACE "[ ]*-O[^ ]+[ ]*" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "[ ]*-Wl,-O2 -Wl,[^ ]+[ ]*" " " CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
string(REGEX REPLACE "[ ]*-Wl,-O2 -Wl,[^ ]+[ ]*" " " CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

# CCache setup
if(CSP_USE_CCACHE)
    set(CMAKE_C_COMPILE_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif()

if(NOT DEFINED CSP_PYTHON_VERSION)
    set(CSP_PYTHON_VERSION 3.12)
endif()

# Path to python folder for autogen
if(NOT DEFINED CSP_PYTHON_FOLDER)
    set(CSP_PYTHON_FOLDER ${CMAKE_SOURCE_DIR}/csp)
    get_filename_component(CSP_PYTHON_FOLDER ${CSP_PYTHON_FOLDER} REALPATH BASE_DIR ${CMAKE_BINARY_DIR})
endif()

if(MACOS)
    # fix for threads on osx
    # assume built-in pthreads on MacOS
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)

    # don't link against build python
    # https://blog.tim-smith.us/2015/09/python-extension-modules-os-x/
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
endif()


###################################################################################################################################################
# Version Information #
#######################
# Set version from cmake and extract latest hash if available
set(CSP_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CSP_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CSP_VERSION_PATCH ${PROJECT_VERSION_PATCH})
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  # Get latest commit
  execute_process(COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE CSP_VERSION_COMMIT_SHA)
  # strip newline
  string(REGEX REPLACE "\n$" "" CSP_VERSION_COMMIT_SHA "${CSP_VERSION_COMMIT_SHA}")
else()
  set(CSP_VERSION_COMMIT_SHA "release")
endif()


###################################################################################################################################################
# RPath #
#########
if(MACOS)
    set(CMAKE_INSTALL_RPATH "@loader_path/")
elseif(LINUX)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()

###################################################################################################################################################
# Flags #
# Optimization Flags
if(WIN32)
    if(CMAKE_BUILD_TYPE_LOWER STREQUAL debug)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEBUG /Z7 /Zi")
        add_definitions(-DCSP_DEBUG)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
        add_definitions(-DNDEBUG)
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    foreach(warning 4244 4251 4267 4275 4290 4786 4305 4996)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd${warning}")
    endforeach(warning)
    add_compile_definitions(WIN32 _WIN32)

else()
    if(CSP_BUILD_NO_CXX_ABI)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
    if (COVERAGE)
        # TODO windows
        add_compile_options(--coverage)
        set(CMAKE_CXX_FLAGS "-O0 ${CMAKE_CXX_FLAGS}")
        link_libraries(gcov)
    endif ()
    if (GPROF_BUILD)
        set(CMAKE_CXX_FLAGS "-pg ${CMAKE_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "-pg ${CMAKE_EXE_LINKER_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "-pg ${CMAKE_SHARED_LINKER_FLAGS}")
    endif ()
    if(CMAKE_BUILD_TYPE_LOWER STREQUAL debug)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -O0 \
        -g3 \
        ")
        add_definitions(-DCSP_DEBUG)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -O3 \
        -g0 \
        -Wall \
        -Werror \
        -Wno-deprecated-declarations \
        -Wno-deprecated \
        ")
        add_definitions(-DNDEBUG)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
            -Wno-maybe-uninitialized \
            ")
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-assume-unique-vtables")
            endif()
        endif()
    endif()
endif()

if(CSP_ENABLE_ASAN)
    message(STATUS "Enabling Address Sanitizer")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fno-omit-frame-pointer)
endif()

if(CSP_ENABLE_UBSAN)
    message(STATUS "Enabling Undefined Behavior Sanitizer")
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined -fno-omit-frame-pointer)
endif()


###################################################################################################################################################
# Messages #
############
message("\n${Green}Building CSP version v${CSP_VERSION_MAJOR}.${CSP_VERSION_MINOR}.${CSP_VERSION_PATCH} [${CSP_VERSION_COMMIT_SHA}]")
message("\n${Green}Building C++ binding${ColorReset}")
message("\n${Green}Building Python ${Red}${CSP_PYTHON_VERSION}${Green} binding${ColorReset}")
if(CMAKE_BUILD_TYPE_LOWER STREQUAL debug)
    message("\n${Red}Building DEBUG${ColorReset}")
else()
    message("\n${Green}Building RELEASE${ColorReset}")
endif()
message("\n${Green}CMake Search Path: ${CMAKE_MODULE_PATH}${ColorReset}")

###################################################################################################################################################
# Helpers #
###########
find_package(csp_autogen REQUIRED)


###################################################################################################################################################
# Dependencies #
################
find_package(DepsBase REQUIRED)

# Adapter dependencies
if(CSP_BUILD_ARROW_ADAPTER)
    find_package(DepsArrowAdapter REQUIRED)
endif()

if(CSP_BUILD_KAFKA_ADAPTER)
    find_package(DepsKafkaAdapter REQUIRED)
endif()

if(CSP_BUILD_PARQUET_ADAPTER)
    find_package(DepsParquetAdapter REQUIRED)
endif()


# PYTHON
if(CSP_MANYLINUX)
    # Manylinux docker images have no shared libraries
    # The instead use a statically built python.
    # Cmake's default FindPython can't find the python headers
    # without also finding (or failing to find) the python libraries
    # so we use a custom FindPythonHeaders that is the same as the
    # default, but ignores when the python libraries can't be found.
    message("${Red}Manylinux build has no python shared libraries${ColorReset}")
    find_package(Python ${CSP_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter)
    find_package(PythonHeaders ${CSP_PYTHON_VERSION} EXACT REQUIRED)

    # Run with exact version so its cached for pybind
    find_package(PythonInterp ${CSP_PYTHON_VERSION} EXACT REQUIRED)
else()
    message("${Cyan}Use python shared libraries${ColorReset}")
    find_package(Python ${CSP_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter Development)

    # Run with exact version so its cached for pybind
    find_package(PythonInterp ${CSP_PYTHON_VERSION} EXACT REQUIRED)
    find_package(PythonLibs ${CSP_PYTHON_VERSION} EXACT REQUIRED)

    link_directories(${Python_LIBRARY_DIRS})
endif()


message("${Cyan}Using Python ${Python_VERSION}\nPython_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}\nPython_LIBRARIES: ${Python_LIBRARIES}\nPython_EXECUTABLE: ${Python_EXECUTABLE} ${ColorReset}")
include_directories(${Python_INCLUDE_DIRS})

# NUMPY
find_package(Numpy)
if(NOT NUMPY_FOUND)
    message(FATAL_ERROR "${Red}Numpy could not be located${ColorReset}")
else()
    message("${Cyan}Numpy found: ${NUMPY_INCLUDE_DIR}${ColorReset}")
    include_directories(${NUMPY_INCLUDE_DIR})
endif()
#####################

###################################################################################################################################################
# Asset names #
################
# prefix is _ by default
set(CMAKE_SHARED_LIBRARY_PREFIX _)

if(NOT WIN32)
    # shared suffix is .so for both linux and mac
    set(CMAKE_SHARED_LIBRARY_SUFFIX .so)

    # static suffix is _static.a # TODO decide if we want this
    set(CMAKE_STATIC_LIBRARY_SUFFIX _static.a)
else()
    # shared suffix is .pyd for windows
    set(CMAKE_SHARED_LIBRARY_SUFFIX .pyd)
endif()


###################################################################################################################################################
# Build assets #
################
include_directories("${CMAKE_SOURCE_DIR}/cpp")
# for autogen
include_directories("${CMAKE_BINARY_DIR}/cpp")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(WIN32)
    # On windows, force dlls into lib folder
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CSP_RUNTIME_INSTALL_SUBDIR lib/)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CSP_RUNTIME_INSTALL_SUBDIR bin/)
endif()

add_subdirectory(cpp/csp/adapters)
add_subdirectory(cpp/csp/core)
add_subdirectory(cpp/csp/cppnodes)
add_subdirectory(cpp/csp/engine)
add_subdirectory(cpp/csp/python)
add_subdirectory(cpp/csp/python/adapters)

###################################################################################################################################################
# Tests #
#########
if(CSP_BUILD_TESTS)
    add_subdirectory(cpp/tests)
endif()
