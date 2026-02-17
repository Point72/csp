cmake_minimum_required(VERSION 3.20.0)
project(csp-example-counter-adapter VERSION "0.0.1")
set(CMAKE_CXX_STANDARD 17)

include(CheckCCompilerFlag)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOS ON)
    set(LINUX OFF)
else()
    set(MACOS OFF)
    set(LINUX ON)
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cpp/cmake/modules")
set(ENV{PYTHONPATH} "${CMAKE_SOURCE_DIR}/../../../:$ENV{PYTHONPATH}")

set(CMAKE_MACOSX_RPATH TRUE)
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_NAME_DIR "@rpath")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED PYTHON_VERSION)
    set(PYTHON_VERSION 3.11)
endif()

find_package(Color)

if(MACOS)
    # fix for threads on osx
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)

    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version")

    # don't link against build python
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-ld_classic")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-ld_classic")

    # Support cross build
    check_c_compiler_flag("-arch x86_64" x86_64Supported)
    check_c_compiler_flag("-arch arm64" arm64Supported)

    if(x86_64Supported AND arm64Supported)
        set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build universal architecture for OSX" FORCE)
    elseif(x86_64Supported)
        set(CMAKE_REQUIRED_LINK_OPTIONS "-arch;x86_64")
        set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build universal architecture for OSX" FORCE)
    elseif(arm64Supported)
        set(CMAKE_REQUIRED_LINK_OPTIONS "-arch;arm64")
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build universal architecture for OSX" FORCE)
    endif()

    set(CMAKE_INSTALL_RPATH "@loader_path/../csp/lib")

    message("${Cyan}Use python shared libraries${ColorReset}")
    find_package(Python ${PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter Development)
    find_package(PythonInterp ${PYTHON_VERSION} EXACT REQUIRED)
    find_package(PythonLibs ${PYTHON_VERSION} EXACT REQUIRED)

    link_directories(${Python_LIBRARY_DIRS})
elseif(LINUX)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN/../csp/lib")

    message("${Red}Manylinux build has no python shared libraries${ColorReset}")
    find_package(Python ${PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter)
    find_package(PythonHeaders ${PYTHON_VERSION} EXACT REQUIRED)
    find_package(PythonInterp ${PYTHON_VERSION} EXACT REQUIRED)
endif()

message("${Cyan}Using Python ${Python_VERSION}\nPython_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}\nPython_LIBRARIES: ${Python_LIBRARIES}\nPython_EXECUTABLE: ${Python_EXECUTABLE} ${ColorReset}")
include_directories(${Python_INCLUDE_DIRS})

# prefix is _ by default
set(CMAKE_SHARED_LIBRARY_PREFIX _)

# shared suffix is .so for both linux and mac
set(CMAKE_SHARED_LIBRARY_SUFFIX .so)

find_package(CSP REQUIRED)
message("${Cyan}Found CSP:\n\tincludes in: ${CSP_INCLUDE_DIR}\n\tlibraries in: ${CSP_LIBS_DIR}${ColorReset}")
include_directories(${CSP_INCLUDE_DIR})



# Example C adapters demonstrating the C ABI interface

set(C_EXAMPLE_HEADER_FILES
        cpp/ExamplePushInputAdapter.h
        cpp/ExampleOutputAdapter.h
        cpp/ExampleManagedAdapter.h
)

set(C_EXAMPLE_SOURCE_FILES
        cpp/ExamplePushInputAdapter.c
        cpp/ExampleOutputAdapter.c
        cpp/ExampleManagedAdapter.c
)

add_library(csp_c_example_adapter STATIC ${C_EXAMPLE_SOURCE_FILES} ${C_EXAMPLE_HEADER_FILES})

# Include the engine c headers
target_include_directories(csp_c_example_adapter PUBLIC
    ${CMAKE_SOURCE_DIR}/cpp
)

# Set C standard
set_target_properties(csp_c_example_adapter PROPERTIES
    PUBLIC_HEADER "${C_EXAMPLE_HEADER_FILES}"
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# On non-Windows, we need pthread for the example threaded adapters
if(NOT WIN32)
    target_link_libraries(csp_c_example_adapter pthread)
endif()

install(TARGETS csp_c_example_adapter
        LIBRARY DESTINATION exampleadapter/lib
)

# Python bindings for example C adapters

# This creates a Python extension module
Python_add_library(_exampleadapterimpl MODULE cpp/exampleadapterimpl.c)

# Link with the example adapter library
target_link_libraries(_exampleadapterimpl PRIVATE
    csp_c_example_adapter
)

# Include directories
target_include_directories(_exampleadapterimpl PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp
    ${Python_INCLUDE_DIRS}
)

# Set output name without lib prefix
set_target_properties(_exampleadapterimpl PROPERTIES
    PREFIX ""
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Install to the Python package location
install(TARGETS _exampleadapterimpl
    LIBRARY DESTINATION exampleadapter
)
