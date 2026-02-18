cmake_minimum_required(VERSION 3.20.0)
project(csp-example-piglatin VERSION "0.0.1")
set(CMAKE_CXX_STANDARD 17)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Find CSP include and lib paths
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import csp; print(csp.get_include_path(), end='')"
    OUTPUT_VARIABLE CSP_INCLUDE_DIR)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import csp; print(csp.get_lib_path(), end='')"
    OUTPUT_VARIABLE CSP_LIB_DIR)

find_library(CSP_LIBRARY NAMES _cspimpl.so PATHS "${CSP_LIB_DIR}" NO_DEFAULT_PATH)

# Position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Shared library naming conventions
set(CMAKE_SHARED_LIBRARY_PREFIX _)
set(CMAKE_SHARED_LIBRARY_SUFFIX .so)

include_directories(${CSP_INCLUDE_DIR})
include_directories(${Python_INCLUDE_DIRS})

add_library(piglatin SHARED cpp/piglatin.cpp)
target_link_libraries(piglatin ${CSP_LIBRARY})

install(TARGETS piglatin LIBRARY DESTINATION piglatin)
