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

# Windows links the import library; elsewhere the module itself is linked. CMake applies the
# leading underscore to the module but not to its import library, so accept either name.
if(WIN32)
    find_library(CSP_LIBRARY NAMES _cspimpl cspimpl PATHS "${CSP_LIB_DIR}" NO_DEFAULT_PATH)
else()
    find_library(CSP_LIBRARY NAMES _cspimpl.so PATHS "${CSP_LIB_DIR}" NO_DEFAULT_PATH)
endif()

# Position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# macOS: don't link against build python
if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
endif()

# Python extension modules are _name.so on Unix (macOS included) and _name.pyd on Windows
set(CMAKE_SHARED_LIBRARY_PREFIX _)
if(WIN32)
    set(CMAKE_SHARED_LIBRARY_SUFFIX .pyd)
else()
    set(CMAKE_SHARED_LIBRARY_SUFFIX .so)
endif()

include_directories(${CSP_INCLUDE_DIR})
include_directories(${Python_INCLUDE_DIRS})

add_library(piglatin SHARED cpp/piglatin.cpp)
# Python::Module supplies pythonXY.lib on Windows, where an extension module must link it.
# On Unix it deliberately links no libpython, leaving the symbols to the host interpreter.
target_link_libraries(piglatin ${CSP_LIBRARY} Python::Module)

install(TARGETS piglatin LIBRARY DESTINATION piglatin RUNTIME DESTINATION piglatin)
