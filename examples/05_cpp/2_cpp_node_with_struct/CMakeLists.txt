cmake_minimum_required(VERSION 3.20.0)
project(csp-example-struct VERSION "0.0.1")
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

set(MYSTRUCT_SRC "${CMAKE_CURRENT_BINARY_DIR}/mystruct.cpp")
set(MYSTRUCT_HDR "${CMAKE_CURRENT_BINARY_DIR}/mystruct.h")

add_custom_command(OUTPUT "${MYSTRUCT_SRC}" "${MYSTRUCT_HDR}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}" "${Python_EXECUTABLE}" -m csp.build.csp_autogen -m mystruct.struct -d "${CMAKE_CURRENT_BINARY_DIR}" -o mystruct
)
                    
add_library(mystruct SHARED cpp/struct.cpp ${MYSTRUCT_SRC})
# Python::Module supplies pythonXY.lib on Windows, where an extension module must link it.
# On Unix it deliberately links no libpython, leaving the symbols to the host interpreter.
target_link_libraries(mystruct ${CSP_LIBRARY} Python::Module)
target_include_directories(mystruct PRIVATE "${CMAKE_BINARY_DIR}")
set_target_properties(mystruct PROPERTIES PUBLIC_HEADER "${MYSTRUCT_HDR}")

install(TARGETS mystruct LIBRARY DESTINATION mystruct RUNTIME DESTINATION mystruct PUBLIC_HEADER DESTINATION mystruct/include)
