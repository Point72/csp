cmake_minimum_required(VERSION 3.20.0)
project(csp-example-c-api-adapter VERSION "0.0.1")
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

# Example C adapters demonstrating the C ABI interface
add_library(csp_c_example_adapter STATIC
    cpp/ExamplePushInputAdapter.c
    cpp/ExampleOutputAdapter.c
    cpp/ExampleManagedAdapter.c
)
target_include_directories(csp_c_example_adapter PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
set_target_properties(csp_c_example_adapter PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

if(NOT WIN32)
    target_link_libraries(csp_c_example_adapter pthread)
endif()

install(TARGETS csp_c_example_adapter ARCHIVE DESTINATION exampleadapter/lib)

# Python bindings for example C adapters
Python_add_library(_exampleadapterimpl MODULE cpp/exampleadapterimpl.c)
target_link_libraries(_exampleadapterimpl PRIVATE csp_c_example_adapter ${CSP_LIBRARY})
target_include_directories(_exampleadapterimpl PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${Python_INCLUDE_DIRS})
set_target_properties(_exampleadapterimpl PROPERTIES PREFIX "" C_STANDARD 11 C_STANDARD_REQUIRED ON)

install(TARGETS _exampleadapterimpl LIBRARY DESTINATION exampleadapter RUNTIME DESTINATION exampleadapter)
