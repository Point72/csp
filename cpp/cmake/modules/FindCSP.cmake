# Find the Python CSP package
# CSP_INCLUDE_DIR
# CSP_LIBS_DIR
# CSP_FOUND
#
# CSP_AUTOGEN
# CSP_LIBRARY
# CSP_CORE_LIBRARY
# CSP_ENGINE_LIBRARY
# CSP_TYPES_LIBRARY
# CSP_TYPES_STATIC_LIBRARY
# CSP_BASELIB_LIBARY
# CSP_BASELIB_STATIC_LIBRARY
# CSP_BASKETLIB_LIBRARY
# CSP_BASKETLIB_STATIC_LIBRARY
# CSP_STATS_LIBRARY
# CSP_STATS_STATIC_LIBRARY
# CSP_NPSTATS_LIBRARY
# CSP_ADAPTER_UTILS_LIBRARY
# CSP_KAFKAADAPTER_LIBRARY
# CSP_KAFKAADAPTER_STATIC_LIBRARY
# CSP_PARQUETADAPTER_LIBRARY
# CSP_PARQUETADAPTER_STATIC_LIBRARY
#
# will be set by this script

cmake_minimum_required(VERSION 3.7.2)

find_package(Python ${CSP_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter)

set(ENV{PYTHONPATH} "${CMAKE_SOURCE_DIR}/ext:$ENV{PYTHONPATH}")

# Find out the base path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import os.path;import csp;print(os.path.dirname(csp.__file__), end='')"
          OUTPUT_VARIABLE __csp_base_path)

# Find out the include path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import csp;print(csp.get_include_path(), end='')"
          OUTPUT_VARIABLE __csp_include_path)

# Find out the lib path
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "from __future__ import print_function;import csp;print(csp.get_lib_path(), end='')"
            OUTPUT_VARIABLE __csp_lib_path)

# And the version
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import csp;print(csp.__version__, end='')"
  OUTPUT_VARIABLE __csp_version)

# Now look for files
find_file(CSP_AUTOGEN csp_autogen.py HINTS "${__csp_base_path}/build" NO_DEFAULT_PATH)
find_path(CSP_INCLUDE_DIR csp/core/System.h HINTS "${__csp_include_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)
find_path(CSP_LIBS_DIR _cspimpl.so HINTS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_LIBRARY NAMES _cspimpl.so PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)
find_library(CSP_CORE_LIBRARY NAMES libcsp_core_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)
find_library(CSP_ENGINE_LIBRARY NAMES libcsp_engine_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_TYPES_LIBRARY NAMES _csptypesimpl.so PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)
find_library(CSP_TYPES_STATIC_LIBRARY NAMES libcsp_types_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_BASELIB_LIBARY NAMES _cspbaselibimpl.so PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)
find_library(CSP_BASELIB_STATIC_LIBRARY NAMES libbaselibimpl_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_BASKETLIB_LIBRARY NAMES _cspbasketlibimpl.so PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)
find_library(CSP_BASKETLIB_STATIC_LIBRARY NAMES libbasketlibimpl_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_STATS_LIBRARY NAMES _cspstatsimpl.so PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)
find_library(CSP_STATS_STATIC_LIBRARY NAMES libstatsimpl_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_NPSTATS_LIBRARY NAMES _cspnpstatsimpl.so PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_ADAPTER_UTILS_LIBRARY NAMES libcsp_adapter_utils_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_KAFKAADAPTER_LIBRARY NAMES _kafkaadapterimpl.so PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)
find_library(CSP_KAFKAADAPTER_STATIC_LIBRARY NAMES libcsp_kafka_adapter_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

find_library(CSP_PARQUETADAPTER_LIBRARY NAMES _parquetadapterimpl.so PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)
find_library(CSP_PARQUETADAPTER_STATIC_LIBRARY NAMES libcsp_parquet_adapter_static.a PATHS "${__csp_lib_path}" NO_DEFAULT_PATH)

if(CSP_INCLUDE_DIR AND CSP_LIBS_DIR AND CSP_AUTOGEN)
  set(CSP_FOUND 1 CACHE INTERNAL "CSP found")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CSP REQUIRED_VARS CSP_INCLUDE_DIR CSP_LIBS_DIR CSP_AUTOGEN VERSION_VAR __csp_version)

function(csp_autogen MODULE_NAME DEST_FILENAME HEADER_NAME_OUTVAR SOURCE_NAME_OUTVAR)
    string( REPLACE "." "\/" MODULE_FILENAME ${MODULE_NAME} )
    string( JOIN "." MODULE_FILENAME ${MODULE_FILENAME} "py" )

    add_custom_target( mkdir_autogen_${MODULE_NAME} ALL COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen" )

    # VARARGS done by position
    if(ARGV4)
        set(CSP_AUTOGEN_EXTRA_ARGS "${ARGV4}")
    else()
        set(CSP_AUTOGEN_EXTRA_ARGS "")
    endif()

    add_custom_command(OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.h"
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${PROJECT_BINARY_DIR}/lib:${CMAKE_SOURCE_DIR}:ext:$$PYTHONPATH" "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:$$LD_LIBRARY_PATH}" ${Python_EXECUTABLE} ${CSP_AUTOGEN} -m ${MODULE_NAME} -d ${CMAKE_CURRENT_BINARY_DIR}/csp_autogen -o ${DEST_FILENAME} ${CSP_AUTOGEN_EXTRA_ARGS}
            COMMENT "generating csp c++ types from module ${MODULE_NAME} ${PROJECT_BINARY_DIR} ${CSP_AUTOGEN}"
            DEPENDS mkdir_autogen_${MODULE_NAME} ${CMAKE_SOURCE_DIR}/${MODULE_FILENAME})

    set(${SOURCE_NAME_OUTVAR} "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.cpp" PARENT_SCOPE )
    set(${HEADER_NAME_OUTVAR} "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.h" PARENT_SCOPE )
endfunction()
