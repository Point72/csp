# Find the Python PyArrow package
# PYARROW_INCLUDE_DIR
# PYARROW_LIB_DIR
# PYARROW_LIBRARY
# PYARROW_FOUND
# will be set by this script

cmake_minimum_required(VERSION 3.7.2)

find_package(Python ${CSP_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter)

# Find out the include path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import pyarrow;print(pyarrow.get_include(), end='')"
          OUTPUT_VARIABLE __pyarrow_include_path)
# Find out the lib path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import pyarrow;print(pyarrow.get_library_dirs()[0], end='')"
          OUTPUT_VARIABLE __pyarrow_libs_path)
# Find out the version path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import pyarrow;print(pyarrow.__version__, end='')"
  OUTPUT_VARIABLE __pyarrow_version)

find_path(PYARROW_INCLUDE_DIR arrow/python/pyarrow.h
  HINTS "${__pyarrow_include_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

find_path(PYARROW_LIB_DIR libarrow_python.so
  HINTS "${__pyarrow_libs_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

if(PYARROW_INCLUDE_DIR AND PYARROW_LIB_DIR)
  set(PYARROW_FOUND 1 CACHE INTERNAL "Python pyarrow found")
  set(PYARROW_LIBRARY libarrow_python.so)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PyArrow REQUIRED_VARS PYARROW_INCLUDE_DIR PYARROW_LIB_DIR PYARROW_LIBRARY
                                        VERSION_VAR __pyarrow_version)