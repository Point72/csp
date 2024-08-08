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
          OUTPUT_VARIABLE PYARROW_INCLUDE_DIR)
# Find out the lib path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import pyarrow;print(pyarrow.get_library_dirs()[0], end='')"
          OUTPUT_VARIABLE PYARROW_LIB_DIR)
# Find out the version path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import pyarrow;print(pyarrow.__version__, end='')"
  OUTPUT_VARIABLE __pyarrow_version)

if(PYARROW_INCLUDE_DIR AND PYARROW_LIB_DIR)
  set(PYARROW_FOUND 1 CACHE INTERNAL "Python pyarrow found")
  if(NOT WIN32)
    set(PYARROW_LIBRARY libarrow_python.so)
  else()
    set(PYARROW_LIBRARY ${PYARROW_LIB_DIR}/arrow_python.lib)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PyArrow REQUIRED_VARS PYARROW_INCLUDE_DIR PYARROW_LIB_DIR PYARROW_LIBRARY
                                        VERSION_VAR __pyarrow_version)