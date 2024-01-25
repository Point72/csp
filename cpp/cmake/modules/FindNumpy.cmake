# Find the Python NumPy package
# NUMPY_INCLUDE_DIR
# NUMPY_FOUND
# will be set by this script

cmake_minimum_required(VERSION 3.7.2)

find_package(Python ${CSP_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter)

# Find out the include path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import numpy;print(numpy.get_include(), end='')"
          OUTPUT_VARIABLE __numpy_path)
# And the version
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from __future__ import print_function;import numpy;print(numpy.__version__, end='')"
  OUTPUT_VARIABLE __numpy_version)

find_path(NUMPY_INCLUDE_DIR numpy/arrayobject.h
  HINTS "${__numpy_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

if(NUMPY_INCLUDE_DIR)
  set(NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
endif(NUMPY_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Numpy REQUIRED_VARS NUMPY_INCLUDE_DIR
                                        VERSION_VAR __numpy_version)