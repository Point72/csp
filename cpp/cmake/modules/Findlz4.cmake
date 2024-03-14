find_path(Lz4_INCLUDE_DIR lz4.h)
find_library(Lz4_LIBRARY NAMES lz4 liblz4)

if (NOT TARGET lz4::lz4)
  add_library(lz4::lz4 STATIC IMPORTED)
  set_property(TARGET lz4::lz4 PROPERTY IMPORTED_LOCATION "${Lz4_LIBRARY}")
  target_include_directories(lz4::lz4 INTERFACE ${Lz4_INCLUDE_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(lz4 DEFAULT_MSG Lz4_LIBRARY Lz4_INCLUDE_DIR)
mark_as_advanced(Lz4_INCLUDE_DIR Lz4_LIBRARY)
