find_path(Avro_INCLUDE_DIR NAMES avro/Encoder.hh)
find_library(Avro_LIBRARY NAMES avrocpp libavrocpp)

if (NOT TARGET Avro::avrocpp)
  add_library(Avro::avrocpp SHARED IMPORTED)
  set_property(TARGET Avro::avrocpp PROPERTY
          IMPORTED_LOCATION "${Avro_LIBRARY}")
  target_include_directories(Avro::avrocpp INTERFACE ${Avro_INCLUDE_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Avro DEFAULT_MSG Avro_LIBRARY Avro_INCLUDE_DIR)
mark_as_advanced(Avro_INCLUDE_DIR Avro_LIBRARY Avro::avrocpp)
