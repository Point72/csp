# NOTE: this only assumes that rdkafka is a shared library
find_path(RdKafka_INCLUDE_DIR NAMES librdkafka/rdkafkacpp.h)
find_library(RdKafka_LIBRARY NAMES rdkafka++ librdkafkacpp)
find_library(RdKafka_C_LIBRARY NAMES rdkafka librdkafka)

if (NOT TARGET RdKafka::rdkafka)
  add_library(RdKafka::rdkafka STATIC IMPORTED)
  set_property(TARGET RdKafka::rdkafka PROPERTY
          IMPORTED_LOCATION "${RdKafka_C_LIBRARY}")
  target_include_directories(RdKafka::rdkafka INTERFACE ${RdKafka_INCLUDE_DIR})

  add_library(RdKafka::rdkafka++ STATIC IMPORTED)
  set_property(TARGET RdKafka::rdkafka++ PROPERTY
          IMPORTED_LOCATION "${RdKafka_LIBRARY}")
  target_include_directories(RdKafka::rdkafka++ INTERFACE ${RdKafka_INCLUDE_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RdKafka DEFAULT_MSG RdKafka_LIBRARY RdKafka_C_LIBRARY RdKafka_INCLUDE_DIR)
mark_as_advanced(RdKafka_INCLUDE_DIR RdKafka_LIBRARY RdKafka_C_LIBRARY RdKafka::rdkafka++ RdKafka::rdkafka)
