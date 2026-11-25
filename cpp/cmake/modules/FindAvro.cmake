find_path(Avro_INCLUDE_DIR NAMES avro/Encoder.hh)
find_library(Avro_LIBRARY NAMES avrocpp libavrocpp)

# =============================================================================
# Check for conda-forge avro-cpp fmt::formatter incompatibility on Windows
# =============================================================================
# conda-forge's avro-cpp has fmt::formatter specializations with non-const
# format() methods, but fmt v12+ requires const. This causes MSVC error C2766.
#
# If detected, Avro_FOUND is set to FALSE and Kafka adapter will be disabled.
# =============================================================================

set(Avro_COMPATIBLE TRUE)

if(WIN32 AND Avro_INCLUDE_DIR AND NOT CSP_USE_VCPKG)
    set(_avro_node_hh "${Avro_INCLUDE_DIR}/avro/Node.hh")
    if(EXISTS "${_avro_node_hh}")
        file(READ "${_avro_node_hh}" _node_hh_content)
        string(FIND "${_node_hh_content}" "fmt::formatter<avro::Name>" _has_formatter)
        if(NOT _has_formatter EQUAL -1)
            # Check for non-const format() - the bug pattern
            string(REGEX MATCH "auto format\\([^)]+\\)[^c]*\\{" _buggy_pattern "${_node_hh_content}")
            if(_buggy_pattern)
                set(Avro_COMPATIBLE FALSE)
                message(WARNING
                    "avro-cpp has incompatible fmt::formatter (non-const format()). "
                    "Kafka adapter will be disabled. Update avro-cpp when conda-forge releases a fix.")
            endif()
        endif()
    endif()
endif()

if(Avro_COMPATIBLE AND Avro_INCLUDE_DIR AND Avro_LIBRARY)
    if(NOT TARGET Avro::avrocpp)
        add_library(Avro::avrocpp SHARED IMPORTED)
        if(WIN32)
            set_property(TARGET Avro::avrocpp PROPERTY IMPORTED_IMPLIB "${Avro_LIBRARY}")
        else()
            set_property(TARGET Avro::avrocpp PROPERTY IMPORTED_LOCATION "${Avro_LIBRARY}")
        endif()
        target_include_directories(Avro::avrocpp INTERFACE ${Avro_INCLUDE_DIR})
    endif()
endif()

include(FindPackageHandleStandardArgs)
if(Avro_COMPATIBLE)
    find_package_handle_standard_args(Avro DEFAULT_MSG Avro_LIBRARY Avro_INCLUDE_DIR)
else()
    set(Avro_FOUND FALSE)
endif()
mark_as_advanced(Avro_INCLUDE_DIR Avro_LIBRARY)
