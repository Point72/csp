find_path(Avro_INCLUDE_DIR NAMES avro/Encoder.hh)
find_library(Avro_LIBRARY NAMES avrocpp libavrocpp)

# =============================================================================
# Extract version from library soname and check compatibility on Windows
# =============================================================================
# avro-cpp versions <= 1.12.0 have fmt::formatter with non-const format()
# methods, but fmt v12+ requires const. This causes MSVC error C2766.
# Require avro-cpp >= 1.12.1 on Windows which has the fix.
# =============================================================================

set(Avro_COMPATIBLE TRUE)
set(Avro_VERSION "")

if(Avro_LIBRARY)
    get_filename_component(_avro_realpath "${Avro_LIBRARY}" REALPATH)
    if(_avro_realpath MATCHES "libavrocpp\\.so\\.([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        set(Avro_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(Avro_VERSION_MINOR "${CMAKE_MATCH_2}")
        set(Avro_VERSION_PATCH "${CMAKE_MATCH_3}")
        set(Avro_VERSION "${Avro_VERSION_MAJOR}.${Avro_VERSION_MINOR}.${Avro_VERSION_PATCH}")
    elseif(_avro_realpath MATCHES "avrocpp\\.dll")
        # Windows DLL doesn't have version in filename - try library name
        if(_avro_realpath MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
            set(Avro_VERSION_MAJOR "${CMAKE_MATCH_1}")
            set(Avro_VERSION_MINOR "${CMAKE_MATCH_2}")
            set(Avro_VERSION_PATCH "${CMAKE_MATCH_3}")
            set(Avro_VERSION "${Avro_VERSION_MAJOR}.${Avro_VERSION_MINOR}.${Avro_VERSION_PATCH}")
        endif()
    endif()

    if(WIN32 AND NOT CSP_USE_VCPKG)
        if(NOT Avro_VERSION)
            # Could not detect version - assume buggy and skip
            set(Avro_COMPATIBLE FALSE)
            message(WARNING
                "Could not detect avro-cpp version on Windows. "
                "Kafka adapter will be disabled to avoid potential fmt::formatter incompatibility. "
                "Use vcpkg or upgrade to avro-cpp >= 1.12.1.")
        elseif(Avro_VERSION VERSION_LESS "1.12.1")
            set(Avro_COMPATIBLE FALSE)
            message(WARNING
                "avro-cpp ${Avro_VERSION} has incompatible fmt::formatter on Windows. "
                "Kafka adapter will be disabled. Upgrade to avro-cpp >= 1.12.1.")
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
