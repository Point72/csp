find_path(Avro_INCLUDE_DIR NAMES avro/Encoder.hh)
find_library(Avro_LIBRARY NAMES avrocpp libavrocpp)

# =============================================================================
# Workaround for conda-forge avro-cpp fmt::formatter incompatibility on Windows
# =============================================================================
# conda-forge's avro-cpp has fmt::formatter specializations with non-const
# format() methods, but fmt v12+ requires const. This causes MSVC error C2766.
#
# Solution: On Windows, we check if the avro headers have this bug, and if so,
# we create patched versions in the build directory and prepend them to the
# include path so they shadow the broken originals.
#
# This workaround can be removed once conda-forge updates avro-cpp.
# =============================================================================

set(CSP_AVRO_PATCHED_INCLUDE_DIR "")

if(WIN32 AND Avro_INCLUDE_DIR AND NOT CSP_USE_VCPKG)
    # Check if avro/Node.hh has the non-const format() bug
    # The buggy pattern is: "auto format(...) {" without "const" before the brace
    set(_avro_node_hh "${Avro_INCLUDE_DIR}/avro/Node.hh")
    set(_avro_types_hh "${Avro_INCLUDE_DIR}/avro/Types.hh")
    set(_needs_patching FALSE)

    if(EXISTS "${_avro_node_hh}")
        file(READ "${_avro_node_hh}" _node_hh_content)

        # Check if the file contains fmt::formatter and non-const format()
        # We look for "auto format" followed by ")" then whitespace then "{"
        # without "const" in between
        string(FIND "${_node_hh_content}" "fmt::formatter<avro::Name>" _has_formatter)
        if(NOT _has_formatter EQUAL -1)
            # Check specifically for the non-const pattern
            # Buggy: auto format(const avro::Name &n, FormatContext &ctx) {
            # Fixed: auto format(const avro::Name &n, FormatContext &ctx) const {
            string(REGEX MATCH "auto format\\(const avro::Name[^)]+\\)[^c]*\\{" _buggy_pattern "${_node_hh_content}")
            if(_buggy_pattern)
                set(_needs_patching TRUE)
            endif()
        endif()
    endif()

    if(_needs_patching)
        message(STATUS "Detected avro-cpp with non-const fmt::formatter bug - applying build-time patch")

        # Create patched headers directory structure
        set(CSP_AVRO_PATCHED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_patched_avro_headers")
        set(_patched_avro_dir "${CSP_AVRO_PATCHED_INCLUDE_DIR}/avro")
        file(MAKE_DIRECTORY "${_patched_avro_dir}")

        # Patch Node.hh
        # Replace: auto format(const avro::Name &n, FormatContext &ctx) {
        # With:    auto format(const avro::Name &n, FormatContext &ctx) const {
        string(REGEX REPLACE
            "(auto format\\(const avro::Name[^)]+\\))[ \t\r\n]*(\\{)"
            "\\1 const \\2"
            _patched_node_content
            "${_node_hh_content}"
        )
        file(WRITE "${_patched_avro_dir}/Node.hh" "${_patched_node_content}")
        message(STATUS "  Patched: avro/Node.hh -> ${_patched_avro_dir}/Node.hh")

        # Patch Types.hh if it exists and has the same bug
        if(EXISTS "${_avro_types_hh}")
            file(READ "${_avro_types_hh}" _types_hh_content)
            string(FIND "${_types_hh_content}" "fmt::formatter<avro::Type>" _has_type_formatter)
            if(NOT _has_type_formatter EQUAL -1)
                string(REGEX MATCH "auto format\\(avro::Type[^)]+\\)[^c]*\\{" _types_buggy "${_types_hh_content}")
                if(_types_buggy)
                    string(REGEX REPLACE
                        "(auto format\\(avro::Type[^)]+\\))[ \t\r\n]*(\\{)"
                        "\\1 const \\2"
                        _patched_types_content
                        "${_types_hh_content}"
                    )
                    file(WRITE "${_patched_avro_dir}/Types.hh" "${_patched_types_content}")
                    message(STATUS "  Patched: avro/Types.hh -> ${_patched_avro_dir}/Types.hh")
                endif()
            endif()
        endif()
    endif()
endif()

if (NOT TARGET Avro::avrocpp)
  add_library(Avro::avrocpp SHARED IMPORTED)

  # On Windows, IMPORTED_IMPLIB is the .lib file, IMPORTED_LOCATION is the .dll
  # On other platforms, IMPORTED_LOCATION is the shared library
  if(WIN32)
    set_property(TARGET Avro::avrocpp PROPERTY IMPORTED_IMPLIB "${Avro_LIBRARY}")
  else()
    set_property(TARGET Avro::avrocpp PROPERTY IMPORTED_LOCATION "${Avro_LIBRARY}")
  endif()

  # If we have patched headers, prepend them to include path so they shadow originals
  if(CSP_AVRO_PATCHED_INCLUDE_DIR)
    target_include_directories(Avro::avrocpp INTERFACE
        ${CSP_AVRO_PATCHED_INCLUDE_DIR}  # Patched headers first (shadows originals)
        ${Avro_INCLUDE_DIR}               # Original headers for everything else
    )
  else()
    target_include_directories(Avro::avrocpp INTERFACE ${Avro_INCLUDE_DIR})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Avro DEFAULT_MSG Avro_LIBRARY Avro_INCLUDE_DIR)
mark_as_advanced(Avro_INCLUDE_DIR Avro_LIBRARY Avro::avrocpp)
