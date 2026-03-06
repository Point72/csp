cmake_minimum_required(VERSION 3.7.2)

# ARROW
find_package(Arrow REQUIRED)
include_directories(${ARROW_INCLUDE_DIR})

# Resolve Arrow link targets based on platform and vcpkg configuration.
# Sets CSP_ARROW_LINK_LIBS for use in target_link_libraries().
# On Windows with vcpkg, also applies the ws2_32.dll fix and defines ARROW_STATIC.
if(WIN32)
    if(CSP_USE_VCPKG)
        set(CSP_ARROW_LINK_LIBS Arrow::arrow_static)
        add_compile_definitions(ARROW_STATIC)
    else()
        # Until we manage to get the fix for ws2_32.dll in arrow-16 into conda, manually fix the error here
        get_target_property(LINK_LIBS Arrow::arrow_shared INTERFACE_LINK_LIBRARIES)
        string(REPLACE "ws2_32.dll" "ws2_32" FIXED_LINK_LIBS "${LINK_LIBS}")
        set_target_properties(Arrow::arrow_shared PROPERTIES INTERFACE_LINK_LIBRARIES "${FIXED_LINK_LIBS}")
        set(CSP_ARROW_LINK_LIBS arrow_shared)
    endif()
else()
    if(CSP_USE_VCPKG)
        set(CSP_ARROW_LINK_LIBS arrow_static)
    else()
        set(CSP_ARROW_LINK_LIBS arrow)
    endif()
endif()
