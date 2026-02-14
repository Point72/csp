add_subdirectory(core)
add_subdirectory(engine)

# Ignore bogus gcc warning which is coming from gtest headers
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105651
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(SUBDIRS core engine)
    foreach(SUBDIR ${SUBDIRS})
        get_directory_property(SUBDIR_TARGETS DIRECTORY ${SUBDIR} BUILDSYSTEM_TARGETS)
        foreach(TARGET ${SUBDIR_TARGETS})
            set_target_properties(${TARGET} PROPERTIES COMPILE_OPTIONS "-Wno-restrict")
        endforeach()
    endforeach()
endif()