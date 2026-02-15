# Example C adapters demonstrating the C ABI interface

set(C_EXAMPLE_HEADER_FILES
        ExamplePushInputAdapter.h
        ExampleOutputAdapter.h
        ExampleManagedAdapter.h
)

set(C_EXAMPLE_SOURCE_FILES
        ExamplePushInputAdapter.c
        ExampleOutputAdapter.c
        ExampleManagedAdapter.c
)

add_library(csp_c_example_adapter STATIC ${C_EXAMPLE_SOURCE_FILES} ${C_EXAMPLE_HEADER_FILES})

# Include the engine c headers
target_include_directories(csp_c_example_adapter PUBLIC
    ${CMAKE_SOURCE_DIR}/cpp
)

# Set C standard
set_target_properties(csp_c_example_adapter PROPERTIES
    PUBLIC_HEADER "${C_EXAMPLE_HEADER_FILES}"
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# On non-Windows, we need pthread for the example threaded adapters
if(NOT WIN32)
    target_link_libraries(csp_c_example_adapter pthread)
endif()

install(TARGETS csp_c_example_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/c/example
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
)
