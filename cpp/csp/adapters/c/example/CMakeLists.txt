
set(C_EXAMPLE_HEADER_FILES
        ExamplePushInputAdapter.h
        ExampleOutputAdapter.h
)

set(C_EXAMPLE_SOURCE_FILES
        ExamplePushInputAdapter.c
        ExampleOutputAdapter.c
        ${KAFKA_HEADER_FILES}
)

add_library(csp_c_example_adapter STATIC ${C_EXAMPLE_SOURCE_FILES})
set_target_properties(csp_c_example_adapter PROPERTIES PUBLIC_HEADER "${C_EXAMPLE_HEADER_FILES}")

install(TARGETS csp_c_example_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/example
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
