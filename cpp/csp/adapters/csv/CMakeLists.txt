set(CSV_HEADER_FILES
    CsvInputAdapterManager.h
)

set(CSV_SOURCE_FILES
    CsvInputAdapterManager.cpp
    ${CSV_HEADER_FILES}
)

add_library(csp_csv_adapter STATIC ${CSV_SOURCE_FILES})

set_target_properties(csp_csv_adapter PROPERTIES
    PUBLIC_HEADER "${CSV_HEADER_FILES}"
)

target_link_libraries(csp_csv_adapter
    PRIVATE
        csp_engine
)

install(TARGETS csp_csv_adapter
    PUBLIC_HEADER DESTINATION include/csp/adapters/csv
    RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
    LIBRARY DESTINATION lib/
)