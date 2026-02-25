set(ARROW_ADAPTER_HEADER_FILES
        ArrowFieldReader.h
        ArrowFieldWriter.h
        RecordBatchToStruct.h
        StructToRecordBatch.h
)

set(ARROW_ADAPTER_SOURCE_FILES
        ArrowFieldReader.cpp
        ArrowFieldWriter.cpp
        RecordBatchToStruct.cpp
        StructToRecordBatch.cpp
        ${ARROW_ADAPTER_HEADER_FILES}
)

add_library(csp_arrow_adapter STATIC ${ARROW_ADAPTER_SOURCE_FILES})
set_target_properties(csp_arrow_adapter PROPERTIES PUBLIC_HEADER "${ARROW_ADAPTER_HEADER_FILES}")

target_link_libraries(csp_arrow_adapter PRIVATE csp_core csp_types csp_engine ${CSP_ARROW_LINK_LIBS})

install(TARGETS csp_arrow_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/arrow
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
