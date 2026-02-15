if(CSP_BUILD_ARROW_ADAPTER)
    add_library(arrowadapterimpl SHARED PyArrowInputAdapter.cpp ArrowInputAdapter.h)

    if(WIN32)
        if(CSP_USE_VCPKG)
            set(ARROW_PACKAGES_TO_LINK Arrow::arrow_static)
            target_compile_definitions(arrowadapterimpl PUBLIC ARROW_STATIC)
        else()
            # use dynamic variants
            # Until we manage to get the fix for ws3_32.dll in arrow-16 into conda, manually fix the error here
            get_target_property(LINK_LIBS Arrow::arrow_shared INTERFACE_LINK_LIBRARIES)
            string(REPLACE "ws2_32.dll" "ws2_32" FIXED_LINK_LIBS "${LINK_LIBS}")
            set_target_properties(Arrow::arrow_shared PROPERTIES INTERFACE_LINK_LIBRARIES "${FIXED_LINK_LIBS}")
            set(ARROW_PACKAGES_TO_LINK arrow_shared)
        endif()
    else()
        if(CSP_USE_VCPKG)
            # use static variants
            set(ARROW_PACKAGES_TO_LINK arrow_static)
        else()
            # use dynamic variants
            set(ARROW_PACKAGES_TO_LINK arrow)
        endif()
    endif()

    target_link_libraries(arrowadapterimpl csp_core csp_engine cspimpl ${ARROW_PACKAGES_TO_LINK})
    target_include_directories(arrowadapterimpl PUBLIC ${ARROW_INCLUDE_DIR})
    install(TARGETS arrowadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR} )
endif()

if(CSP_BUILD_KAFKA_ADAPTER)
    add_library(kafkaadapterimpl SHARED kafkaadapterimpl.cpp)
    target_link_libraries(kafkaadapterimpl csp_core csp_engine cspimpl csp_kafka_adapter)
    install(TARGETS kafkaadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR})
endif()

if(CSP_BUILD_PARQUET_ADAPTER)
    add_library(parquetadapterimpl SHARED parquetadapterimpl.cpp)
    target_link_libraries(parquetadapterimpl csp_core csp_engine cspimpl csp_parquet_adapter)
    target_include_directories(parquetadapterimpl PUBLIC ${ARROW_INCLUDE_DIR} ${PARQUET_INCLUDE_DIR})
    install(TARGETS parquetadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR} )
endif()

if(CSP_BUILD_WS_CLIENT_ADAPTER)
    add_library(websocketadapterimpl SHARED websocketadapterimpl.cpp)
    target_link_libraries(websocketadapterimpl csp_core csp_engine cspimpl csp_websocket_client_adapter)
    install(TARGETS websocketadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR})
endif()
