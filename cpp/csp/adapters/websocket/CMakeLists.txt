csp_autogen( csp.adapters.websocket_types websocket_types WEBSOCKET_HEADER WEBSOCKET_SOURCE )

set(WS_CLIENT_HEADER_FILES
        ClientAdapterManager.h
        ClientInputAdapter.h
        ClientOutputAdapter.h
        ClientHeaderUpdateAdapter.h
        WebsocketEndpoint.h
        ${WEBSOCKET_HEADER} 
)

set(WS_CLIENT_SOURCE_FILES
        ClientAdapterManager.cpp
        ClientInputAdapter.cpp
        ClientOutputAdapter.cpp
        ClientHeaderUpdateAdapter.cpp
        WebsocketEndpoint.cpp
        ${WS_CLIENT_HEADER_FILES}
        ${WEBSOCKET_SOURCE}
)

add_library(csp_websocket_client_adapter STATIC ${WS_CLIENT_SOURCE_FILES})
set_target_properties(csp_websocket_client_adapter PROPERTIES PUBLIC_HEADER "${WS_CLIENT_HEADER_FILES}")

find_package(Boost REQUIRED)
find_package(OpenSSL REQUIRED)

target_link_libraries(csp_websocket_client_adapter
	PRIVATE
	csp_adapter_utils
        Boost::boost
        OpenSSL::SSL
)

install(TARGETS csp_websocket_client_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/websocket
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
