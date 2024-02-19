set(WS_CLIENT_HEADER_FILES
        WSClientAdapterManager.h
        WSClientInputAdapter.h
        WSClientOutputAdapter.h
)

set(WS_CLIENT_SOURCE_FILES
        WSClientAdapterManager.cpp
        WSClientInputAdapter.cpp
        WSClientOutputAdapter.cpp
        ${WS_CLIENT_HEADER_FILES}
)

add_library(csp_websocket_client_adapter STATIC ${WS_CLIENT_SOURCE_FILES})
set_target_properties(csp_websocket_client_adapter PROPERTIES PUBLIC_HEADER "${WS_CLIENT_HEADER_FILES}")
find_package(websocketpp REQUIRED)
target_link_libraries(csp_websocket_client_adapter csp_adapter_utils websocketpp::websocketpp)

install(TARGETS csp_websocket_client_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/websocket_client
        RUNTIME DESTINATION bin/
        LIBRARY DESTINATION lib/
       )

