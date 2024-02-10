set(REDIS_HEADER_FILES
        RedisAdapterManager.h
        RedisConsumer.h
        RedisInputAdapter.h
        RedisOutputAdapter.h
        RedisPublisher.h
        RedisSubscriber.h
)

set(REDIS_SOURCE_FILES
        RedisAdapterManager.cpp
        RedisConsumer.cpp
        RedisInputAdapter.cpp
        RedisOutputAdapter.cpp
        RedisPublisher.cpp
        RedisSubscriber.cpp
        ${REDIS_HEADER_FILES}
)

add_library(csp_redis_adapter STATIC ${REDIS_SOURCE_FILES})
set_target_properties(csp_redis_adapter PROPERTIES PUBLIC_HEADER "${REDIS_HEADER_FILES}")

find_package(redis++ CONFIG REQUIRED)
target_link_libraries(csp_redis_adapter csp_adapter_utils redis++::redis++_static)

install(TARGETS csp_redis_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/redis
        RUNTIME DESTINATION bin/
        LIBRARY DESTINATION lib/
       )
