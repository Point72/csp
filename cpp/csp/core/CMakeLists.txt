set(CORE_PUBLIC_HEADERS
      BasicAllocator.h
      ${CMAKE_CURRENT_BINARY_DIR}/cpp/csp/core/Config.h
      DynamicBitSet.h
      Enum.h
      EnumBitSet.h
      Exception.h
      FileUtils.h
      Generator.h
      Hash.h
      Likely.h
      QueueWaiter.h
      Platform.h
      SRMWLockFreeQueue.h
      System.h
      TaggedPointerUnion.h
      Time.h
      )

set(CORE_SOURCE_FILES
        Exception.cpp
        Time.cpp
        )

configure_file(Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/cpp/csp/core/Config.h)

add_library(csp_core SHARED ${CORE_SOURCE_FILES})
set_target_properties(csp_core PROPERTIES PUBLIC_HEADER "${CORE_PUBLIC_HEADERS}")
target_compile_definitions(csp_core PRIVATE CSP_CORE_EXPORTS=1)

install(TARGETS csp_core
        PUBLIC_HEADER DESTINATION include/csp/core
        RUNTIME DESTINATION lib/
        LIBRARY DESTINATION lib/
       )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpp/csp/core/Config.h DESTINATION include/csp/core)
