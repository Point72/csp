csp_autogen( csp.impl.types.autogen_types autogen_types ENGINE_AUTOGEN_HEADER ENGINE_AUTOGEN_SOURCE )

set(CSP_TYPES_PUBLIC_HEADERS
        CspEnum.h
        CspType.h
        DialectGenericType.h
        PartialSwitchCspType.h
        Struct.h
        TypeCast.h
)

set(CSP_TYPES_SOURCE_FILES
        CspEnum.cpp
        CspType.cpp
        DialectGenericType.h
        PartialSwitchCspType.h
        Struct.cpp
        TypeCast.h
)

set(ENGINE_PUBLIC_HEADERS
        ${ENGINE_AUTOGEN_HEADER}
        AdapterManager.h
        AlarmInputAdapter.h
        BasketInfo.h
        ConstInputAdapter.h
        Consumer.h
        CppNode.h
        CycleStepTable.h
        Dictionary.h
        DynamicEngine.h
        DynamicNode.h
        Engine.h
        Enums.h
        EventPropagator.h
        Feedback.h
        GraphOutputAdapter.h
        InputAdapter.h
        InputId.h
        Node.h
        OutputAdapter.h
        PendingPushEvents.h
        Profiler.h
        PushPullEvent.h
        PushEvent.h
        PullInputAdapter.h
        PushInputAdapter.h
        PushPullInputAdapter.h
        RootEngine.h
        Scheduler.h
        StatusAdapter.h
        TickBuffer.h
        TimerInputAdapter.h
        TimeSeries.h
        TimeSeriesProvider.h
        VectorContainer.h
        WindowBuffer.h
        )

set(ENGINE_SOURCE_FILES
        ${ENGINE_AUTOGEN_SOURCE}
        AdapterManager.cpp
        BasketInfo.cpp
        Consumer.cpp
        CycleStepTable.cpp
        Dictionary.cpp
        DynamicEngine.cpp
        DynamicNode.cpp
        Engine.cpp
        Enums.cpp
        EventPropagator.cpp
        GraphOutputAdapter.cpp
        InputAdapter.cpp
        Node.cpp
        OutputAdapter.cpp
        PendingPushEvents.cpp
        PushPullInputAdapter.cpp
        RootEngine.cpp
        Scheduler.cpp
        StatusAdapter.cpp
        TimeSeries.cpp
        TimeSeriesProvider.cpp
        VectorContainer.cpp
        ${ENGINE_PUBLIC_HEADERS}
        )

add_library(csp_types STATIC ${CSP_TYPES_SOURCE_FILES})
set_target_properties(csp_types PROPERTIES PUBLIC_HEADER "${CSP_TYPES_PUBLIC_HEADERS}")

add_library(csp_engine STATIC ${ENGINE_SOURCE_FILES})
set_target_properties(csp_engine PROPERTIES PUBLIC_HEADER "${ENGINE_PUBLIC_HEADERS}")
target_link_libraries(csp_engine csp_core csp_types)

install(FILES ${CSP_TYPES_PUBLIC_HEADERS} ${ENGINE_PUBLIC_HEADERS} DESTINATION include/csp/engine)

install(TARGETS csp_types csp_engine
        PUBLIC_HEADER DESTINATION include/csp/engine
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
