# types library
set(CSPTYPESIMPL_PUBLIC_HEADERS
        CspTypeFactory.h
        PyCspEnum.h
        PyCspType.h
        PyStruct.h
        PyStructList.h
        PyStructList_impl.h
        PyStructFastList.h
        PyStructFastList_impl.h
        VectorWrapper.h )

add_library(csptypesimpl
            csptypesimpl.cpp
            CspTypeFactory.cpp
            PyCspEnum.cpp
            PyCspType.cpp
            PyStruct.cpp
            PyStructToJson.cpp
            PyStructToDict.cpp)

set_target_properties(csptypesimpl PROPERTIES PUBLIC_HEADER "${CSPTYPESIMPL_PUBLIC_HEADERS}")
target_compile_definitions(csptypesimpl PUBLIC RAPIDJSON_HAS_STDSTRING=1)
target_link_libraries(csptypesimpl csp_core csp_types)
target_compile_definitions(csptypesimpl PRIVATE CSPTYPESIMPL_EXPORTS=1)

set(CSPIMPL_PUBLIC_HEADERS
        Common.h
        Conversions.h
        Exception.h
        InitHelper.h
        NumpyConversions.h
        NumpyInputAdapter.h
        PyAdapterManagerWrapper.h
        PyBasketInputProxy.h
        PyBasketOutputProxy.h
        PyCppNode.h
        PyEngine.h
        PyInputAdapterWrapper.h
        PyInputProxy.h
        PyIterator.h
        PyCppNode.h
        PyNode.h
        PyNodeWrapper.h
        PyNumbaNode.h
        PyObjectPtr.h
        PyOutputAdapterWrapper.h
        PyOutputProxy.h
        PyConstants.h
        PyStructToJson.h
        PyStructToDict.h)

add_library(cspimpl SHARED
        cspimpl.cpp
        Conversions.cpp
        NumpyConversions.cpp
        PyAdapterManager.cpp
        PyAdapterManagerWrapper.cpp
        PyConstAdapter.cpp
        PyCppNode.cpp
        PyEngine.cpp
        PyGraphOutputAdapter.cpp
        PyInputAdapterWrapper.cpp
        PyBasketInputProxy.cpp
        PyBasketOutputProxy.cpp
        PyDynamicNode.cpp
        PyFeedbackAdapter.cpp
        PyInputProxy.cpp
        PyNode.cpp
        PyNodeWrapper.cpp
        PyNumbaNode.cpp
        PyNumpyAdapter.cpp
        PyOutputAdapter.cpp
        PyOutputAdapterWrapper.cpp
        PyOutputProxy.cpp
        PyPullInputAdapter.cpp
        PyPushInputAdapter.cpp
        PyPushPullInputAdapter.cpp
        PyManagedSimInputAdapter.cpp
        PyTimerAdapter.cpp
        PyConstants.cpp
        ${CSPIMPL_PUBLIC_HEADERS})

set_target_properties(cspimpl PROPERTIES PUBLIC_HEADER "${CSPIMPL_PUBLIC_HEADERS}")

target_link_libraries(cspimpl csptypesimpl csp_core csp_engine )

target_compile_definitions(cspimpl PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
target_compile_definitions(cspimpl PRIVATE CSPIMPL_EXPORTS=1)

## Baselib c++ module
add_library(cspbaselibimpl SHARED cspbaselibimpl.cpp)
target_link_libraries(cspbaselibimpl cspimpl baselibimpl)

# Include exprtk include directory for exprtk node
target_include_directories(cspbaselibimpl PRIVATE ${EXPRTK_INCLUDE_DIRS})

## Basketlib c++ module
add_library(cspbasketlibimpl SHARED cspbasketlibimpl.cpp)
target_link_libraries(cspbasketlibimpl cspimpl basketlibimpl)

## Math c++ module
add_library(cspmathimpl SHARED cspmathimpl.cpp)
target_link_libraries(cspmathimpl cspimpl mathimpl)

## Stats c++ module
add_library(cspstatsimpl SHARED cspstatsimpl.cpp)
target_link_libraries(cspstatsimpl cspimpl statsimpl)

## Testlib c++ module
add_library(csptestlibimpl SHARED csptestlibimpl.cpp)
target_link_libraries(csptestlibimpl cspimpl)

## NumPy stats c++ module
add_library(npstatsimpl STATIC npstatsimpl.cpp)
add_library(cspnpstatsimpl SHARED cspnpstatsimpl.cpp)
target_link_libraries(cspnpstatsimpl cspimpl npstatsimpl)
target_include_directories(npstatsimpl PRIVATE ${NUMPY_INCLUDE_DIRS})
target_include_directories(cspnpstatsimpl PRIVATE ${NUMPY_INCLUDE_DIRS})

install(TARGETS csptypesimpl cspimpl cspbaselibimpl cspbasketlibimpl cspmathimpl cspstatsimpl csptestlibimpl cspnpstatsimpl
        PUBLIC_HEADER DESTINATION include/csp/python
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
