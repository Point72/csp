# types library
set(CSPTYPESIMPL_PUBLIC_HEADERS
        CspTypeFactory.h
        PyCspEnum.h
        PyCspType.h
        PyStruct.h)
add_library(csptypesimpl
            csptypesimpl.cpp
            CspTypeFactory.cpp
            PyCspEnum.cpp
            PyCspType.cpp
            PyStruct.cpp)
set_target_properties(csptypesimpl PROPERTIES PUBLIC_HEADER "${CSPTYPESIMPL_PUBLIC_HEADERS}")
target_compile_definitions(csptypesimpl PRIVATE CSP_TYPES_EXPORTS=1)
target_link_libraries(csptypesimpl csp_core csp_types)

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
        PyConstants.h)

add_library(cspimpl SHARED
        cspimpl.cpp
        Conversions.cpp
        NumpyConversions.cpp
        PyAdapterManager.cpp
        PyAdapterManagerWrapper.cpp
        PyConstAdapter.cpp
        PyConstants.cpp
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
        ${CSPIMPL_PUBLIC_HEADERS})

set_target_properties(cspimpl PROPERTIES PUBLIC_HEADER "${CSPIMPL_PUBLIC_HEADERS}")
target_compile_definitions(cspimpl PRIVATE CSP_IMPL_EXPORTS=1)

target_link_libraries(cspimpl csptypesimpl csp_core csp_engine)
target_compile_definitions(cspimpl PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

## Baselib c++ module
add_library(cspbaselibimpl SHARED cspbaselibimpl.cpp)
target_compile_definitions(cspbaselibimpl PRIVATE CSP_EXPORTS=1)
target_link_libraries(cspbaselibimpl cspimpl baselibimpl)

# Include exprtk include directory for exprtk node
target_include_directories(cspbaselibimpl PRIVATE ${EXPRTK_INCLUDE_DIRS})

## Basketlib c++ module
add_library(cspbasketlibimpl SHARED cspbasketlibimpl.cpp)
target_compile_definitions(cspbasketlibimpl PRIVATE CSP_EXPORTS=1)
target_link_libraries(cspbasketlibimpl cspimpl basketlibimpl)

## Math c++ module
add_library(cspmathimpl SHARED cspmathimpl.cpp)
set_target_properties(cspmathimpl PROPERTIES DEFINITIONS CSP_EXPORTS=1)
target_link_libraries(cspmathimpl cspimpl mathimpl)

## Stats c++ module
add_library(cspstatsimpl SHARED cspstatsimpl.cpp)
target_compile_definitions(cspstatsimpl PRIVATE CSP_EXPORTS=1)
target_link_libraries(cspstatsimpl cspimpl statsimpl)

## NumPy stats c++ module
add_library(npstatsimpl STATIC npstatsimpl.cpp)
target_include_directories(npstatsimpl PRIVATE ${NUMPY_INCLUDE_DIRS})

add_library(cspnpstatsimpl SHARED cspnpstatsimpl.cpp)
target_compile_definitions(cspnpstatsimpl PRIVATE CSP_EXPORTS=1)
target_link_libraries(cspnpstatsimpl cspimpl npstatsimpl)
target_include_directories(cspnpstatsimpl PRIVATE ${NUMPY_INCLUDE_DIRS})

install(TARGETS csptypesimpl cspimpl cspbaselibimpl cspbasketlibimpl cspmathimpl cspstatsimpl cspnpstatsimpl
        PUBLIC_HEADER DESTINATION include/csp/python
        RUNTIME DESTINATION bin/
        LIBRARY DESTINATION lib/
       )
