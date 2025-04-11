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


find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

if(WIN32)
   if(CSP_USE_VCPKG)
     set(ARROW_PACKAGES_TO_LINK Arrow::arrow_static Parquet::parquet_static )
     target_compile_definitions(csp_parquet_adapter PUBLIC ARROW_STATIC)
     target_compile_definitions(csp_parquet_adapter PUBLIC PARQUET_STATIC)
   else()
     # use dynamic variants
     # Until we manage to get the fix for ws3_32.dll in arrow-16 into conda, manually fix the error here
     get_target_property(LINK_LIBS Arrow::arrow_shared INTERFACE_LINK_LIBRARIES)
     string(REPLACE "ws2_32.dll" "ws2_32" FIXED_LINK_LIBS "${LINK_LIBS}")
     set_target_properties(Arrow::arrow_shared PROPERTIES INTERFACE_LINK_LIBRARIES "${FIXED_LINK_LIBS}")
     set(ARROW_PACKAGES_TO_LINK parquet_shared arrow_shared)
   endif()
else()
  if(CSP_USE_VCPKG)
    # use static variants
    set(ARROW_PACKAGES_TO_LINK parquet_static arrow_static)
  else()
    # use dynamic variants
    set(ARROW_PACKAGES_TO_LINK parquet arrow)
  endif()
endif()

## Baselib c++ module
add_library(cspbaselibimpl SHARED cspbaselibimpl.cpp)
target_link_libraries(cspbaselibimpl cspimpl baselibimpl csp_parquet_adapter ${ARROW_PACKAGES_TO_LINK})

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
