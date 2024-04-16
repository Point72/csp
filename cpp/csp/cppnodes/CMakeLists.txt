add_library(baselibimpl STATIC baselibimpl.cpp)
target_link_libraries(baselibimpl csp_core csp_engine)

add_library(basketlibimpl STATIC basketlibimpl.cpp)
target_link_libraries(basketlibimpl baselibimpl csp_core csp_engine)

add_library(mathimpl STATIC mathimpl.cpp)
target_link_libraries(mathimpl csp_core csp_engine)

add_library(statsimpl STATIC statsimpl.cpp)
set_target_properties(statsimpl PROPERTIES PUBLIC_HEADER statsimpl.h)
target_link_libraries(statsimpl baselibimpl csp_core csp_engine)

install(TARGETS baselibimpl basketlibimpl mathimpl statsimpl
        PUBLIC_HEADER DESTINATION include/csp/cppnodes
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
