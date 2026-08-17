find_package(GTest CONFIG REQUIRED)

add_executable(test_dictionary test_dictionary.cpp )
target_link_libraries(test_dictionary csp_engine GTest::gtest GTest::gtest_main)
# For DialectGeneric impl registration ( needed on windows to get it exported )
target_compile_definitions(test_dictionary PRIVATE CSPTYPESIMPL_EXPORTS=1)

add_executable(test_tick_buffer test_tick_buffer.cpp)
target_link_libraries(test_tick_buffer csp_engine GTest::gtest GTest::gtest_main)

add_executable(test_time_series test_time_series.cpp)

target_link_libraries(test_time_series csp_engine GTest::gtest GTest::gtest_main )

add_executable(test_partial_switch_csp_type test_partial_switch_csp_type.cpp)
target_link_libraries(test_partial_switch_csp_type csp_engine GTest::gtest GTest::gtest_main)

add_executable(test_window_buffer test_window_buffer.cpp)
target_link_libraries(test_window_buffer csp_engine GTest::gtest GTest::gtest_main)

# The value/string/time layer of the C API has no engine dependency, so it is compiled directly
# into the test rather than linked from the extension module that normally hosts it.
add_executable(test_c_api
        test_c_api.cpp
        ../../csp/engine/ErrorExtern.cpp
        ../../csp/engine/StringExtern.cpp
        ../../csp/engine/TimeExtern.cpp
        ../../csp/engine/ValueExtern.cpp)
# These sources define the C API symbols, so the export macro must resolve to dllexport rather
# than dllimport; without this MSVC emits LNK4217/LNK4286 for every locally defined symbol.
target_compile_definitions(test_c_api PRIVATE CSPIMPL_EXPORTS)
target_link_libraries(test_c_api GTest::gtest GTest::gtest_main)

install(TARGETS test_tick_buffer test_time_series test_partial_switch_csp_type test_dictionary test_window_buffer test_c_api
        RUNTIME DESTINATION tests/bin/
        LIBRARY DESTINATION tests/lib/
        ARCHIVE DESTINATION tests/lib/)
