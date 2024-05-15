function(csp_autogen MODULE_NAME DEST_FILENAME HEADER_NAME_OUTVAR SOURCE_NAME_OUTVAR)
    string( REPLACE "." "\/" MODULE_FILENAME ${MODULE_NAME} )
    string( JOIN "." MODULE_FILENAME ${MODULE_FILENAME} "py" )

    add_custom_target( mkdir_autogen_${MODULE_NAME}
        ALL COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen" )

    # VARARGS done by position
    if(ARGV4)
        set(CSP_AUTOGEN_EXTRA_ARGS "${ARGV4}")
    else()
        set(CSP_AUTOGEN_EXTRA_ARGS "")
    endif()

    cmake_path(SET CSP_AUTOGEN_MODULE_PATH NORMALIZE "${CMAKE_SOURCE_DIR}/csp/build/csp_autogen.py")
    cmake_path(SET CSP_AUTOGEN_DESTINATION_FOLDER NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen")
    cmake_path(SET CSP_AUTOTGEN_CPP_OUT NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.cpp")
    cmake_path(SET CSP_AUTOTGEN_H_OUT NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.h")

    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(CSP_AUTOGEN_PYTHONPATH ${PROJECT_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE};${CMAKE_SOURCE_DIR};%PYTHONPATH% )
    else()
        set(CSP_AUTOGEN_PYTHONPATH ${PROJECT_BINARY_DIR}/lib:${CMAKE_SOURCE_DIR}:$$PYTHONPATH )
    endif()

    add_custom_command(OUTPUT "${CSP_AUTOTGEN_CPP_OUT}" "${CSP_AUTOTGEN_H_OUT}"
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CSP_AUTOGEN_PYTHONPATH}" ${Python_EXECUTABLE} ${CSP_AUTOGEN_MODULE_PATH} -m ${MODULE_NAME} -d ${CSP_AUTOGEN_DESTINATION_FOLDER} -o ${DEST_FILENAME} ${CSP_AUTOGEN_EXTRA_ARGS}
        COMMENT "generating csp c++ types from module ${MODULE_NAME}"
        DEPENDS mkdir_autogen_${MODULE_NAME} 
                    ${CMAKE_SOURCE_DIR}/csp/build/csp_autogen.py
                    ${CMAKE_SOURCE_DIR}/${MODULE_FILENAME}
                    csptypesimpl
    )

    set(${SOURCE_NAME_OUTVAR} "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.cpp" PARENT_SCOPE )
    set(${HEADER_NAME_OUTVAR} "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.h" PARENT_SCOPE )
endfunction()