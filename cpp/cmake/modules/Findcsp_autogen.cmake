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

    add_custom_command(OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.h"
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${PROJECT_BINARY_DIR}/lib:${CMAKE_SOURCE_DIR}:$$PYTHONPATH" "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:$$LD_LIBRARY_PATH}" ${Python_EXECUTABLE} ${CSP_AUTOGEN} -m ${MODULE_NAME} -d ${CMAKE_CURRENT_BINARY_DIR}/csp_autogen -o ${DEST_FILENAME} ${CSP_AUTOGEN_EXTRA_ARGS}
            COMMENT "generating csp c++ types from module ${MODULE_NAME} ${PROJECT_BINARY_DIR} ${CSP_AUTOGEN}"
            DEPENDS mkdir_autogen_${MODULE_NAME} ${CMAKE_SOURCE_DIR}/${MODULE_FILENAME})

    set(${SOURCE_NAME_OUTVAR} "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.cpp" PARENT_SCOPE )
    set(${HEADER_NAME_OUTVAR} "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.h" PARENT_SCOPE )
endfunction()
