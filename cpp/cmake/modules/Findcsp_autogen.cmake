function(csp_autogen MODULE_NAME DEST_FILENAME HEADER_NAME_OUTVAR SOURCE_NAME_OUTVAR)
    string(REPLACE "." "_" MODULE_TARGETNAME ${MODULE_NAME})
    string(REPLACE "." "\/" MODULE_FILENAME ${MODULE_NAME})
    string(JOIN "." MODULE_FILENAME ${MODULE_FILENAME} "py")

    add_custom_target(mkdir_autogen_${MODULE_TARGETNAME}
        ALL COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen")

    # VARARGS done by position
    if(ARGV4)
        set(CSP_AUTOGEN_EXTRA_ARGS "${ARGV4}")
    else()
        set(CSP_AUTOGEN_EXTRA_ARGS "")
    endif()

    find_package(CSP REQUIRED)
    cmake_path(SET CSP_AUTOGEN_MODULE_PATH NORMALIZE "${CSP_AUTOGEN}")
    cmake_path(SET CSP_AUTOGEN_DESTINATION_FOLDER NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen")
    cmake_path(SET CSP_AUTOGEN_CPP_OUT NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.cpp")
    cmake_path(SET CSP_AUTOGEN_H_OUT NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/csp_autogen/${DEST_FILENAME}.h")

    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(CSP_AUTOGEN_PYTHONPATH ${PROJECT_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE};${CMAKE_SOURCE_DIR};%PYTHONPATH%)
    else()
        set(CSP_AUTOGEN_PYTHONPATH ${PROJECT_BINARY_DIR}/lib:${CMAKE_SOURCE_DIR}:$$PYTHONPATH)
    endif()

    if(CSP_ENABLE_ASAN)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang - use DYLD_INSERT_LIBRARIES
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libclang_rt.asan_osx_dynamic.dylib
                OUTPUT_VARIABLE ASAN_LIB_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            set(PRELOAD_CMD "DYLD_INSERT_LIBRARIES=${ASAN_LIB_PATH}")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            # GCC - use LD_PRELOAD
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE ASAN_LIB_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            set(PRELOAD_CMD "LD_PRELOAD=${ASAN_LIB_PATH}")
        endif()
        # Turn off leak checks as we are using PyMalloc when we run autogen 
        set(ASAN_PRELOAD_CMD "ASAN_OPTIONS=detect_leaks=0" ${PRELOAD_CMD})
    else()
        set(ASAN_PRELOAD_CMD "")
    endif()

    add_custom_command(OUTPUT "${CSP_AUTOGEN_CPP_OUT}" "${CSP_AUTOGEN_H_OUT}"
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CSP_AUTOGEN_PYTHONPATH}" ${ASAN_PRELOAD_CMD} ${Python_EXECUTABLE} ${CSP_AUTOGEN_MODULE_PATH} -m ${MODULE_NAME} -d ${CSP_AUTOGEN_DESTINATION_FOLDER} -o ${DEST_FILENAME} ${CSP_AUTOGEN_EXTRA_ARGS}
        COMMENT "generating csp c++ types from module ${MODULE_NAME}"
        DEPENDS mkdir_autogen_${MODULE_TARGETNAME} 
                    ${CSP_AUTOGEN_MODULE_PATH}
                    ${CMAKE_SOURCE_DIR}/${MODULE_FILENAME}
                    ${CSP_TYPES_LIBRARY_FOR_AUTOGEN}
    )

    set(${SOURCE_NAME_OUTVAR} "${CSP_AUTOGEN_CPP_OUT}" PARENT_SCOPE)
    set(${HEADER_NAME_OUTVAR} "${CSP_AUTOGEN_H_OUT}" PARENT_SCOPE)
endfunction()