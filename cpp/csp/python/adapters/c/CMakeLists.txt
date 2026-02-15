# Python bindings for example C adapters

# This creates a Python extension module
Python_add_library(_exampleadapterimpl MODULE exampleadapterimpl.c)

# Link with the example adapter library
target_link_libraries(_exampleadapterimpl PRIVATE
    csp_c_example_adapter
)

# Include directories
target_include_directories(_exampleadapterimpl PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp
    ${Python_INCLUDE_DIRS}
)

# Set output name without lib prefix
set_target_properties(_exampleadapterimpl PROPERTIES
    PREFIX ""
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Install to the Python package location
install(TARGETS _exampleadapterimpl
    LIBRARY DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
    RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
)
