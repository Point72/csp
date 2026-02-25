cmake_minimum_required(VERSION 3.7.2)

# ARROW (reuse FindDepsArrowAdapter for find_package + link target resolution)
find_package(DepsArrowAdapter REQUIRED)

# PARQUET
find_package(Parquet REQUIRED)
include_directories(${PARQUET_INCLUDE_DIR})

# Resolve Parquet link targets based on platform and vcpkg configuration.
# Sets CSP_PARQUET_LINK_LIBS for use in target_link_libraries().
if(WIN32)
    if(CSP_USE_VCPKG)
        set(CSP_PARQUET_LINK_LIBS Parquet::parquet_static)
        add_compile_definitions(PARQUET_STATIC)
    else()
        set(CSP_PARQUET_LINK_LIBS parquet_shared)
    endif()
else()
    if(CSP_USE_VCPKG)
        set(CSP_PARQUET_LINK_LIBS parquet_static)
    else()
        set(CSP_PARQUET_LINK_LIBS parquet)
    endif()
endif()

# Other deps
find_package(Thrift REQUIRED)
find_package(Brotli REQUIRED)
find_package(Snappy REQUIRED)
# find_package(unofficial-brotli CONFIG REQUIRED)
find_package(utf8proc REQUIRED)
# find_package(unofficial-utf8proc CONFIG REQUIRED)
find_package(lz4 REQUIRED)
# find_package(PyArrow REQUIRED)
