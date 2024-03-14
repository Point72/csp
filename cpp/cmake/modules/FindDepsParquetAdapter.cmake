cmake_minimum_required(VERSION 3.7.2)

# ARROW
find_package(Arrow REQUIRED)
include_directories(${ARROW_INCLUDE_DIR})

# PARQUET
find_package(Parquet REQUIRED)
include_directories(${PARQUET_INCLUDE_DIR})

# Other deps
find_package(Thrift REQUIRED)
find_package(Brotli REQUIRED)
find_package(Snappy REQUIRED)
# find_package(unofficial-brotli CONFIG REQUIRED)
find_package(utf8proc REQUIRED)
# find_package(unofficial-utf8proc CONFIG REQUIRED)
find_package(lz4 REQUIRED)
# find_package(PyArrow REQUIRED)
