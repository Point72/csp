cmake_minimum_required(VERSION 3.7.2)

find_package(RdKafka REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(lz4 REQUIRED)

pkg_check_modules(SASL libsasl2)
if(SASL_FOUND)
  set(CSP_LINK_SASL ON)
else()
  set(CSP_LINK_SASL OFF)
endif()
