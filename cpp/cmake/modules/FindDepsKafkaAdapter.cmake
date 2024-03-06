cmake_minimum_required(VERSION 3.7.2)

if (CSP_USE_VCPKG)
  find_package(RdKafka CONFIG REQUIRED)
else()
  find_package(RdKafka REQUIRED)
endif()
