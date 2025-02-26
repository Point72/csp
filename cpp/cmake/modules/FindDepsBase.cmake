cmake_minimum_required(VERSION 3.7.2)

# RapidJson (for adapter utils)
find_package(RapidJSON CONFIG REQUIRED)
include_directories(${RapidJSON_INCLUDE_DIRS})

# For EXPRTK node
find_path(EXPRTK_INCLUDE_DIRS "exprtk.hpp")
