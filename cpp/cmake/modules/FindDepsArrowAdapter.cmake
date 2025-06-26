cmake_minimum_required(VERSION 3.7.2)

# ARROW
find_package(Arrow REQUIRED)
include_directories(${ARROW_INCLUDE_DIR})
