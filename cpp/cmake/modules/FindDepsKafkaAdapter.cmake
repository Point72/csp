cmake_minimum_required(VERSION 3.7.2)

if (CSP_USE_VCPKG)
  find_package(RdKafka CONFIG REQUIRED)
  if(NOT WIN32) 
    # Bad, but a temporary workaround for
    # https://github.com/microsoft/vcpkg/issues/40320
    link_directories(${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib)
  endif()
else()
  find_package(RdKafka REQUIRED)
endif()
