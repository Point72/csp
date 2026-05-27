cmake_minimum_required(VERSION 3.7.2)

if (CSP_USE_VCPKG)
  find_package(RdKafka CONFIG REQUIRED)
  find_package(unofficial-avro-cpp CONFIG REQUIRED)

  # Check avro-cpp version on Windows - require >= 1.12.1 for fmt v12 compatibility
  # avro-cpp 1.12.0 has fmt::formatter with non-const format() which causes MSVC C2766
  if(WIN32 AND unofficial-avro-cpp_VERSION VERSION_LESS "1.12.1")
    message(WARNING
      "avro-cpp ${unofficial-avro-cpp_VERSION} has incompatible fmt::formatter on Windows. "
      "Avro support will be disabled. Update vcpkg to get avro-cpp >= 1.12.1.")
    set(CSP_BUILD_AVRO OFF CACHE INTERNAL "")
  else()
    if(NOT WIN32)
      # Bad, but a temporary workaround for
      # https://github.com/microsoft/vcpkg/issues/40320
      link_directories(${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib)
    endif()
    set(CSP_AVRO_TARGET unofficial::avro-cpp::avrocpp CACHE INTERNAL "")
    set(CSP_BUILD_AVRO ON CACHE INTERNAL "")
  endif()
  set(DepsKafkaAdapter_FOUND TRUE)
else()
  find_package(RdKafka REQUIRED)
  find_package(Avro)
  if(Avro_FOUND)
    set(CSP_AVRO_TARGET Avro::avrocpp CACHE INTERNAL "")
    set(CSP_BUILD_AVRO ON CACHE INTERNAL "")
  else()
    set(CSP_BUILD_AVRO OFF CACHE INTERNAL "")
  endif()
  set(DepsKafkaAdapter_FOUND TRUE)
endif()
