cmake_minimum_required(VERSION 3.7.2)

# RdKafka is required for Kafka adapter
find_package(RdKafka REQUIRED)

# Workaround for vcpkg static library transitive dependencies not propagating properly
# https://github.com/microsoft/vcpkg/issues/40320
if(CSP_USE_VCPKG)
  link_directories(${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib)
endif()

# Avro is optional - Kafka adapter works with JSON/raw bytes without it
set(CSP_KAFKA_ENABLE_AVRO FALSE)

if (CSP_USE_VCPKG)
  find_package(unofficial-avro-cpp CONFIG)

  if(unofficial-avro-cpp_FOUND)
    # Check avro-cpp version on Windows - require >= 1.12.1 for fmt compatibility
    if(WIN32 AND unofficial-avro-cpp_VERSION VERSION_LESS "1.12.1")
      message(WARNING
        "avro-cpp ${unofficial-avro-cpp_VERSION} has incompatible fmt::formatter on Windows. "
        "Kafka Avro support will be disabled. Update vcpkg to get avro-cpp >= 1.12.1.")
    else()
      set(CSP_KAFKA_ENABLE_AVRO TRUE)
      set(AVRO_LIBRARIES unofficial::avro-cpp::avrocpp CACHE INTERNAL "")
    endif()
  endif()

else()
  find_package(Avro)

  if(Avro_FOUND)
    set(CSP_KAFKA_ENABLE_AVRO TRUE)
    set(AVRO_LIBRARIES Avro::avrocpp CACHE INTERNAL "")
  endif()
endif()

# Kafka adapter is available if RdKafka is found
set(DepsKafkaAdapter_FOUND TRUE)
