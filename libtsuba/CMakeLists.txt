set(sources
   src/tsuba_api.cpp
   src/s3.cpp
)

add_library(tsuba STATIC ${sources})

# The conan generated package name is aws-sdk-cpp while the
# package name when built from source is AWSSDK.
find_package(aws-sdk-cpp QUIET)
if(aws-sdk-cpp_FOUND)
  target_link_libraries(tsuba aws-sdk-cpp::aws-sdk-cpp)
else()
  find_package(AWSSDK REQUIRED COMPONENTS transfer)
  target_link_libraries(tsuba ${AWSSDK_LINK_LIBRARIES})
endif()

target_include_directories(tsuba PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS tsuba
  EXPORT GaloisTargets
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  INCLUDES DESTINATION "${RELATIVE_INCLUDE_FROM_INSTALL_PREFIX}"
)
