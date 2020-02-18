set(sources
   src/tsuba_api.cpp
   src/s3.cpp
)

find_package(AWSSDK REQUIRED COMPONENTS transfer)

add_library(tsuba STATIC ${sources})
target_link_libraries(tsuba ${AWSSDK_LINK_LIBRARIES})
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

