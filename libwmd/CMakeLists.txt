add_library(galois_wmd INTERFACE)
add_library(Galois::wmd ALIAS galois_wmd)
set_target_properties(galois_wmd PROPERTIES EXPORT_NAME wmd)
add_dependencies(lib galois_wmd)

target_include_directories(galois_wmd INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(galois_wmd INTERFACE Galois::dist_async Galois::cusp)

add_subdirectory(test)

install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h"
)

install(TARGETS galois_wmd
  EXPORT GaloisTargets
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT shlib
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
