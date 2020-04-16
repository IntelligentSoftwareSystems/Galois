add_library(galois_cusp INTERFACE)
add_library(Galois::cusp ALIAS galois_cusp)
set_target_properties(galois_cusp PROPERTIES EXPORT_NAME cusp)
add_dependencies(lib galois_cusp)

target_include_directories(galois_cusp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(galois_cusp INTERFACE galois_dist_async)

install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}"
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h"
)

install(TARGETS galois_cusp
  EXPORT GaloisTargets
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT shlib
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
