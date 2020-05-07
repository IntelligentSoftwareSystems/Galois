add_library(galois_gluon STATIC)
add_library(Galois::gluon ALIAS galois_gluon)
set_target_properties(galois_gluon PROPERTIES EXPORT_NAME gluon)
add_dependencies(lib galois_gluon)

target_sources(galois_gluon PRIVATE
        src/cuda_device.cpp
        src/SyncStructures.cpp
        src/GlobalObj.cpp
        src/GluonSubstrate.cpp
)

target_link_libraries(galois_gluon PUBLIC galois_dist_async)

target_include_directories(galois_gluon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (GALOIS_COMM_STATS)
  target_compile_definitions(galois_gluon PRIVATE GALOIS_COMM_STATS=1)
endif()

if (GALOIS_USE_BARE_MPI)
  target_compile_definitions(galois_gluon PRIVATE GALOIS_USE_BARE_MPI=1)
endif()

install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h"
)

install(TARGETS galois_gluon
  EXPORT GaloisTargets
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT shlib
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
