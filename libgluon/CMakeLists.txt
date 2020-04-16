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

target_link_libraries(galois_gluon PUBLIC galois_dist_async LLVMSupport)

target_include_directories(galois_gluon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (REPORT_COMM_STATS)
  target_compile_definitions(galois_gluon PRIVATE MORE_COMM_STATS=1)
endif()

if (USE_BARE_MPI)
  target_compile_definitions(galois_gluon PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
endif()

if (USE_LCI)
  add_dependencies(galois_gluon lci)
  target_link_libraries(galois_gluon ${LWCI_LIBRARY} -lpsm2)
  target_include_directories(galois_gluon PUBLIC ${LWCI_INCLUDE})
endif()

install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}"
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
