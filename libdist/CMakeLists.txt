add_library(galois_dist_async STATIC)
add_library(Galois::dist_async ALIAS galois_dist_async)
add_dependencies(lib galois_dist_async)
set_target_properties(galois_dist_async PROPERTIES EXPORT_NAME dist_async)

target_sources(galois_dist_async PRIVATE
        src/Barrier.cpp
        src/DistGalois.cpp
        src/DistStats.cpp
        src/Network.cpp
        src/NetworkBuffered.cpp
        src/NetworkIOMPI.cpp
        src/NetworkLCI.cpp
)

target_include_directories(galois_dist_async PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(galois_dist_async PUBLIC MPI::MPI_CXX)
target_link_libraries(galois_dist_async PUBLIC galois_shmem)
target_link_libraries(galois_dist_async PRIVATE tsuba)

target_compile_definitions(galois_dist_async PRIVATE __GALOIS_HET_ASYNC__=1)

if (USE_BARE_MPI)
  target_compile_definitions(galois_dist_async PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
endif()

if (USE_LCI)
  add_definitions(-DGALOIS_USE_LWCI)
  set(LWCI_ROOT "${PROJECT_SOURCE_DIR}/liblci")
  set(LWCI_INCLUDE "${LWCI_ROOT}/include")
  set(LWCI_LIBRARY "${LWCI_ROOT}/liblci.a")

  add_custom_command(
    OUTPUT ${LWCI_LIBRARY}
    COMMAND make -C ${LWCI_ROOT}
  )
  add_custom_target(lci DEPENDS ${LWCI_LIBRARY})

  add_dependencies(galois_dist_async lci)
  target_link_libraries(galois_dist_async PRIVATE ${LWCI_LIBRARY} -lpsm2)
  target_include_directories(galois_dist_async PUBLIC ${LWCI_INCLUDE})
endif(USE_LCI)

install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h"
)

install(TARGETS galois_dist_async
  EXPORT GaloisTargets
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT shlib
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
