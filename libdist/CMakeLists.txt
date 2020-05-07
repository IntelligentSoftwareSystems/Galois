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

target_compile_definitions(galois_dist_async PRIVATE __GALOIS_HET_ASYNC__=1)

if (GALOIS_USE_BARE_MPI)
  target_compile_definitions(galois_dist_async PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
endif()

if (GALOIS_USE_LCI)
  add_definitions(-DGALOIS_USE_LCI)
  set(LCI_ROOT "${CMAKE_BINARY_DIR}/libdist/external/src/lci")
  set(LCI_INCLUDE "${LCI_ROOT}/include")
  set(LCI_LIBRARY "${LCI_ROOT}/liblci.a")

  include(ExternalProject)
  # do not clone submodules for external projects
  cmake_policy(SET CMP0097 NEW)

  ExternalProject_Add(lci
    PREFIX external
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
    LOG_OUTPUT_ON_FAILURE 1
    GIT_REPOSITORY "https://github.com/uiuc-hpc/LC.git"
    GIT_SUBMODULES ""
    GIT_TAG "9bf912829339879e1132614c6d24cd032c32366b")

  add_dependencies(galois_dist_async lci)
  target_link_libraries(galois_dist_async PRIVATE ${LCI_LIBRARY} -lpsm2)
  target_include_directories(galois_dist_async PUBLIC 
    $<BUILD_INTERFACE:${LCI_INCLUDE}>
    $<INSTALL_INTERFACE:include>
  )
endif(GALOIS_USE_LCI)

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
