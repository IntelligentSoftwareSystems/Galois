add_library(galois_gpu)
add_library(Galois::gpu ALIAS galois_gpu)
set_target_properties(galois_gpu PROPERTIES EXPORT_NAME gpu)
add_dependencies(lib galois_gpu)

target_sources(galois_gpu PRIVATE
  src/csr_graph.cu
  src/ggc_rt.cu
  moderngpu/src/mgpucontext.cu
  moderngpu/src/mgpuutil.cpp
)

target_include_directories(galois_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/galois/gpu>
)
install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/galois/gpu"
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".cuh"
)

target_include_directories(galois_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/moderngpu/include>
  $<INSTALL_INTERFACE:include/galois/gpu/moderngpu>
)
install(
  DIRECTORY moderngpu/include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/galois/gpu/moderngpu"
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".cuh"
)

target_include_directories(galois_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cub>
  $<INSTALL_INTERFACE:include/galois/gpu/cub>
)
install(
  DIRECTORY cub/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/galois/gpu/cub"
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".cuh"
)

target_compile_definitions(galois_gpu PRIVATE _FORCE_INLINES)
target_include_directories(galois_gpu PRIVATE moderngpu)
set_property(TARGET galois_gpu PROPERTY CUDA_STANDARD 14)

install(TARGETS galois_gpu
  EXPORT GaloisTargets
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT shlib
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
