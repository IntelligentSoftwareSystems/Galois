add_library(galois_cusp INTERFACE)
add_dependencies(lib galois_cusp)

target_include_directories(galois_cusp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(galois_cusp INTERFACE galois_dist_async)
