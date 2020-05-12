add_subdirectory(graph-convert)
add_subdirectory(graph-remap)
add_subdirectory(graph-stats)

if (GALOIS_ENABLE_DIST)
  add_subdirectory(dist-graph-convert)
endif()
