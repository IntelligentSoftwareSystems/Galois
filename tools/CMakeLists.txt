add_subdirectory(graph-convert)
add_subdirectory(graph-remap)
add_subdirectory(graph-stats)

if (ENABLE_DIST_GALOIS)
  add_subdirectory(dist-graph-convert)
endif()
