include_directories("${PROJECT_SOURCE_DIR}/libgluon/include")
include_directories("${PROJECT_SOURCE_DIR}/lonestardist/include")

add_subdirectory(bc)
add_subdirectory(bfs)
add_subdirectory(cc)
add_subdirectory(kcore)
add_subdirectory(pagerank)
add_subdirectory(partition)
add_subdirectory(sgd)
add_subdirectory(sssp)

# currently tc only has a GPU implementation
if(ENABLE_HETERO_GALOIS)
  add_subdirectory(tc)
endif()
