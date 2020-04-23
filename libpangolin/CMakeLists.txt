add_library(pangolin STATIC)
add_library(Galois::pangolin ALIAS pangolin)
add_dependencies(lib pangolin)

target_sources(pangolin PRIVATE
  src/equivalence.cpp
  src/quick_pattern.cpp
  src/base_embedding.cpp
  src/vertex_embedding.cpp
  src/BfsMining/embedding_list.cpp
)

target_include_directories(pangolin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (ENABLE_HETERO_GALOIS)
  target_include_directories(pangolin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu_mining>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(pangolin PUBLIC galois_gpu)
endif()

target_include_directories(pangolin PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/bliss>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(pangolin PUBLIC galois_shmem)
