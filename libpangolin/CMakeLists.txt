set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(USE_DFS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_DFS")
endif()

set(sources
#  src/miner.cpp
  src/equivalence.cpp
  src/quick_pattern.cpp
  src/base_embedding.cpp
  src/vertex_embedding.cpp
  src/BfsMining/embedding_list.cpp
#if(USE_DFS)
  src/DfsMining/edgelist.cpp
  src/DfsMining/embedding_list_dfs.cpp
#else()
#endif()
)

add_library(pangolin STATIC ${sources})

target_link_libraries(pangolin galois_shmem gllvm)

target_include_directories(pangolin PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libpangolin/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties (pangolin PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE On
  POSITION_INDEPENDENT_CODE On
)
