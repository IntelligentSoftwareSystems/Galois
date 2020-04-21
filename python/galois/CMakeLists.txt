include_directories(cpp)

add_cython_target(shmem shmem.pyx CXX OUTPUT_VAR GALOIS_SOURCES)
add_library(shmem MODULE ${GALOIS_SOURCES})
python_extension_module(shmem)
target_link_libraries(shmem Galois::shmem)

# Avoid collisions with existing application targets with the same name.
add_cython_target(_bfs _bfs.pyx shmem.pyx CXX OUTPUT_VAR BFS_SOURCES)
add_library(_bfs MODULE ${BFS_SOURCES})
python_extension_module(_bfs)
target_link_libraries(_bfs Galois::shmem)

add_cython_target(_pagerank _pagerank.pyx shmem.pyx CXX OUTPUT_VAR PAGERANK_SOURCES)
add_library(_pagerank MODULE ${PAGERANK_SOURCES})
python_extension_module(_pagerank)
target_link_libraries(_pagerank Galois::shmem)

install(
  TARGETS shmem _bfs _pagerank
  LIBRARY DESTINATION python/galois
)

install(
  TARGETS galois_shmem 
  PUBLIC_HEADER DESTINATION include
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
)

# In order to preserve the directory structure, galois_shmem uses
# install(DIRECTORY) rather than setting the PUBLIC_HEADER property on the
# galois_shmem itself. Mirror the install(DIRECTORY) logic here.
get_target_property(GALOIS_SOURCE_DIR galois_shmem SOURCE_DIR)
get_target_property(GALOIS_BINARY_DIR galois_shmem BINARY_DIR)
install(
  DIRECTORY "${GALOIS_SOURCE_DIR}/include/" "${GALOIS_BINARY_DIR}/include/"
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)
