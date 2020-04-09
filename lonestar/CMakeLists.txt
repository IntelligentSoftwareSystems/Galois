add_library(lonestar STATIC src/BoilerPlate.cpp)

target_include_directories(lonestar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(lonestar galois_shmem LLVMSupport)

add_subdirectory(barneshut)
add_subdirectory(betweennesscentrality) 
add_subdirectory(bfs)
add_subdirectory(bipart)
add_subdirectory(bipart_csr)
add_subdirectory(boruvka)
add_subdirectory(connectedcomponents)
add_subdirectory(delaunayrefinement)
add_subdirectory(delaunaytriangulation)
add_subdirectory(gmetis)
add_subdirectory(independentset)
add_subdirectory(kcore)
add_subdirectory(matching)
add_subdirectory(matrixcompletion)
add_subdirectory(pagerank)
add_subdirectory(pointstoanalysis)
add_subdirectory(preflowpush)
add_subdirectory(sssp)
add_subdirectory(surveypropagation)
add_subdirectory(triangles)
add_subdirectory(tutorial_examples)
add_subdirectory(ktruss)
add_subdirectory(louvainClustering)

