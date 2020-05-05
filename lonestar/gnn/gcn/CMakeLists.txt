#app(gcn gcn.cpp)
add_executable(gcn gcn.cpp)
target_link_libraries(gcn PRIVATE Galois::shmem lonestar)

target_link_libraries(gcn PRIVATE dg_cpu)
if(ENABLE_DIST_GALOIS)
  target_link_libraries(gcn PRIVATE distgraphloader)
endif()

if(ENABLE_HETERO_GALOIS)
  target_link_libraries(gcn PRIVATE dg_gpu)
  target_link_libraries(gcn PRIVATE -lcudart -lcublas -lcurand -lcudadevrt)
endif()
