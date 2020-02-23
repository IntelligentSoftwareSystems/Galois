app(gcn gcn.cpp)
target_link_libraries(gcn dg_cpu)
if(NOT USE_CPU)
  target_link_libraries(gcn dg_gpu)
endif()
target_link_libraries(gcn -lcudart -lcublas -lcurand -lcudadevrt)

