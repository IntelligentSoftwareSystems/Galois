app(gcn gcn.cpp)
target_link_libraries(gcn dg_cpu dg_gpu)
target_link_libraries(gcn -lcudart -lcublas -lcurand -lcudadevrt)

