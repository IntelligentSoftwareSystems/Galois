# link libgnn library and all should go well
add_executable(gcn-dist gcn-dist.cpp)
target_link_libraries(gcn-dist galois_gnn gnnbench)
