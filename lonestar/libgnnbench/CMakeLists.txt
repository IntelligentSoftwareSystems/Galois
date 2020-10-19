add_library(gnnbench STATIC src/Input.cpp)
target_include_directories(gnnbench PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(gnnbench galois_gnn LLVMSupport)
