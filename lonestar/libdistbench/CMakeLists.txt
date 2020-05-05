include_directories("${PROJECT_SOURCE_DIR}/libgluon/include")
include_directories("${PROJECT_SOURCE_DIR}/lonestardist/include")

add_library(distbench STATIC src/DistBenchStart.cpp src/DistributedGraphLoader.cpp)
target_include_directories(distbench PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(distbench Galois::cusp Galois::gluon LLVMSupport)
