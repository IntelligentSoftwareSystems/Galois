add_library(distbench STATIC src/Start.cpp src/Input.cpp src/Output.cpp)
target_include_directories(distbench PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(distbench Galois::cusp Galois::gluon LLVMSupport)
