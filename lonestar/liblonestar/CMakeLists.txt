add_library(lonestar STATIC src/BoilerPlate.cpp)

target_include_directories(lonestar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(lonestar Galois::shmem LLVMSupport)
