include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)

add_subdirectory(libminingbench)

add_subdirectory(cpu)

if(GALOIS_ENABLE_GPU)
  target_compile_definitions(miningbench PRIVATE GALOIS_ENABLE_GPU=1)
  add_subdirectory(gpu)
endif()
