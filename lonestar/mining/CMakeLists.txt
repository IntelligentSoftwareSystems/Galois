include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)

add_subdirectory(libminingbench)

add_subdirectory(cpu)

if(GALOIS_ENABLE_GPU)
  add_subdirectory(gpu)
endif()
