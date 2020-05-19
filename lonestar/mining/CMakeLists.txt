include_directories(include)

add_subdirectory(cpu)

if(GALOIS_ENABLE_GPU)
  add_subdirectory(gpu)
endif()
