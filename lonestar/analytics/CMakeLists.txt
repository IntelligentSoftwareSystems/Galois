add_subdirectory(cpu)

if(GALOIS_ENABLE_DIST)
  add_subdirectory(distributed)
endif()

if(GALOIS_ENABLE_GPU)
  add_subdirectory(gpu)
endif()
