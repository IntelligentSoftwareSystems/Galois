add_subdirectory(cpu)

if(GALOIS_ENABLE_DIST)
  add_subdirectory(distributed)
endif()

if(ENABLE_HETERO_GALOIS)
  add_subdirectory(gpu)
endif()
