add_subdirectory(cpu)

if(ENABLE_DIST_GALOIS)
  add_subdirectory(distributed)
endif()

if(ENABLE_HETERO_GALOIS)
  add_subdirectory(gpu)
endif()
