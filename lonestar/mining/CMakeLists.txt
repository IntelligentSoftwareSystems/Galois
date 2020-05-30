include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)

add_subdirectory(cpu)

if(GALOIS_ENABLE_GPU)
  add_subdirectory(gpu)
endif()
