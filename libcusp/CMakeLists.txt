set(sources
  $<TARGET_OBJECTS:galois_shmem_obj>
  $<TARGET_OBJECTS:galois_dist_async_obj>
  $<TARGET_OBJECTS:gllvm_obj>
)

add_library(galois_cusp STATIC ${sources})

target_link_libraries(galois_cusp galois_shmem galois_dist_async gllvm)
target_link_libraries(galois_cusp ${MPI_CXX_LIBRARIES})

target_include_directories(galois_cusp PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_SOURCE_DIR}/libcusp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
