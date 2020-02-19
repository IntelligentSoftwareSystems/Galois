set(sources
  $<TARGET_OBJECTS:galois_shmem_obj>
  $<TARGET_OBJECTS:gllvm_obj>
)

add_library(galois_pangolin STATIC ${sources})

target_link_libraries(galois_pangolin galois_shmem gllvm)

target_include_directories(galois_pangolin PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libpangolin/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties (galois_pangolin PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE On
  POSITION_INDEPENDENT_CODE On
)
