set(sources 
)

# FIXME: duplicate library target. 
add_library(galois_substrate STATIC ${sources})
target_include_directories(galois_substrate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(galois_substrate_obj OBJECT ${sources})
target_include_directories(galois_substrate_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties (galois_substrate_obj PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)

