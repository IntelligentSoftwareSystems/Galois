set(sources
        src/cuda_device.cpp
        src/Tracer.cpp
        src/StatCollector.cpp
)

add_library(galois_dist STATIC ${sources})
target_include_directories(galois_dist PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(galois_dist galois_runtime gllvm)

target_include_directories(galois_dist PUBLIC 
  ${CMAKE_SOURCE_DIR}/libnet/include
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libsubstrate/include
  ${CMAKE_SOURCE_DIR}/libruntime/include 
  ${CMAKE_SOURCE_DIR}/libgraphs/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
set_target_properties (galois_dist PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)

