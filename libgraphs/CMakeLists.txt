set(sources
        src/FileGraph.cpp
        src/FileGraphParallel_cpp11.cpp
#        src/FileGraphParallel_pthread.cpp
        src/OCFileGraph.cpp
)

add_library(galois_graphs STATIC ${sources})
target_include_directories(galois_graphs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_graphs galois_runtime)

add_library(galois_graphs_obj OBJECT ${sources})
target_include_directories(galois_graphs_obj PUBLIC 
  ${CMAKE_SOURCE_DIR}/libsubstrate/include
  ${CMAKE_SOURCE_DIR}/libruntime/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties (galois_graphs_obj PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)

