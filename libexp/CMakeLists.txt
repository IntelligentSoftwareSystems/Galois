set(sources
        src/DoAllWrap.cpp
        src/ParaMeter.cpp
        src/Queue.cpp
)

add_library(galois_exp STATIC ${sources})
target_include_directories(galois_exp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_exp galois_runtime gllvm)

add_library(galois_exp_obj OBJECT ${sources})
target_include_directories(galois_exp_obj PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libsubstrate/include
  ${CMAKE_SOURCE_DIR}/libruntime/include 
  ${CMAKE_SOURCE_DIR}/libgraphs/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
set_target_properties (galois_exp_obj PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)

