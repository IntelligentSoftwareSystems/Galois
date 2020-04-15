set(sources
        src/DoAllWrap.cpp
        src/ParaMeter.cpp
        src/Queue.cpp
)

add_library(galois_exp_obj OBJECT ${sources})
target_include_directories(galois_exp_obj PUBLIC
  ${CMAKE_SOURCE_DIR}/libsubstrate/include
  ${CMAKE_SOURCE_DIR}/libruntime/include
  ${CMAKE_SOURCE_DIR}/libgraphs/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(galois_exp STATIC $<TARGET_OBJECTS:galois_exp_obj>)
target_include_directories(galois_exp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(galois_exp galois_runtime LLVMSupport)

