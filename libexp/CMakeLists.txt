set(sources
        src/DoAllWrap.cpp
        src/ParaMeter.cpp
        src/Queue.cpp
)

add_library(galois_exp SHARED ${sources})
add_library(galois_exp_static STATIC ${sources})

target_include_directories(galois_exp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(galois_exp_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_exp galois gllvm)
target_link_libraries(galois_exp_static galois_static gllvm_static)

