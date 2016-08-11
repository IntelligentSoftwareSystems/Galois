set(sources
        src/DoAllWrap.cpp
        src/ParaMeter.cpp
        src/Queue.cpp
        src/cuda_device.cpp
)

add_library(galois_exp STATIC ${sources})
target_include_directories(galois_exp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_exp galois_runtime)

