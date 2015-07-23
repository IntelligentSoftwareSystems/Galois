project(gRuntime)

set(sources
        src/Context.cpp
        src/PreAlloc.cpp
        src/Support.cpp
        src/Threads.cpp
        src/Sampling.cpp
        src/Timer.cpp
        src/Mem.cpp
        src/NumaMem.cpp
        src/PageAlloc.cpp
        src/Deterministic.cpp
)

add_library(galois_runtime STATIC ${sources})
target_include_directories(galois_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_runtime galois_substrate)
