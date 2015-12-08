set(sources
        src/Context.cpp
        src/PreAlloc.cpp
        src/Support.cpp
        src/Threads.cpp
        src/Sampling.cpp
        src/Timer.cpp
        src/Mem.cpp
        src/PagePool.cpp
        src/Deterministic.cpp
        src/Substrate.cpp
        src/Version.cpp
        src/PagePool.cpp
        src/Tracer.cpp
)

add_library(galois_runtime STATIC ${sources})
target_include_directories(galois_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_runtime galois_substrate)
target_link_libraries(galois_runtime gllvm)
if (NUMA_FOUND)
  target_link_libraries(galois_runtime ${NUMA_LIBRARY})
endif()
