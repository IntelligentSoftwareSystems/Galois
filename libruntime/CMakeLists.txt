set(sources
        src/Support.cpp
        src/Threads.cpp
        src/Timer.cpp
        src/Mem.cpp
        src/Deterministic.cpp
        src/Substrate.cpp
        src/Version.cpp
)

add_library(galois_runtime STATIC ${sources})
target_include_directories(galois_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_runtime galois)
target_link_libraries(galois_runtime gllvm)
if (NUMA_FOUND)
  target_link_libraries(galois_runtime ${NUMA_LIBRARY})
endif()
