project(gRuntime)

set(sources
        src/Context.cpp
        src/PreAlloc.cpp
        src/Support.cpp
        src/Threads.cpp
        src/Sampling.cpp
        src/Termination.cpp
        src/Timer.cpp
        src/Mem.cpp
        src/NumaMem.cpp
        src/PageAlloc.cpp
        src/Deterministic.cpp
)

add_library(galois_runtime STATIC ${sources})
target_link_libraries(galois_runtime galois_substrate)

#target_include_directories(galois_runtime PUBLIC ${gSubstrate_SOURCE_DIR}/include)

configure_file("${PROJECT_SOURCE_DIR}/include/Galois/config.h.in" "${PROJECT_BINARY_DIR}/include/Galois/config.h")

