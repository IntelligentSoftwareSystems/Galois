set(sources
        src/Timer.cpp
        src/Version.cpp
        src/Threads.cpp
        src/Runtime/Barrier.cpp
        src/Runtime/Barrier_Topo.cpp
        src/Runtime/ThreadPool.cpp
        src/Runtime/HWTopoLinux.cpp
        src/Runtime/SimpleLock.cpp
        src/Runtime/PtrLock.cpp
        src/Runtime/PerThreadStorage.cpp
        src/Runtime/PageAlloc.cpp
        src/Runtime/Termination.cpp
        src/Runtime/EnvCheck.cpp
        src/Runtime/SyncContext.cpp
        src/Runtime/StatCollector.cpp
        src/Runtime/PagePool.cpp
        src/Runtime/Support.cpp
        src/Runtime/Sampling.cpp
        src/Runtime/ErrorFeedBack.cpp
        src/Runtime/GaloisConfig.cpp
        src/Runtime/Statistics.cpp
        src/Runtime/NumaMem.cpp
        )

add_library(galois STATIC ${sources})
target_include_directories(galois PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (NUMA_FOUND)
  target_link_libraries(galois ${NUMA_LIBRARY})
endif()
