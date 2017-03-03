set(sources
        src/Runtime/Barrier_Topo.cpp
        src/Runtime/EnvCheck.cpp
        src/Runtime/HWTopoLinux.cpp
        src/Runtime/NumaMem.cpp
        src/Runtime/PageAlloc.cpp
        src/Runtime/PerThreadStorage.cpp
        src/Runtime/PtrLock.cpp
        src/Runtime/SimpleLock.cpp
        src/Runtime/Termination.cpp
        src/Runtime/ThreadPool.cpp
        src/Runtime/ErrorFeedBack.cpp
        src/Runtime/PagePool.cpp
        src/Threads.cpp
        )

add_library(galois SHARED ${sources})
add_library(galois_static STATIC ${sources})
add_library(galois_obj OBJECT ${sources})

target_include_directories(galois PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(galois_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(galois_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (NUMA_FOUND)
  target_link_libraries(galois ${NUMA_LIBRARY})
  target_link_libraries(galois_static ${NUMA_LIBRARY})
endif()

if (VTune_FOUND)
  target_link_libraries(galois ${VTune_LIBRARIES})
  target_link_libraries(galois_static ${VTune_LIBRARIES})
endif()
