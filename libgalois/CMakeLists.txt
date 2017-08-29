set(sources
        src/Init.cpp
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
        src/StatCollector.cpp
        src/Barrier.cpp
        src/Barrier_Counting.cpp
        src/Barrier_Dissemination.cpp 
        src/Barrier_MCS.cpp
        src/Barrier_Topo.cpp 
        src/Barrier_Pthread.cpp
        src/Barrier_Simple.cpp
        src/gIO.cpp
        src/ThreadPool.cpp
        src/SimpleLock.cpp
        src/PtrLock.cpp
        src/EnvCheck.cpp
        src/PerThreadStorage.cpp
        src/HWTopoLinux.cpp
        src/Termination.cpp
        src/NumaMem.cpp
        src/PageAlloc.cpp
        src/SubsInit.cpp
        src/FileGraph.cpp
        src/FileGraphParallel_cpp11.cpp
#        src/FileGraphParallel_pthread.cpp
        src/OCFileGraph.cpp
)

add_library(galois STATIC ${sources})
target_include_directories(galois PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_link_libraries(galois gllvm)
if (NUMA_FOUND)
  target_link_libraries(galois ${NUMA_LIBRARY})
endif()

if (VTune_FOUND)
  target_link_libraries(galois ${VTune_LIBRARIES})
  target_link_libraries(galois dl)
endif()

add_library(galois_obj OBJECT ${sources})
target_include_directories(galois_obj PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties (galois_obj PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)

