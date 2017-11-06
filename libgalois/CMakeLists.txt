set(sources
        src/Galois.cpp
        src/Context.cpp
        src/PreAlloc.cpp
        src/Support.cpp
        src/Threads.cpp
        src/Timer.cpp
        src/Mem.cpp
        src/PagePool.cpp
        src/Deterministic.cpp
        src/Substrate.cpp
        src/Version.cpp
        src/PagePool.cpp
        src/Statistics.cpp
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
        src/GraphHelpers.cpp
)

add_library(galois_shmem STATIC ${sources})
target_include_directories(galois_shmem PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(galois_shmem gllvm ${Boost_SERIALIZATION_LIBRARY_RELEASE})

if (CMAKE_HAVE_PTHREAD_H)
   add_definitions(-DGALOIS_HAVE_PTHREAD)
   target_link_libraries(galois_shmem Threads::Threads)
endif()

find_package(NUMA)
if(NUMA_FOUND)
  add_definitions(-DGALOIS_USE_NUMA)
  target_link_libraries(galois_shmem ${NUMA_LIBRARY})
else()
  message(WARNING "No NUMA Support.  Likely poor performance")
endif()

if (VTune_FOUND)
  target_link_libraries(galois_shmem ${VTune_LIBRARIES})
  target_link_libraries(galois_shmem dl)
endif()

add_library(galois_shmem_obj OBJECT ${sources})
target_include_directories(galois_shmem_obj PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties (galois_shmem_obj PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)

