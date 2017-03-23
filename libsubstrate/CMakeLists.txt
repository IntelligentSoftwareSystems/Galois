set(sources 
            src/gIO.cpp
            src/Barrier.cpp
            src/Barrier_Counting.cpp
            src/Barrier_Dissemination.cpp 
            src/Barrier_MCS.cpp
            src/Barrier_Topo.cpp 
            src/Barrier_Pthread.cpp
            src/Barrier_Simple.cpp
            src/ThreadPool.cpp
            src/SimpleLock.cpp
            src/PtrLock.cpp
            src/EnvCheck.cpp
            src/PerThreadStorage.cpp
            src/HWTopoLinux.cpp
            src/Termination.cpp
            src/NumaMem.cpp
            src/PageAlloc.cpp
)

add_library(galois_substrate STATIC ${sources})
target_include_directories(galois_substrate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (CMAKE_HAVE_PTHREAD_H)
   add_definitions(-DGALOIS_HAVE_PTHREAD)
   target_link_libraries(galois_substrate Threads::Threads)
endif()

find_package(NUMA)
if(NUMA_FOUND)
  add_definitions(-DGALOIS_USE_NUMA)
  target_link_libraries(galois_substrate ${NUMA_LIBRARY})
else()
  message(WARNING "No NUMA Support.  Likely poor performance")
endif()

add_library(galois_substrate_obj OBJECT ${sources})
target_include_directories(galois_substrate_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties (galois_substrate_obj PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)

