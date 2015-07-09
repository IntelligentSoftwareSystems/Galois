project(gSubstrate)

set(sources 
            src/Barrier.cpp
            src/Barrier_Counting.cpp
            src/Barrier_Dissemination.cpp 
            src/Barrier_MCS.cpp
            src/Barrier_Topo.cpp 
            src/ThreadPool.cpp
            src/SimpleLock.cpp
            src/PtrLock.cpp
            src/ThreadPool_cpp11.cpp
            src/EnvCheck.cpp
)

if (HAVE_PTHREAD)
  set(sources ${sources} src/Barrier_Simple.cpp src/Barrier_Pthread.cpp src/ThreadPool_pthread.cpp)
endif()



add_library(galois_substrate STATIC ${sources})
#target_include_directories(galois_substrate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
