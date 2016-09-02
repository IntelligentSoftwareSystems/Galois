set(sources 
            src/gIO.cpp
#            src/Barrier_Counting.cpp
#            src/Barrier_Dissemination.cpp 
#            src/Barrier_MCS.cpp
#            src/Barrier_Pthread.cpp
#            src/Barrier_Simple.cpp
#            src/PerThreadStorage.cpp
            src/NumaMem.cpp
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
