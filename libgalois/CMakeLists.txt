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
        src/Profile.cpp
        src/EnvCheck.cpp
        src/PerThreadStorage.cpp
        src/Termination.cpp
        src/NumaMem.cpp
        src/PageAlloc.cpp
        src/SubsInit.cpp
        src/FileGraph.cpp
        src/FileGraphParallel.cpp
        src/OCFileGraph.cpp
        src/GraphHelpers.cpp
        src/ParaMeter.cpp
        src/DynamicBitset.cpp
        src/Tracer.cpp
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  list(APPEND sources src/HWTopoDarwin.cpp)
else()
  include(CheckSchedSetAffinity)
  if (SCHED_SETAFFINITY_FOUND)
    add_definitions(-DGALOIS_USE_SCHED_SETAFFINITY)
  elseif (USE_STRICT_CONFIG)
    message(FATAL_ERROR "Need sched_setaffinity")
  endif()

  list(APPEND sources src/HWTopoLinux.cpp)
endif()

add_library(galois_shmem_obj OBJECT ${sources})
target_include_directories(galois_shmem_obj PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${RELATIVE_INCLUDE_FROM_INSTALL_PREFIX}>
)
set_target_properties (galois_shmem_obj PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE On
  POSITION_INDEPENDENT_CODE On
)

add_library(galois_shmem $<TARGET_OBJECTS:galois_shmem_obj>)
target_include_directories(galois_shmem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${GALOIS_RELATIVE_INCLUDE_DIR}>
)

if (SCHED_SETAFFINITY_FOUND)
  target_link_libraries(galois_shmem ${SCHED_SETAFFINITY_LIBRARIES})
endif()

target_link_libraries(galois_shmem ${Boost_SERIALIZATION_LIBRARY_RELEASE})

if (CMAKE_HAVE_PTHREAD_H)
   add_definitions(-DGALOIS_HAVE_PTHREAD)
   target_link_libraries(galois_shmem Threads::Threads)
endif()

find_package(NUMA)
if (NUMA_FOUND)
  add_definitions(-DGALOIS_USE_NUMA)
  target_link_libraries(galois_shmem ${NUMA_LIBRARY})
else()
  message(WARNING "No NUMA Support.  Likely poor performance for multi-socket systems.")
endif()

if (VTune_FOUND)
  target_link_libraries(galois_shmem ${VTune_LIBRARIES})
  target_link_libraries(galois_shmem dl)
endif()

install(DIRECTORY include DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}/.." COMPONENT dev
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".git" EXCLUDE
  PATTERN ".svn" EXCLUDE)

install(TARGETS galois_shmem
  EXPORT GaloisTargets
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT lib
  INCLUDES DESTINATION "${RELATIVE_INCLUDE_FROM_INSTALL_PREFIX}"
)
