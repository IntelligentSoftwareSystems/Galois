set(hwtopo CACHE STRING "")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  if(ARCH STREQUAL "mic")
    set(hwtopo "LinuxMIC")
  else()
    set(hwtopo "Linux")
  endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "CYGWIN")
  set(hwtopo "Cygwin")
elseif(CMAKE_SYSTEM_NAME MATCHES "BlueGeneQ")
  set(hwtopo "BlueGeneQ")
elseif(CMAKE_SYSTEM_NAME MATCHES "(Solaris|SunOS)")
  set(hwtopo "Solaris")
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(hwtopo "Darwin")
elseif(CMAKE_SYSTEM_NAME MATCHES "AIX")
  set(hwtopo "AIX")
else()
  message(FATAL_ERROR "Unknown system name: ${CMAKE_SYSTEM_NAME}")
endif()

set(sources Barrier.cpp Barrier_MCS.cpp Barrier_Topo.cpp Barrier_Pthread.cpp
  Barrier_Counting.cpp Barrier_Dissemination.cpp
  Context.cpp FileGraph.cpp 
  #DeterministicWork.cpp
  OCFileGraph.cpp PerThreadStorage.cpp PreAlloc.cpp Sampling.cpp Support.cpp
  Termination.cpp Threads.cpp ThreadPool.cpp Timer.cpp Lockable.cpp)
if(HAVE_CXX11_THREAD_YIELD)
  set(sources ${sources} ThreadPool_cpp11.cpp FileGraphParallel_cpp11.cpp)
else()
  set(sources ${sources} ThreadPool_pthread.cpp FileGraphParallel_pthread.cpp)
endif()

set(include_dirs "${PROJECT_SOURCE_DIR}/include/")
if(USE_EXP)
  file(GLOB exp_sources ../exp/src/*.cpp)
  set(include_dirs ${include_dirs} "${PROJECT_SOURCE_DIR}/exp/include/")
  set(sources ${sources} ${exp_sources})
endif()
set(include_dirs ${include_dirs} "${PROJECT_BINARY_DIR}/include/")

function(add_internal_library name)
  add_library(${name} OBJECT ${ARGN})
endfunction()

function(add_galois_library name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs LIBS)
  cmake_parse_arguments(X "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(objects)
  foreach(lib ${X_LIBS})
    list(APPEND objects "$<TARGET_OBJECTS:${lib}>")
  endforeach()
  add_library(${name} ${X_UNPARSED_ARGUMENTS} ${objects})
endfunction()

add_galois_library(galois ${sources} LIBS llvm mm ll)
add_galois_library(galois-nothreads FileGraph.cpp LIBS llvm mm-standalone ll-standalone)

add_subdirectory(ll)
add_subdirectory(ll-standalone)
add_subdirectory(llvm)
add_subdirectory(mm)
add_subdirectory(mm-standalone)

target_link_libraries(galois ${CMAKE_THREAD_LIBS_INIT})

if(NUMA_FOUND)
  target_link_libraries(galois ${NUMA_LIBRARIES})
  target_link_libraries(galois-nothreads ${NUMA_LIBRARIES})
endif()

if(USE_VTUNE)
  set(VTune_ROOT "/opt/intel/vtune_amplifier_xe_2013;/opt/intel/vtune_amplifier_xe_2011")
  find_package(VTune)
  if(VTune_FOUND)
    set(GALOIS_USE_VTUNE on)
    include_directories(${VTune_INCLUDE_DIRS})
    target_link_libraries(galois ${VTune_LIBRARIES})
    target_link_libraries(galois dl)
  else()
    message(WARNING "VTUNE required but not found")
  endif()
endif()

if(USE_HPCTOOLKIT)
  if(NOT "$ENV{HPCTOOLKIT_HOME}" STREQUAL "")
    set(HPCToolKit_ROOT $ENV{HPCTOOLKIT_HOME})
  endif()
  find_package(HPCToolKit)
  if(HPCToolKit_FOUND)
    set(GALOIS_USE_HPCTOOLKIT on)
    include_directories(${HPCToolKit_INCLUDE_DIRS})
    target_link_libraries(galois ${HPCToolKit_LIBRARIES})
  else()
    message(WARNING "HPCToolKit required but not found")
  endif()
endif()

if(USE_PAPI)
  # Sadly, none of our machines places papi in a well-known place or provides
  # a method for finding papi automatically. Hardcode some constants.
  set(PAPI_ROOT "/h1/lenharth/papi;/usr/global/tools/papi/bgqos_0/papi-5.0.1-V1R1M2;/usr/lib64/papi-5.1.1/usr")
  find_package(PAPI)
  if (PAPI_FOUND)
    set(GALOIS_USE_PAPI on)
    include_directories(${PAPI_INCLUDE_DIRS})
    target_link_libraries(galois ${PAPI_LIBRARIES})
  else()
    message(WARNING "PAPI required but not found")
  endif()
endif()

if(USE_EXP)
	target_link_libraries(galois ${MPI_LIBRARIES})
        target_link_libraries(galois ${GASNET_LIBRARIES})
endif()

if(USE_EMBEDDED_REVISION)
  add_dependencies(galois revision)
endif()

configure_file("${PROJECT_SOURCE_DIR}/include/Galois/config.h.in" "${PROJECT_BINARY_DIR}/include/Galois/config.h")

###### Installation ######
install(TARGETS galois
  EXPORT GaloisTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT dev
  )
install(DIRECTORY ${include_dirs} DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE)
