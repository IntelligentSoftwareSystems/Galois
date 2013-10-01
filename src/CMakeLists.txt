set(sources Barrier.cpp Context.cpp FileGraph.cpp FileGraphParallel.cpp
  OCFileGraph.cpp PerThreadStorage.cpp PreAlloc.cpp Sampling.cpp Support.cpp
  Termination.cpp Threads.cpp ThreadPool_pthread.cpp Timer.cpp)
set(include_dirs "${PROJECT_SOURCE_DIR}/include/")
if(USE_EXP)
  file(GLOB exp_sources ../exp/src/*.cpp)
  set(include_dirs ${include_dirs} "${PROJECT_SOURCE_DIR}/exp/include/")
  set(sources ${sources} ${exp_sources})
endif()
set(include_dirs ${include_dirs} "${PROJECT_BINARY_DIR}/include/")

include(ParseArguments)

if(CMAKE_VERSION VERSION_GREATER 2.8.8 OR CMAKE_VERSION VERSION_EQUAL 2.8.8)
  function(add_internal_library name)
    add_library(${name} OBJECT ${ARGN})
  endfunction()

  function(add_galois_library name)
    PARSE_ARGUMENTS(X "LIBS" "" ${ARGN})
    set(objects)
    foreach(lib ${X_LIBS})
      list(APPEND objects "$<TARGET_OBJECTS:${lib}>")
    endforeach()
    add_library(${name} ${X_DEFAULT_ARGS} ${objects})
  endfunction()
else()
  function(add_internal_library name)
    add_library(${name} ${ARGN})
    install(TARGETS ${name}
      EXPORT GaloisTargets
      RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
      LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
      ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT dev
      )
    export(TARGETS ${name} APPEND FILE
      "${PROJECT_BINARY_DIR}/GaloisTargets.cmake")
  endfunction()

  function(add_galois_library name)
    PARSE_ARGUMENTS(X "LIBS" "" ${ARGN})
    add_library(${name} ${X_DEFAULT_ARGS})
    foreach(lib ${X_LIBS})
      target_link_libraries(${name} ${lib})
    endforeach()
  endfunction()
endif()

add_galois_library(galois ${sources} LIBS llvm mm ll)
add_galois_library(galois-nothreads FileGraph.cpp LIBS llvm mm-nonuma ll)

add_subdirectory(ll)
add_subdirectory(llvm)
add_subdirectory(mm)
add_subdirectory(mm-nonuma)

target_link_libraries(galois ${CMAKE_THREAD_LIBS_INIT})

if(NUMA_FOUND)
  target_link_libraries(galois ${NUMA_LIBRARIES})
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
  set(PAPI_ROOT "/h1/lenharth/papi;/usr/global/tools/papi/bgqos_0/papi-5.0.1-V1R1M2")
  find_package(PAPI)
  if (PAPI_FOUND)
    set(GALOIS_USE_PAPI on)
    include_directories(${PAPI_INCLUDE_DIRS})
    target_link_libraries(galois ${PAPI_LIBRARIES})
  else()
    message(WARNING "PAPI required but not found")
  endif()
endif()

if(USE_SUBVERSION_REVISION)
  add_dependencies(galois svnversion)
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
