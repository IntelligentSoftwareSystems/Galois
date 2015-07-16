#if(HAVE_CXX11_THREAD_YIELD)
#  set(sources ${sources} FileGraphParallel_cpp11.cpp)
#else()
#  set(sources ${sources} FileGraphParallel_pthread.cpp)
#endif()

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

set(exp_libs)
if(USE_EXP)
#  set(exp_libs exp)
endif()

add_galois_library(galois ${sources} LIBS llvm ${exp_libs})
add_galois_library(galois-nothreads  LIBS llvm ${exp_libs})

#add_subdirectory(ll)
add_subdirectory(llvm)
#add_subdirectory(mm)
#add_subdirectory(mm-standalone)

#Long term starts here
add_subdirectory(substrate)
add_subdirectory(runtime)
add_subdirectory(graphs)
#Long term ends here

#include_directories(${gSubstrate_SOURCE_DIR})
#include_directories(${gRuntime_SOURCE_DIR})


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
  set(PAPI_ROOT 
    "/h1/lenharth/papi"
    "/usr/global/tools/papi/bgqos_0/papi-5.0.1-V1R1M2"
    "/usr/lib64/papi-5.1.1/usr")
  find_package(PAPI)
  if(PAPI_FOUND)
    set(GALOIS_USE_PAPI on)
    include_directories(${PAPI_INCLUDE_DIRS})
    target_link_libraries(galois ${PAPI_LIBRARIES})
  else()
    message(WARNING "PAPI required but not found")
  endif()
endif()

if(USE_EMBEDDED_REVISION)
  add_dependencies(galois revision)
endif()


###### Installation ######
install(TARGETS galois
  EXPORT GaloisTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT dev)
