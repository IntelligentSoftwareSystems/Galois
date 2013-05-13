#file(GLOB Sources *.cpp llvm/*.cpp mm/*.cpp ll/*.cpp)
file(GLOB_RECURSE Sources *.cpp)
set(include_dirs "${PROJECT_SOURCE_DIR}/include/")
if(USE_EXP)
  file(GLOB ExpSources ../exp/src/*.cpp)
  set(include_dirs ${include_dirs} "${PROJECT_SOURCE_DIR}/exp/include/")
  set(Sources ${Sources} ${ExpSources})
endif()

configure_file("${PROJECT_SOURCE_DIR}/include/Galois/config.h.in" "${PROJECT_BINARY_DIR}/include/Galois/config.h")
set(include_dirs ${include_dirs} "${PROJECT_BINARY_DIR}/include/")

add_library(galois ${Sources})

target_link_libraries(galois ${CMAKE_THREAD_LIBS_INIT})

if(NUMA_FOUND)
  target_link_libraries(galois ${NUMA_LIBRARIES})
endif()

if(USE_VTUNE)
  find_package(VTune)
  if(VTune_FOUND)
    set(GALOIS_USE_VTUNE on)
    include_directories(${VTune_INCLUDE_DIRS})
    target_link_libraries(galois ${VTune_LIBRARIES})
  else()
    message(WARNING "VTUNE required but not found")
  endif()
endif()

if(USE_PAPI)
  find_package(PAPI)
  if (PAPI_FOUND)
    set(GALOIS_USE_PAPI on)
    include_directories(${PAPI_INCLUDE_DIR})
    target_link_libraries(galois ${PAPI_LIBRARIES})
  else()
    message(WARNING "PAPI required but not found")
  endif()
endif()

if(USE_SUBVERSION_REVISION)
  add_dependencies(galois svnversion)
endif()

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
