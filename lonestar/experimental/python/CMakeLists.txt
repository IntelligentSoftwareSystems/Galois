add_library(galois_python SHARED 
  PythonGraph.cpp
  PythonGraphReport.cpp
  PythonGraphSimulation.cpp
  ../graphsimulation/GraphSimulation.cpp
  #AnalyzeBFS.cpp
  #SearchSubgraph.cpp
  #PageRank.cpp
  #Filter.cpp
  #Auxiliary.cpp
  #Reachability.cpp
  #Coarsen.cpp
  $<TARGET_OBJECTS:gllvm_obj>
  $<TARGET_OBJECTS:galois_shmem_obj> 
)
target_include_directories(galois_python PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libgalois/experimental/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(galois_python ${Boost_SERIALIZATION_LIBRARY_RELEASE})

if (USE_VTUNE)
  find_package(VTune)
  if (VTune_FOUND)
    target_link_libraries(galois_python ${VTune_LIBRARIES})
  endif()
endif()

if (CMAKE_HAVE_PTHREAD_H)
   add_definitions(-DGALOIS_HAVE_PTHREAD)
   target_link_libraries(galois_python Threads::Threads)
endif()

find_package(NUMA)
if(NUMA_FOUND)
  add_definitions(-DGALOIS_USE_NUMA)
  target_link_libraries(galois_python ${NUMA_LIBRARY})
else()
  message(WARNING "No NUMA Support.  Likely poor performance")
endif()
