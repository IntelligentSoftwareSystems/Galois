set(sources
        src/cuda_device.cpp
        src/SyncStructures.cpp
        src/GlobalObj.cpp
        src/DistributedGraph.cpp
        src/DistributedGraphLoader.cpp
)
add_library(galois_gluon STATIC ${sources})
add_library(galois_gluon_async STATIC ${sources})

target_link_libraries(galois_gluon galois_shmem galois_dist gllvm)
target_link_libraries(galois_gluon_async galois_shmem galois_dist gllvm)

if(REPORT_COMM_STATS)
  target_compile_definitions(galois_gluon PRIVATE MORE_COMM_STATS=1)
  target_compile_definitions(galois_gluon_async PRIVATE MORE_COMM_STATS=1)
endif()
target_compile_definitions(galois_gluon_async PRIVATE __GALOIS_HET_ASYNC__=1)
if (USE_BARE_MPI)
  target_compile_definitions(galois_gluon PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  target_compile_definitions(galois_gluon_async PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
endif()

if (USE_LCI)
  add_dependencies(galois_gluon lci)
  add_dependencies(galois_gluon_async lci)
  target_link_libraries(galois_gluon ${LWCI_LIBRARY} -lpsm2)
  target_link_libraries(galois_gluon_async ${LWCI_LIBRARY} -lpsm2)
endif()
target_link_libraries(galois_gluon ${MPI_CXX_LIBRARIES})
target_link_libraries(galois_gluon_async ${MPI_CXX_LIBRARIES})

target_include_directories(galois_gluon PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(galois_gluon_async PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if (USE_LCI)
  target_include_directories(galois_gluon PUBLIC ${LWCI_INCLUDE})
  target_include_directories(galois_gluon_async PUBLIC ${LWCI_INCLUDE})
endif()

set_target_properties (galois_gluon PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE On
  POSITION_INDEPENDENT_CODE On
)
set_target_properties (galois_gluon_async PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE On
  POSITION_INDEPENDENT_CODE On
)
