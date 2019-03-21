set(sources
        src/Barrier.cpp
        src/DistStats.cpp
        src/Network.cpp
        src/NetworkBuffered.cpp
        src/NetworkIOMPI.cpp
        src/NetworkLCI.cpp
)
# new galois net library; link to shared memory galois
add_library(galois_net STATIC ${sources})
target_link_libraries(galois_net galois_shmem gllvm)

if (USE_BARE_MPI)
  target_compile_definitions(galois_net PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
endif()

if (USE_LCI)
  add_dependencies(galois_net lci)
  target_link_libraries(galois_net ${LWCI_LIBRARY} -lpsm2)
endif()
target_link_libraries(galois_net ${MPI_CXX_LIBRARIES})

target_include_directories(galois_net PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

if (USE_LCI)
  target_include_directories(galois_net PUBLIC ${LWCI_INCLUDE})
endif()

set_target_properties (galois_net PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)
