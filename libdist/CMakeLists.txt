set(sources
        src/cuda_device.cpp
        src/Tracer.cpp
        src/DistStatCollector.cpp
        src/buffered.cpp
        src/routed.cpp
        src/pkt_mpi.cpp
        src/NetworkLWCI.cpp
        src/Network.cpp
        src/Barrier.cpp
        src/hash/crc32.cpp
)

add_library(galois_dist STATIC ${sources})

target_link_libraries(galois_dist galois_shmem gllvm)

if (USE_LWCI)
  target_link_libraries(galois_dist ${LWCI_LIBRARY} -lpsm2)
endif()
target_link_libraries(galois_dist ${MPI_CXX_LIBRARIES})

target_include_directories(galois_dist PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
if (USE_LWCI)
  target_include_directories(galois_dist PUBLIC $ENV{LWCI_ROOT}/include)
endif()

set_target_properties (galois_dist PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)
