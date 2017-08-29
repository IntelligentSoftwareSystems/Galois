set(sources
        src/cuda_device.cpp
        src/Tracer.cpp
        src/StatCollector.cpp
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

target_include_directories(galois_dist PUBLIC 
  ${CMAKE_SOURCE_DIR}/libnet/include
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
set_target_properties (galois_dist PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)
