set(sources
        src/buffered.cpp
        src/routed.cpp
        src/pkt_mpi.cpp
        src/NetworkLWCI.cpp
        src/Network.cpp
        src/Barrier.cpp
        src/hash/crc32.cpp
)

add_library(galois_net STATIC ${sources})
target_include_directories(galois_net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (USE_LWCI)
  target_include_directories(galois_net PUBLIC $ENV{LWCI_ROOT}/include)
endif()
target_link_libraries(galois_net galois_substrate)
target_link_libraries(galois_net galois_runtime)
if (USE_LWCI)
  target_link_libraries(galois_net ${LWCI_LIBRARY} -lpsm2)
endif()
target_link_libraries(galois_net ${MPI_CXX_LIBRARIES})
