set(sources
        src/buffered.cpp
        src/routed.cpp
        src/pkt_mpi.cpp
        src/NetworkIOMV.cpp
        src/Network.cpp
	      src/Barrier.cpp
        src/hash/crc32.cpp
)

add_library(galois_net STATIC ${sources})
target_include_directories(galois_net PUBLIC /home1/03490/tg827895/mv-knl/include)
target_include_directories(galois_net PUBLIC /opt/ofed/include)
target_include_directories(galois_net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(galois_net /home1/03490/tg827895/mv-knl/lib/liblwci.a -lpsm2)
target_link_libraries(galois_net galois_substrate)
target_link_libraries(galois_net galois_runtime)
target_link_libraries(galois_net ${MPI_CXX_LIBRARIES})
