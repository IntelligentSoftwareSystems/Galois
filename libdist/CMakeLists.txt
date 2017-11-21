set(sources
        src/DistGalois.cpp
        src/cuda_device.cpp
        src/Tracer.cpp
        src/DistStats.cpp
        src/NetworkBuffered.cpp
        src/NetworkIOMPI.cpp
        src/NetworkIOLWCI.cpp
        src/Network.cpp
        src/Barrier.cpp
        src/SyncStructures.cpp
        src/GlobalObj.cpp
        src/DistributedGraph.cpp
        src/DistributedGraphLoader.cpp
        src/DynamicBitset.cpp
)
add_library(galois_dist STATIC ${sources})

target_link_libraries(galois_dist galois_shmem gllvm)

if (USE_BARE_MPI)
  target_compile_definitions(galois_dist PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
endif()

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
  target_include_directories(galois_dist PUBLIC ${LWCI_INCLUDE})
endif()

set_target_properties (galois_dist PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)
