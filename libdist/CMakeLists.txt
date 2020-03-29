set(sources
        src/Barrier.cpp
        src/DistGalois.cpp
        src/DistStats.cpp
        src/Network.cpp
        src/NetworkBuffered.cpp
        src/NetworkIOMPI.cpp
        src/NetworkLCI.cpp
)

add_library(galois_dist_async_obj OBJECT ${sources})
target_include_directories(galois_dist_async_obj PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# new galois net library; link to shared memory galois
add_library(galois_dist_async STATIC $<TARGET_OBJECTS:galois_dist_async_obj>)
target_link_libraries(galois_dist_async galois_shmem gllvm)

if (USE_BARE_MPI)
  target_compile_definitions(galois_dist_async PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
endif()

target_compile_definitions(galois_dist_async PRIVATE __GALOIS_HET_ASYNC__=1)

if (USE_LCI)
  add_dependencies(galois_dist_async lci)
  target_link_libraries(galois_dist_async ${LWCI_LIBRARY} -lpsm2)
endif()
target_link_libraries(galois_dist_async ${MPI_CXX_LIBRARIES})

target_include_directories(galois_dist_async PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (USE_LCI)
  target_include_directories(galois_dist_async PUBLIC ${LWCI_INCLUDE})
endif()
