set(sources
        src/cuda_device.cpp
        src/SyncStructures.cpp
        src/GlobalObj.cpp
        src/GluonSubstrate.cpp
)
add_library(galois_gluon STATIC ${sources})

target_link_libraries(galois_gluon galois_shmem galois_dist_async gllvm)

if(REPORT_COMM_STATS)
  target_compile_definitions(galois_gluon PRIVATE MORE_COMM_STATS=1)
endif()
if (USE_BARE_MPI)
  target_compile_definitions(galois_gluon PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
endif()

if (USE_LCI)
  add_dependencies(galois_gluon lci)
  target_link_libraries(galois_gluon ${LWCI_LIBRARY} -lpsm2)
endif()
target_link_libraries(galois_gluon ${MPI_CXX_LIBRARIES})

target_include_directories(galois_gluon PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if (USE_LCI)
  target_include_directories(galois_gluon PUBLIC ${LWCI_INCLUDE})
endif()
