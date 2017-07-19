set(sources
        src/Context.cpp
        src/PreAlloc.cpp
        src/Support.cpp
        src/Threads.cpp
        src/Sampling.cpp
        src/Timer.cpp
        src/Mem.cpp
        src/PagePool.cpp
        src/Deterministic.cpp
        src/Substrate.cpp
        src/Version.cpp
        src/PagePool.cpp
        src/Tracer.cpp
        src/StatCollector.cpp
)

add_library(galois_runtime STATIC ${sources})
target_include_directories(galois_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_runtime galois_net)
target_link_libraries(galois_runtime galois_substrate)
target_link_libraries(galois_runtime gllvm)
if (NUMA_FOUND)
  target_link_libraries(galois_runtime ${NUMA_LIBRARY})
endif()
if (VTune_FOUND)
  target_link_libraries(galois_runtime ${VTune_LIBRARY})
  target_link_libraries(galois_runtime dl)
endif()

add_library(galois_runtime_obj OBJECT ${sources})
target_include_directories(galois_runtime_obj PUBLIC 
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libsubstrate/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties (galois_runtime_obj PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)

