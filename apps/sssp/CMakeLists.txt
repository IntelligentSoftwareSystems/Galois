file(GLOB Sources *.cpp)
add_executable(sssp ${Sources})
target_link_libraries (sssp galois) 
target_link_libraries (sssp ${CMAKE_THREAD_LIBS_INIT}) 
