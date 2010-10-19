file(GLOB Sources *.cpp)
add_executable(preflowpush ${Sources})
target_link_libraries (preflowpush galois) 
target_link_libraries (preflowpush ${CMAKE_THREAD_LIBS_INIT}) 
