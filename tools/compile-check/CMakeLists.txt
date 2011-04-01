file(GLOB Sources *.cpp)
add_executable(compile-check ${Sources})
target_link_libraries (compile-check galois) 
target_link_libraries (compile-check ${CMAKE_THREAD_LIBS_INIT}) 
