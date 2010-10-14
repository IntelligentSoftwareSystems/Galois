file(GLOB Sources *.cpp)
add_executable(delaunayrefinement ${Sources})
target_link_libraries (delaunayrefinement galois) 
target_link_libraries (delaunayrefinement ${CMAKE_THREAD_LIBS_INIT}) 
