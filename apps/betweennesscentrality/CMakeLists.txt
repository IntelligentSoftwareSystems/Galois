file(GLOB Sources *.cpp)
add_executable(betweennesscentrality ${Sources})
target_link_libraries (betweennesscentrality galois) 
target_link_libraries (betweennesscentrality ${CMAKE_THREAD_LIBS_INIT}) 

