file(GLOB Sources *.cpp)
add_executable(barneshut ${Sources})
target_link_libraries (barneshut galois) 
target_link_libraries (barneshut ${CMAKE_THREAD_LIBS_INIT}) 
