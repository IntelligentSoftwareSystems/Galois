file(GLOB Sources *.cpp)
add_executable(delaunayrefinement ${Sources})
target_link_libraries (delaunayrefinement galois) 
target_link_libraries (delaunayrefinement ${CMAKE_THREAD_LIBS_INIT}) 

if(WITH_VTUNE)
target_link_libraries (delaunayrefinement /opt/intel/vtune_amplifier_xe_2011/lib64/libittnotify.a)
target_link_libraries (delaunayrefinement dl)
add_definitions(-DWITH_VTUNE)
endif(WITH_VTUNE)

add_definitions("-ffast-math")
