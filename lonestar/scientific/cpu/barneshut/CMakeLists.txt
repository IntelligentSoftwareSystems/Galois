add_executable(barneshut-cpu Barneshut.cpp)
add_dependencies(apps barneshut-cpu)
target_link_libraries(barneshut-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS barneshut-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

if(CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(barneshut-cpu PRIVATE -ffast-math)
endif()

add_test_scale(small barneshut-cpu -n 10000 -steps 1 -seed 0)
