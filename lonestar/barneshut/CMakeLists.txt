add_executable(barneshut Barneshut.cpp)
add_dependencies(apps barneshut)
target_link_libraries(barneshut PRIVATE galois_shmem lonestar)
install(TARGETS barneshut DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

if(CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(barneshut PRIVATE -ffast-math)
endif()

add_test_scale(small barneshut -n 10000 -steps 1 -seed 0)
