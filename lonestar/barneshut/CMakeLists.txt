app(barneshut Barneshut.cpp)
if(CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(barneshut PRIVATE -ffast-math)
endif()

add_test_scale(small barneshut -n 10000 -steps 1 -seed 0)
