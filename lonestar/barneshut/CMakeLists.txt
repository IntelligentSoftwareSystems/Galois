app(barneshut Barneshut.cpp)
if(CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(barneshut PRIVATE -fext-numeric-literals -ffast-math)
endif()
