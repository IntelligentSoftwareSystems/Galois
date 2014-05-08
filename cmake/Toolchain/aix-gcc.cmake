set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
# ddn 05-07-14: Errors using longjmp 
set(USE_LONGJMP FALSE)
# CMAKE_C_FLAGS seems to be clobbered by downstream passes
# Need -pthread for CMake to pickup right library directories
set(ENV{CFLAGS} "$ENV{CFLAGS} -pthread")
set(ENV{CXXFLAGS} "$ENV{CXXFLAGS} -pthread")
