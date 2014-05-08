set(CMAKE_C_COMPILER "gcc")
# CMAKE_C_FLAGS seems to be clobbered by downstream passes
set(ENV{CFLAGS} "$ENV{CFLAGS} -pthread")
set(CMAKE_CXX_COMPILER "g++")
set(ENV{CXXFLAGS} "$ENV{CXXFLAGS} -pthread")
