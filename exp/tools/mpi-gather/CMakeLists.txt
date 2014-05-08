find_package(MPI)
if(MPI_FOUND) 
  set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
  set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")

  include_directories(${MPI_INCLUDE_PATH})
  add_executable(mpi-gather mpi-gather.cpp)
  target_link_libraries(mpi-gather ${MPI_LIBRARIES})
endif()
