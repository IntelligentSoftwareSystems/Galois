add_executable(matrixCompletion matrixCompletion.cpp)
add_dependencies(apps matrixCompletion)
target_link_libraries(matrixCompletion PRIVATE Galois::shmem lonestar)
install(TARGETS matrixCompletion DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

if(CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(matrixCompletion PRIVATE -ffast-math)
endif()

find_package(Eigen CONFIG)
if(Eigen_FOUND)
  target_link_libraries(matrixCompletion Eigen3::Eigen)
  target_compile_definitions(matrixCompletion PRIVATE -DHAS_EIGEN -DEIGEN_DONT_PARALLELIZE)
endif()

if (Eigen_FOUND)
  add_test_scale(small-sync matrixCompletion -algo=syncALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr" NOT_QUICK)

  add_test_scale(small-simple matrixCompletion -algo=simpleALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr" NOT_QUICK)
endif()

add_test_scale(small-edge matrixCompletion -algo=sgdBlockEdge -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr" NOT_QUICK)

add_test_scale(small-jump matrixCompletion -algo=sgdBlockJump -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr" NOT_QUICK)

add_test_scale(small-byitems matrixCompletion -algo=sgdByItems -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr" NOT_QUICK)

add_test_scale(small-byedges matrixCompletion -algo=sgdByEdges -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr" NOT_QUICK)
