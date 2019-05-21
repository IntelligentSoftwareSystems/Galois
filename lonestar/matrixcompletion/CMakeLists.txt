set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

find_package(Eigen)
if(Eigen_FOUND)
  include_directories(${Eigen_INCLUDE_DIRS})
  add_definitions(-DHAS_EIGEN -DEIGEN_DONT_PARALLELIZE)
endif()

app(matrixCompletion matrixCompletion.cpp)

if (Eigen_FOUND)
  add_test_scale(small-sync matrixCompletion -algo=syncALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr")
  #add_test_scale(web-sync matrixCompletion -algo=syncALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

  add_test_scale(small-simple matrixCompletion -algo=simpleALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr")
  #add_test_scale(web-simple matrixCompletion -algo=simpleALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")
endif()

add_test_scale(small-edge matrixCompletion -algo=sgdBlockEdge -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr")
#add_test_scale(web-edge matrixCompletion -algo=sgdBlockEdge -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

add_test_scale(small-jump matrixCompletion -algo=sgdBlockJump -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr")
#add_test_scale(web-jump matrixCompletion -algo=sgdBlockJump -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

add_test_scale(small-byitems matrixCompletion -algo=sgdByItems -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr")
#add_test_scale(web-byitems matrixCompletion -algo=sgdByItems -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

add_test_scale(small-byedges matrixCompletion -algo=sgdByEdges -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr")
#add_test_scale(web-byedges matrixCompletion -algo=sgdByEdges -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")
