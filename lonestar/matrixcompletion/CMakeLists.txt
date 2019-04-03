set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

find_package(Eigen)
if(Eigen_FOUND)
  include_directories(${Eigen_INCLUDE_DIRS})
  add_definitions(-DHAS_EIGEN -DEIGEN_DONT_PARALLELIZE)
endif()

app(matrixCompletion matrixCompletion.cpp)

add_test_scale(sync-small matrixCompletion -algo=syncALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/bgg.gr")
#add_test_scale(sync-web matrixCompletion -algo=syncALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

add_test_scale(simple-small matrixCompletion -algo=simpleALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/bgg.gr")
#add_test_scale(simple-web matrixCompletion -algo=simpleALS -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

add_test_scale(edge-small matrixCompletion -algo=sgdBlockEdge -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/bgg.gr")
#add_test_scale(edge-web matrixCompletion -algo=sgdBlockEdge -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

add_test_scale(jump-small matrixCompletion -algo=sgdBlockJump -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/bgg.gr")
#add_test_scale(jump-web matrixCompletion -algo=sgdBlockJump -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

add_test_scale(byitems-small matrixCompletion -algo=sgdByItems -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/bgg.gr")
#add_test_scale(byitems-web matrixCompletion -algo=sgdByItems -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")

add_test_scale(byedges-small matrixCompletion -algo=sgdByEdges -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/bgg.gr")
#add_test_scale(byedges-web matrixCompletion -algo=sgdByEdges -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.01 -useSameLatentVector -useDetInit "${BASEINPUT}/weighted/bipartite/floatEdgeWts/netflix.gr")
