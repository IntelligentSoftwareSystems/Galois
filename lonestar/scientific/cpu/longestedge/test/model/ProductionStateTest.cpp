#include "../catch.hpp"
#include "../../src/model/Graph.h"
#include "../testUtils.cpp"
#include "../../src/model/ProductionState.h"

TEST_CASE("ProductionState construction lengths test") {
  galois::SharedMemSys G;
  Graph graph{};
  std::vector<GNode> nodes = generateSampleGraph(graph);
  ConnectivityManager connManager{graph};
  ProductionState pState{connManager, nodes[4], false,
                         [](double, double) { return 0.; }};

  REQUIRE(fabs(pState.getLengths()[0] - 1) <= 1e-6);
  REQUIRE(fabs(pState.getLengths()[1] - 1) <= 1e-6);
  REQUIRE(fabs(pState.getLengths()[2] - sqrt(2)) <= 1e-6);
}
