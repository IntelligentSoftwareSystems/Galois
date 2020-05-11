#include "../catch.hpp"
#include "../../src/productions/Production1.h"
#include "../../src/model/Graph.h"
#include "../../src/model/Coordinates.h"
#include "../../src/model/NodeData.h"
#include "../testUtils.cpp"

std::vector<GNode> generateTest1Graph(Graph &graph) {
    std::vector<GNode> nodes;
    ConnectivityManager connManager{graph};
    nodes.push_back(connManager.createNode(NodeData{false, Coordinates{0, 0, 0}, false}));
    nodes.push_back(connManager.createNode(NodeData{false, Coordinates{0, 1, 0}, false}));
    nodes.push_back(connManager.createNode(NodeData{false, Coordinates{1, 0, 0}, false}));
    nodes.push_back(connManager.createNode(NodeData{false, Coordinates{1, 1, 0}, false}));
    nodes.push_back(connManager.createNode(NodeData{false, Coordinates{2, 0, 0}, false}));
    nodes.push_back(connManager.createNode(NodeData{false, Coordinates{2, 1, 0}, false}));

    connManager.createEdge(nodes[0], nodes[1], true, Coordinates{0, 0.5, 0}, 1);
    connManager.createEdge(nodes[1], nodes[3], true, Coordinates{0.5, 1, 0}, 1);
    connManager.createEdge(nodes[2], nodes[3], false, Coordinates{1, 0.5, 0}, 1);
    connManager.createEdge(nodes[0], nodes[2], true, Coordinates{0.5, 0, 0}, 1);
    connManager.createEdge(nodes[2], nodes[4], true, Coordinates{1.5, 0, 0}, 1);
    connManager.createEdge(nodes[3], nodes[5], true, Coordinates{1.5, 1, 0}, 1);
    connManager.createEdge(nodes[4], nodes[5], true, Coordinates{2, 0.5, 0}, 1);
    connManager.createEdge(nodes[1], nodes[2], false, Coordinates{0.5, 0.5, 0}, sqrt(2));
    connManager.createEdge(nodes[2], nodes[5], false, Coordinates{1.5, 0.5, 0}, sqrt(2));

    nodes.push_back(connManager.createInterior(nodes[0], nodes[1], nodes[2]));
    nodes.push_back(connManager.createInterior(nodes[1], nodes[3], nodes[2]));
    nodes.push_back(connManager.createInterior(nodes[2], nodes[3], nodes[5]));
    nodes.push_back(connManager.createInterior(nodes[2], nodes[4], nodes[5]));
    return nodes;
}

TEST_CASE( "Production1 simple Test" ) {
    galois::SharedMemSys G;
    Graph graph{};
    vector<GNode> nodes = generateSampleGraph(graph);
    nodes[5]->getData().setToRefine(true);
    galois::UserContext<GNode> ctx;
    ConnectivityManager connManager{graph};
    Production1 production{connManager};
    ProductionState pState(connManager, nodes[5], false, [](double x, double y){ return 0.;});
    production.execute(pState, ctx);

    REQUIRE(countHEdges(graph) == 3);
    REQUIRE(countVertices(graph) == 5);
}

TEST_CASE( "Production1 complex Test" ) {
    galois::SharedMemSys G;
    Graph graph{};
    vector<GNode> nodes = generateTest1Graph(graph);
    nodes[6]->getData().setToRefine(true);
    nodes[7]->getData().setToRefine(true);
    nodes[8]->getData().setToRefine(true);
    nodes[9]->getData().setToRefine(true);
    galois::UserContext<GNode> ctx;
    ConnectivityManager connManager{graph};
    Production1 production{connManager};
    ProductionState pState1(connManager, nodes[6], false, [](double x, double y){ return 0.;});
    ProductionState pState2(connManager, nodes[7], false, [](double x, double y){ return 0.;});
    ProductionState pState3(connManager, nodes[8], false, [](double x, double y){ return 0.;});
    ProductionState pState4(connManager, nodes[9], false, [](double x, double y){ return 0.;});
    production.execute(pState1, ctx);
    production.execute(pState2, ctx);
    production.execute(pState3, ctx);
    production.execute(pState4, ctx);

    REQUIRE(countHEdges(graph) == 6);
    REQUIRE(countVertices(graph) == 8);
}

