#ifndef GALOIS_GRAPHGENERATOR_H
#define GALOIS_GRAPHGENERATOR_H

#include "Utils.h"

class GraphGenerator {
public:
  static void generateSampleGraph(Graph& graph) {
    vector<GNode> nodes;
    ConnectivityManager connManager{graph};
    nodes.push_back(
        connManager.createNode(NodeData{false, Coordinates{0, 0, 0}, false}));
    nodes.push_back(
        connManager.createNode(NodeData{false, Coordinates{0, 1, 0}, false}));
    nodes.push_back(
        connManager.createNode(NodeData{false, Coordinates{1, 0, 0}, false}));
    nodes.push_back(
        connManager.createNode(NodeData{false, Coordinates{1, 1, 0}, false}));

    connManager.createEdge(nodes[0], nodes[1], true, Coordinates{0, 0.5, 0}, 1);
    connManager.createEdge(nodes[1], nodes[3], true, Coordinates{0.5, 1, 0}, 1);
    connManager.createEdge(nodes[2], nodes[3], true, Coordinates{1, 0.5, 0}, 1);
    connManager.createEdge(nodes[0], nodes[2], true, Coordinates{0.5, 0, 0}, 1);
    connManager.createEdge(nodes[3], nodes[0], false, Coordinates{0.5, 0.5, 0},
                           sqrt(2));

    connManager.createInterior(nodes[0], nodes[1], nodes[3]);
    connManager.createInterior(nodes[0], nodes[3], nodes[2]);
  }

  static void generateSampleGraphWithDataWithConversionToUtm(
      Graph& graph, Map& map, const double west_border,
      const double north_border, const double east_border,
      const double south_border, bool version2D) {
    // temp storage for nodes we care about for this function
    vector<GNode> nodes;
    // wrapper around graph to edit it
    ConnectivityManager connManager{graph};

    Utils::convertToUtm(south_border, west_border,
                        map); // TODO: What is a point of this call?!

    // note the following coordinates should be the same ones used to load
    // terrain data into the map; the height (z coordinate) is retrieved from
    // said map

    // SW
    const Coordinates& coordinates0 =
        Coordinates{Utils::convertToUtm(south_border, west_border, map), map};
    // NW
    const Coordinates& coordinates1 =
        Coordinates{Utils::convertToUtm(north_border, west_border, map), map};
    // SE
    const Coordinates& coordinates2 =
        Coordinates{Utils::convertToUtm(south_border, east_border, map), map};
    // NE
    const Coordinates& coordinates3 =
        Coordinates{Utils::convertToUtm(north_border, east_border, map), map};

    // create the node points for the border intersections
    // NOT a hyperedge or a hanging node (because border points)
    nodes.push_back(
        connManager.createNode(NodeData{false, coordinates0, false}));
    nodes.push_back(
        connManager.createNode(NodeData{false, coordinates1, false}));
    nodes.push_back(
        connManager.createNode(NodeData{false, coordinates2, false}));
    nodes.push_back(
        connManager.createNode(NodeData{false, coordinates3, false}));
    galois::gInfo("Nodes created.");

    //nodes.push_back(connManager.createNode(NodeData{false,
    //Coordinates{west_border, south_border}, false}));
    //nodes.push_back(connManager.createNode(NodeData{false,
    //Coordinates{west_border, north_border}, false}));
    //nodes.push_back(connManager.createNode(NodeData{false,
    //Coordinates{east_border, south_border}, false}));
    //nodes.push_back(connManager.createNode(NodeData{false,
    //Coordinates{east_border, north_border}, false}));

    // 0 = SW, 1 = NW, 2 = SE, 3 = NE
    double leftBorderLength = nodes[0]->getData().getCoords().dist(
        nodes[1]->getData().getCoords(), version2D);
    double topBorderLength = nodes[1]->getData().getCoords().dist(
        nodes[3]->getData().getCoords(), version2D);
    double rightBorderLength = nodes[2]->getData().getCoords().dist(
        nodes[3]->getData().getCoords(), version2D);
    double bottomBorderLength = nodes[0]->getData().getCoords().dist(
        nodes[2]->getData().getCoords(), version2D);
    double SWtoNELength = nodes[3]->getData().getCoords().dist(
        nodes[0]->getData().getCoords(), version2D);

    // @todo can refactor some of the below to make less redundant

    // create 5 edges
    // left border creation
    connManager.createEdge(
        nodes[0], nodes[1], true,
        Coordinates{west_border, (north_border + south_border) / 2.,
                    map.get_height(west_border,
                                   (north_border + south_border) / 2., false)},
        leftBorderLength);
    // top border creation
    connManager.createEdge(
        nodes[1], nodes[3], true,
        Coordinates{(west_border + east_border) / 2., north_border,
                    map.get_height((west_border + east_border) / 2.,
                                   north_border, false)},
        topBorderLength);
    // right border creation
    connManager.createEdge(
        nodes[2], nodes[3], true,
        Coordinates{east_border, (north_border + south_border) / 2.,
                    map.get_height(east_border,
                                   (north_border + south_border) / 2., false)},
        rightBorderLength);
    // left border creation
    connManager.createEdge(
        nodes[0], nodes[2], true,
        Coordinates{(west_border + east_border) / 2., south_border,
                    map.get_height((west_border + east_border) / 2.,
                                   south_border, false)},
        bottomBorderLength);
    // this edge is diagonal from SW to NE with middle point being the center
    // of the map; this is the common edge of the first 2 triangles
    connManager.createEdge(
        nodes[3], nodes[0], false,
        Coordinates{(west_border + east_border) / 2.,
                    (north_border + south_border) / 2.,
                    map.get_height((west_border + east_border) / 2.,
                                   (north_border + south_border) / 2., false)},
        SWtoNELength);

    // creates the hyperedge for the 2 initial triangles of the graph
    connManager.createInterior(nodes[0], nodes[1], nodes[3]);
    connManager.createInterior(nodes[0], nodes[3], nodes[2]);
    galois::gInfo("Graph generated.");
  }

  static void generateSampleGraphWithData(Graph& graph, Map& map,
                                          const double west_border,
                                          const double north_border,
                                          const double east_border,
                                          const double south_border,
                                          bool version2D) {
    vector<GNode> nodes;
    ConnectivityManager connManager{graph};

    Utils::convertToUtm(south_border, west_border, map);

    nodes.push_back(connManager.createNode(
        NodeData{false, Coordinates{south_border, west_border, map}, false}));
    nodes.push_back(connManager.createNode(
        NodeData{false, Coordinates{north_border, west_border, map}, false}));
    nodes.push_back(connManager.createNode(
        NodeData{false, Coordinates{south_border, east_border, map}, false}));
    nodes.push_back(connManager.createNode(
        NodeData{false, Coordinates{north_border, east_border, map}, false}));

    //        nodes.push_back(connManager.createNode(NodeData{false,
    //        Coordinates{west_border, south_border}, false}));
    //        nodes.push_back(connManager.createNode(NodeData{false,
    //        Coordinates{west_border, north_border}, false}));
    //        nodes.push_back(connManager.createNode(NodeData{false,
    //        Coordinates{east_border, south_border}, false}));
    //        nodes.push_back(connManager.createNode(NodeData{false,
    //        Coordinates{east_border, north_border}, false}));

    double length1 = nodes[0]->getData().getCoords().dist(
        nodes[1]->getData().getCoords(), version2D);
    double length2 = nodes[1]->getData().getCoords().dist(
        nodes[3]->getData().getCoords(), version2D);
    double length3 = nodes[2]->getData().getCoords().dist(
        nodes[3]->getData().getCoords(), version2D);
    double length4 = nodes[0]->getData().getCoords().dist(
        nodes[2]->getData().getCoords(), version2D);
    double length5 = nodes[3]->getData().getCoords().dist(
        nodes[0]->getData().getCoords(), version2D);

    connManager.createEdge(
        nodes[0], nodes[1], true,
        Coordinates{
            west_border, (north_border + south_border) / 2.,
            map.get_height(west_border, (north_border + south_border) / 2.)},
        length1);
    connManager.createEdge(
        nodes[1], nodes[3], true,
        Coordinates{
            (west_border + east_border) / 2., north_border,
            map.get_height((west_border + east_border) / 2., north_border)},
        length2);
    connManager.createEdge(
        nodes[2], nodes[3], true,
        Coordinates{
            east_border, (north_border + south_border) / 2.,
            map.get_height(east_border, (north_border + south_border) / 2.)},
        length3);
    connManager.createEdge(
        nodes[0], nodes[2], true,
        Coordinates{
            (west_border + east_border) / 2., south_border,
            map.get_height((west_border + east_border) / 2., south_border)},
        length4);
    connManager.createEdge(
        nodes[3], nodes[0], false,
        Coordinates{(west_border + east_border) / 2.,
                    (north_border + south_border) / 2.,
                    map.get_height((west_border + east_border) / 2.,
                                   (north_border + south_border) / 2.)},
        length5);

    connManager.createInterior(nodes[0], nodes[1], nodes[3]);
    connManager.createInterior(nodes[0], nodes[3], nodes[2]);
  }

  static void generateSampleGraph3(Graph& graph) {
    GNode node1, node2, node3, node4, node5, node6, hEdge1, hEdge2, hEdge3,
        hEdge4;
    node1 = graph.createNode(NodeData{false, Coordinates{0, 0, 0}, false});
    node2 = graph.createNode(NodeData{false, Coordinates{0, 1, 0}, false});
    node3 = graph.createNode(NodeData{false, Coordinates{1, 0, 0}, false});
    node4 = graph.createNode(NodeData{false, Coordinates{1, 1, 0}, false});
    node5 = graph.createNode(NodeData{false, Coordinates{1.5, 0.5, 0}, false});
    node6 = graph.createNode(NodeData{false, Coordinates{0.5, -0.5, 0}, false});
    hEdge1 = graph.createNode(NodeData{true, true, Coordinates{0.3, 0.7, 0}});
    hEdge2 = graph.createNode(NodeData{true, true, Coordinates{0.7, 0.3, 0}});
    hEdge3 = graph.createNode(NodeData{true, true, Coordinates{1.17, 0.5, 0}});
    hEdge4 = graph.createNode(NodeData{true, true, Coordinates{0.5, -0.17, 0}});

    graph.addNode(node1);
    graph.addNode(node2);
    graph.addNode(node3);
    graph.addNode(node4);
    graph.addNode(node5);
    graph.addNode(node6);
    graph.addNode(hEdge1);
    graph.addNode(hEdge2);
    graph.addNode(hEdge3);
    graph.addNode(hEdge4);

    graph.addEdge(node1, node2);
    graph.getEdgeData(graph.findEdge(node1, node2)).setBorder(true);
    graph.getEdgeData(graph.findEdge(node1, node2))
        .setMiddlePoint(Coordinates{0, 0.5, 0});
    graph.getEdgeData(graph.findEdge(node1, node2)).setLength(1);
    graph.addEdge(node2, node4);
    graph.getEdgeData(graph.findEdge(node2, node4)).setBorder(true);
    graph.getEdgeData(graph.findEdge(node2, node4))
        .setMiddlePoint(Coordinates{0.5, 1, 0});
    graph.getEdgeData(graph.findEdge(node2, node4)).setLength(1);
    graph.addEdge(node3, node4);
    graph.getEdgeData(graph.findEdge(node3, node4)).setBorder(false);
    graph.getEdgeData(graph.findEdge(node3, node4))
        .setMiddlePoint(Coordinates{1, 0.5, 0});
    graph.getEdgeData(graph.findEdge(node3, node4)).setLength(1);
    graph.addEdge(node1, node3);
    graph.getEdgeData(graph.findEdge(node1, node3)).setBorder(false);
    graph.getEdgeData(graph.findEdge(node1, node3))
        .setMiddlePoint(Coordinates{0.5, 0, 0});
    graph.getEdgeData(graph.findEdge(node1, node3)).setLength(1);
    graph.addEdge(node4, node1);
    graph.getEdgeData(graph.findEdge(node4, node1)).setBorder(false);
    graph.getEdgeData(graph.findEdge(node4, node1))
        .setMiddlePoint(Coordinates{0.5, 0.5, 0});
    graph.getEdgeData(graph.findEdge(node4, node1)).setLength(sqrt(2));
    graph.addEdge(node5, node4);
    graph.getEdgeData(graph.findEdge(node5, node4)).setBorder(true);
    graph.getEdgeData(graph.findEdge(node5, node4))
        .setMiddlePoint(Coordinates{1.25, 0.75, 0});
    graph.getEdgeData(graph.findEdge(node5, node4)).setLength(sqrt(2) / 2.);
    graph.addEdge(node5, node3);
    graph.getEdgeData(graph.findEdge(node5, node3)).setBorder(true);
    graph.getEdgeData(graph.findEdge(node5, node3))
        .setMiddlePoint(Coordinates{1.25, 0.25, 0});
    graph.getEdgeData(graph.findEdge(node5, node3)).setLength(sqrt(2) / 2.);
    graph.addEdge(node6, node1);
    graph.getEdgeData(graph.findEdge(node6, node1)).setBorder(true);
    graph.getEdgeData(graph.findEdge(node6, node1))
        .setMiddlePoint(Coordinates{0.25, -0.25, 0});
    graph.getEdgeData(graph.findEdge(node6, node1)).setLength(sqrt(2) / 2.);
    graph.addEdge(node6, node3);
    graph.getEdgeData(graph.findEdge(node6, node3)).setBorder(true);
    graph.getEdgeData(graph.findEdge(node6, node3))
        .setMiddlePoint(Coordinates{0.75, -0.25, 0});
    graph.getEdgeData(graph.findEdge(node6, node3)).setLength(sqrt(2) / 2.);

    graph.addEdge(hEdge1, node1);
    graph.getEdgeData(graph.findEdge(hEdge1, node1)).setBorder(false);
    graph.addEdge(hEdge1, node2);
    graph.getEdgeData(graph.findEdge(hEdge1, node2)).setBorder(false);
    graph.addEdge(hEdge1, node4);
    graph.getEdgeData(graph.findEdge(hEdge1, node4)).setBorder(false);

    graph.addEdge(hEdge2, node1);
    graph.getEdgeData(graph.findEdge(hEdge2, node1)).setBorder(false);
    graph.addEdge(hEdge2, node4);
    graph.getEdgeData(graph.findEdge(hEdge2, node4)).setBorder(false);
    graph.addEdge(hEdge2, node3);
    graph.getEdgeData(graph.findEdge(hEdge2, node3)).setBorder(false);

    graph.addEdge(hEdge3, node5);
    graph.getEdgeData(graph.findEdge(hEdge3, node5)).setBorder(false);
    graph.addEdge(hEdge3, node4);
    graph.getEdgeData(graph.findEdge(hEdge3, node4)).setBorder(false);
    graph.addEdge(hEdge3, node3);
    graph.getEdgeData(graph.findEdge(hEdge3, node3)).setBorder(false);

    graph.addEdge(hEdge4, node1);
    graph.getEdgeData(graph.findEdge(hEdge4, node1)).setBorder(false);
    graph.addEdge(hEdge4, node3);
    graph.getEdgeData(graph.findEdge(hEdge4, node3)).setBorder(false);
    graph.addEdge(hEdge4, node6);
    graph.getEdgeData(graph.findEdge(hEdge4, node6)).setBorder(false);
  }

  static void generateSampleGraph2(Graph& graph) {
    GNode nodes[9];
    for (int i = 0; i < 9; ++i) {
      nodes[i] =
          graph.createNode(NodeData{false,
                                    Coordinates{static_cast<double>(i % 3),
                                                static_cast<double>(i / 3), 0},
                                    false});
      graph.addNode(nodes[i]);
    }
    GNode hEdges[8];
    for (auto& hEdge : hEdges) {
      hEdge = graph.createNode(NodeData{true, true});
      graph.addNode(hEdge);
    }

    graph.addEdge(nodes[0], nodes[1]);
    graph.getEdgeData(graph.findEdge(nodes[0], nodes[1])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[0], nodes[1]))
        .setMiddlePoint(Coordinates{0.5, 0, 0});
    graph.getEdgeData(graph.findEdge(nodes[0], nodes[1])).setLength(1);
    graph.addEdge(nodes[1], nodes[2]);
    graph.getEdgeData(graph.findEdge(nodes[1], nodes[2])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[1], nodes[2]))
        .setMiddlePoint(Coordinates{1.5, 0, 0});
    graph.getEdgeData(graph.findEdge(nodes[1], nodes[2])).setLength(1);
    graph.addEdge(nodes[0], nodes[3]);
    graph.getEdgeData(graph.findEdge(nodes[0], nodes[3])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[0], nodes[3]))
        .setMiddlePoint(Coordinates{0, 0.5, 0});
    graph.getEdgeData(graph.findEdge(nodes[0], nodes[3])).setLength(1);
    graph.addEdge(nodes[1], nodes[4]);
    graph.getEdgeData(graph.findEdge(nodes[1], nodes[4])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[1], nodes[4]))
        .setMiddlePoint(Coordinates{1, 0.5, 0});
    graph.getEdgeData(graph.findEdge(nodes[1], nodes[4])).setLength(1);
    graph.addEdge(nodes[2], nodes[5]);
    graph.getEdgeData(graph.findEdge(nodes[2], nodes[5])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[2], nodes[5]))
        .setMiddlePoint(Coordinates{2, 0.5, 0});
    graph.getEdgeData(graph.findEdge(nodes[2], nodes[5])).setLength(1);
    graph.addEdge(nodes[3], nodes[4]);
    graph.getEdgeData(graph.findEdge(nodes[3], nodes[4])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[3], nodes[4]))
        .setMiddlePoint(Coordinates{0.5, 1, 0});
    graph.getEdgeData(graph.findEdge(nodes[3], nodes[4])).setLength(1);
    graph.addEdge(nodes[4], nodes[5]);
    graph.getEdgeData(graph.findEdge(nodes[4], nodes[5])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[4], nodes[5]))
        .setMiddlePoint(Coordinates{1.5, 1, 0});
    graph.getEdgeData(graph.findEdge(nodes[4], nodes[5])).setLength(1);

    graph.addEdge(nodes[3], nodes[6]);
    graph.getEdgeData(graph.findEdge(nodes[3], nodes[6])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[3], nodes[6]))
        .setMiddlePoint(Coordinates{0, 1.5, 0});
    graph.getEdgeData(graph.findEdge(nodes[3], nodes[6])).setLength(1);
    graph.addEdge(nodes[4], nodes[7]);
    graph.getEdgeData(graph.findEdge(nodes[4], nodes[7])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[4], nodes[7]))
        .setMiddlePoint(Coordinates{1, 1.5, 0});
    graph.getEdgeData(graph.findEdge(nodes[4], nodes[7])).setLength(1);
    graph.addEdge(nodes[5], nodes[8]);
    graph.getEdgeData(graph.findEdge(nodes[5], nodes[8])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[5], nodes[8]))
        .setMiddlePoint(Coordinates{2, 1.5, 0});
    graph.getEdgeData(graph.findEdge(nodes[5], nodes[8])).setLength(1);
    graph.addEdge(nodes[6], nodes[7]);
    graph.getEdgeData(graph.findEdge(nodes[6], nodes[7])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[6], nodes[7]))
        .setMiddlePoint(Coordinates{0.5, 2, 0});
    graph.getEdgeData(graph.findEdge(nodes[6], nodes[7])).setLength(1);
    graph.addEdge(nodes[7], nodes[8]);
    graph.getEdgeData(graph.findEdge(nodes[7], nodes[8])).setBorder(true);
    graph.getEdgeData(graph.findEdge(nodes[7], nodes[8]))
        .setMiddlePoint(Coordinates{1.5, 2, 0});
    graph.getEdgeData(graph.findEdge(nodes[7], nodes[8])).setLength(1);

    for (int j = 0; j < 2; ++j) {
      graph.addEdge(nodes[j], nodes[j + 4]);
      graph.getEdgeData(graph.findEdge(nodes[j], nodes[j + 4])).setBorder(true);
      graph.getEdgeData(graph.findEdge(nodes[j], nodes[j + 4]))
          .setMiddlePoint(Coordinates{0.5 + j / 2, 0.5 + j % 2, 0});
      graph.getEdgeData(graph.findEdge(nodes[j], nodes[j + 4]))
          .setLength(sqrt(2));
      graph.addEdge(nodes[j + 3], nodes[j + 7]);
      graph.getEdgeData(graph.findEdge(nodes[j + 3], nodes[j + 7]))
          .setBorder(true);
      graph.getEdgeData(graph.findEdge(nodes[j + 3], nodes[j + 7]))
          .setMiddlePoint(Coordinates{0.5 + j / 2, 0.5 + j % 2, 0});
      graph.getEdgeData(graph.findEdge(nodes[j + 3], nodes[j + 7]))
          .setLength(sqrt(2));
    }

    for (int k = 0; k < 4; ++k) {
      graph.addEdge(hEdges[k], nodes[(k + k / 2) % 9]);
      graph.addEdge(hEdges[k], nodes[(k + 1 + k / 2) % 9]);
      graph.addEdge(hEdges[k], nodes[(k + 4 + k / 2) % 9]);

      graph.addEdge(hEdges[(k + 4) % 8], nodes[(k + k / 2) % 9]);
      graph.addEdge(hEdges[(k + 4) % 8], nodes[(k + 3 + k / 2) % 9]);
      graph.addEdge(hEdges[(k + 4) % 8], nodes[(k + 4 + k / 2) % 9]);
    }
  }
};

#endif // GALOIS_GRAPHGENERATOR_H
