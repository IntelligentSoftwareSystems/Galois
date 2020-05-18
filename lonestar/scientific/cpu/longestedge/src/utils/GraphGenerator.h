#ifndef GALOIS_GRAPHGENERATOR_H
#define GALOIS_GRAPHGENERATOR_H

#include "Utils.h"

class GraphGenerator {
public:
  static void generateSampleGraphWithDataWithConversionToUtm(
      Graph& graph, Map& map, const double west_border,
      const double north_border, const double east_border,
      const double south_border, bool version2D, bool square) {
    // temp storage for nodes we care about for this function
    vector<GNode> nodes;
    // wrapper around graph to edit it
    ConnectivityManager connManager{graph};

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

    std::vector<Coordinates> coords;
    if (!square) {
      coords.push_back(coordinates0);
      coords.push_back(coordinates1);
      coords.push_back(coordinates2);
      coords.push_back(coordinates3);
    } else {
      double north = std::min(coordinates1.getY(), coordinates3.getY());
      double south = std::max(coordinates0.getY(), coordinates2.getY());
      double east  = std::min(coordinates2.getX(), coordinates3.getX());
      double west  = std::max(coordinates0.getX(), coordinates1.getX());
      double diff  = std::min(fabs(north - south), fabs(east - west));
      north        = south + diff;
      east         = west + diff;
      coords.emplace_back(west, south, map);
      coords.emplace_back(west, north, map);
      coords.emplace_back(east, south, map);
      coords.emplace_back(east, north, map);
    }

    // create the node points for the border intersections
    // NOT a hyperedge or a hanging node (because border points)
    nodes.push_back(connManager.createNode(NodeData{false, coords[0], false}));
    nodes.push_back(connManager.createNode(NodeData{false, coords[1], false}));
    nodes.push_back(connManager.createNode(NodeData{false, coords[2], false}));
    nodes.push_back(connManager.createNode(NodeData{false, coords[3], false}));
    galois::gInfo("Nodes created.");

    // nodes.push_back(connManager.createNode(NodeData{false,
    // Coordinates{west_border, south_border}, false}));
    // nodes.push_back(connManager.createNode(NodeData{false,
    // Coordinates{west_border, north_border}, false}));
    // nodes.push_back(connManager.createNode(NodeData{false,
    // Coordinates{east_border, south_border}, false}));
    // nodes.push_back(connManager.createNode(NodeData{false,
    // Coordinates{east_border, north_border}, false}));

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
};

#endif // GALOIS_GRAPHGENERATOR_H
