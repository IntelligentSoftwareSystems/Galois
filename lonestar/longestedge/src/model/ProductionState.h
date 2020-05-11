#ifndef GALOIS_PRODUCTIONSTATE_H
#define GALOIS_PRODUCTIONSTATE_H

#include "Graph.h"
#include "../utils/ConnectivityManager.h"

using std::vector;

/**
 * Wraps a hyperedge representing a triangle and provides methods to make
 * working with it easier.
 */
class ProductionState {
private:

  //! hyperedge ID
  GNode& interior;
  //! hyperedge data
  NodeData& interiorData;
  //! node data of the triangle the hyperedge connects
  vector<NodeData> verticesData;
  //! vertices connected by the hyper edge (i.e. a triangle)
  const vector<GNode> vertices;
  //! Edges of the triangle represented by the hyperedge
  const vector<optional<EdgeIterator>> edgesIterators;
  //! Edge data of all the triangle edges
  vector<galois::optional<EdgeData>> edgesData;
  //! lenghts of the edges of the triangle
  vector<double> lengths;
  //! edges indices that are supposed to exist via the hyperedge but do not
  //! (e.g., removed by another production)
  vector<int> brokenEdges;
  //! Indicator of what version is being used
  bool version2D;
  //! function to get the height of a point in the terrain
  std::function<double(double, double)> zGetter;

public:
  /**
   * Initialize state needed for a production given a hyperedge connecting
   * a triangle
   *
   * There's an assumption from getNeighbours that the edges will be in
   * 0->1, 1->2, 2->0 order that getTriangleEdges from connection manager relies on.
   */
  ProductionState(ConnectivityManager& connManager, GNode& interior,
                  bool version2D, std::function<double(double, double)> zGetter)
      : interior(interior), interiorData(interior->getData()),
        vertices(connManager.getNeighbours(interior)),
        edgesIterators(connManager.getTriangleEdges(vertices)),
        version2D(version2D), zGetter(zGetter) {
    Graph& graph = connManager.getGraph();

    // loop over 3 nodes/edges of triangle (if they exist)
    for (int i = 0; i < 3; ++i) {
      auto maybeEdgeIter = edgesIterators[i];
      edgesData.push_back(
          maybeEdgeIter ? graph.getEdgeData(maybeEdgeIter.get())
                        : galois::optional<EdgeData>()); // TODO: Look for
                                                         // possible optimization
      lengths.push_back(maybeEdgeIter ? edgesData[i].get().getLength() : -1);
      verticesData.push_back(
          graph.getData(vertices[i])); // TODO: Look for possible optimization

      // if an edge doesn't exist, push to broken edges
      if (!maybeEdgeIter) {
        brokenEdges.push_back(i);
      }
    }
  }

  //! find the longest edges (includes ties)
  std::vector<int> getLongestEdges() const {
    std::vector<int> longestEdges;
    for (int i = 0; i < 3; ++i) {
      if (!less(lengths[i], lengths[(i + 1) % 3]) &&
          !less(lengths[i], lengths[(i + 2) % 3])) {
        longestEdges.push_back(i);
      }
    }
    return longestEdges;
  }

  int getAnyBrokenEdge() const {
    if (!brokenEdges.empty()) {
      return brokenEdges[0];
    } else {
      return -1;
    }
  }

  //! Look at all edges, return the indcies of the ones with max distance
  //! among them.
  //! ASSUMPTION: 0->1, 1->2, 2->0 order of edges
  std::vector<int> getLongestEdgesIncludingBrokenOnes() const {
    std::vector<double> verticesDistances(3);
    for (int i = 0; i < 3; ++i) {
      verticesDistances[i] = verticesData[i].getCoords().dist(
          verticesData[(i + 1) % 3].getCoords(), version2D);
    }
    return indexesOfMaxElems(verticesDistances);
  }

  GNode& getInterior() const { return interior; }

  NodeData& getInteriorData() const { return interiorData; }

  const vector<galois::optional<EdgeData>>& getEdgesData() const {
    return edgesData;
  }

  const vector<double>& getLengths() const { return lengths; }

  const vector<NodeData>& getVerticesData() const { return verticesData; }

  const vector<GNode>& getVertices() const { return vertices; }

  const vector<optional<EdgeIterator>>& getEdgesIterators() const {
    return edgesIterators;
  }

  const vector<int>& getBrokenEdges() const { return brokenEdges; }

  bool isVersion2D() const { return version2D; }

  const std::function<double(double, double)>& getZGetter() const {
    return zGetter;
  }
};

#endif // GALOIS_PRODUCTIONSTATE_H
