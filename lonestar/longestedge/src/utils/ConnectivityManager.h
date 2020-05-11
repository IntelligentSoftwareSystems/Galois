#ifndef GALOIS_CONNECTIVITYMANAGER_H
#define GALOIS_CONNECTIVITYMANAGER_H

#include <galois/optional.h>
#include "../model/Graph.h"

class ConnectivityManager {
private:
  Graph& graph;
public:
  ConnectivityManager(Graph& graph) : graph(graph) {}


  //! Return a vector of neighbors given some vertex
  std::vector<GNode> getNeighbours(GNode node) const {
    std::vector<GNode> vertices;
    for (Graph::edge_iterator ii = graph.edge_begin(node),
                              ee = graph.edge_end(node);
         ii != ee; ++ii) {
      vertices.push_back(graph.getEdgeDst(ii));
    }
    return vertices;
  }

  //! Given 3 nodes that comprise a triangle, return the triangle's edges
  //! Key is that edges will be returned such that it is 0->1, 1->2,
  //! and 2->0
  //! Assumption for this to work is that the triangle order is 0->1,
  //! 1->2, and 2->0 else you will get an empty edge
  std::vector<optional<EdgeIterator>>
  getTriangleEdges(std::vector<GNode> vertices) {
    std::vector<optional<EdgeIterator>> edges;
    for (int i = 0; i < 3; i++) {
      edges.emplace_back(getEdge(vertices[i], vertices[(i + 1) % 3]));
    }
    return edges;
  }

  //! Return an edge (if it exists; may have been broken into 2)
  optional<EdgeIterator> getEdge(const GNode& v1, const GNode& v2) const {
    EdgeIterator edge = graph.findEdge(v1, v2);
    return convertToOptionalEdge(edge);
  }

  //! See if an edge exists and return optional if necessary
  optional<EdgeIterator> convertToOptionalEdge(const EdgeIterator& edge) const {
    if (edge.base() == edge.end()) {
      return galois::optional<EdgeIterator>();
    } else {
      return galois::optional<EdgeIterator>(edge);
    }
  }

  //! True if there's a broken edge in the vector of edges
  bool hasBrokenEdge(const std::vector<optional<EdgeIterator>>& edges) const {
    return countBrokenEdges(edges) > 0;
  }

  //! Count the number of edges that don't exist (i.e. broken) in a vector
  //! of edges
  int countBrokenEdges(const std::vector<optional<EdgeIterator>>& edges) const {
    int counter = 0;
    for (const optional<EdgeIterator>& edge : edges) {
      if (!edge) {
        counter++;
      }
    }
    return counter;
  }

  //! Attempts to find a node between two nodes (i.e. find midpoint, if it
  //! exists)
  optional<GNode> findNodeBetween(const GNode& node1,
                                  const GNode& node2) const {
    Coordinates expectedLocation =
        (node1->getData().getCoords() + node2->getData().getCoords()) / 2.;
    std::vector<GNode> neighbours1 = getNeighbours(node1);
    std::vector<GNode> neighbours2 = getNeighbours(node2);
    for (GNode& iNode : neighbours1) {
      auto iNodeData = graph.getData(iNode);
      for (GNode& jNode : neighbours2) {
        if (iNode == jNode &&
            iNodeData.getCoords().isXYequal(expectedLocation)) {
          return optional<GNode>(iNode);
        }
      }
    }
    return optional<GNode>();
  }

  //! Creates a node and adds to specified worklist; returns it as well
  GNode createNode(NodeData& nodeData, galois::UserContext<GNode>& ctx) const {
    GNode node = createNode(nodeData);
    ctx.push(node);
    return std::move(node);
  }

  //! Adds a new node to the graph; returns node id
  GNode createNode(NodeData nodeData) const {
    galois::gInfo("Node creation...");
    auto node = graph.createNode(nodeData);
    galois::gInfo("Node created.");
    graph.addNode(node);
    galois::gInfo("Node added.");
    return node;
  }

  /**
   * Create a new edge; need to specify if border, the middle point, and
   * its length
   *
   * NOTE: can theoretically calculate middle + length given just the
   * two nodes
   */
  void createEdge(GNode& node1, GNode& node2, bool border,
                  const Coordinates& middlePoint, double length) {
    // add the edge
    graph.addEdge(node1, node2);
    // get the edge
    const EdgeIterator& edge = graph.findEdge(node1, node2);
    // edit its edge data
    graph.getEdgeData(edge).setBorder(border);
    graph.getEdgeData(edge).setMiddlePoint(middlePoint);
    graph.getEdgeData(edge).setLength(length);
  }

  /**
   * Connects 3 nodes with a hyperedge; should be a triangle.
   *
   * Adds the new node to a worklist as well.
   */
  void createInterior(const GNode& node1, const GNode& node2,
                      const GNode& node3,
                      galois::UserContext<GNode>& ctx) const {
    // args: is a hyper edge + do not need to refine
    NodeData interiorData = NodeData{true, false};
    auto interior         = createNode(interiorData, ctx);

    // connect hyperedge to triangle
    graph.addEdge(interior, node1);
    graph.addEdge(interior, node2);
    graph.addEdge(interior, node3);
    // located in center of triangle
    interior->getData().setCoords((node1->getData().getCoords() +
                                   node2->getData().getCoords() +
                                   node3->getData().getCoords()) /
                                  3.);
  }

  /**
   * Connects 3 nodes with a hyperedge; should be a triangle. Returns
   * the new node ID.
   *
   * For consistency, node1->node2->node3 edge order is probably
   * preferred.
   */
  GNode createInterior(const GNode& node1, const GNode& node2,
                       const GNode& node3) const {
    // args: is a hyper edge + do not need to refine
    NodeData interiorData = NodeData{true, false};
    auto interior         = createNode(interiorData);

    // connect hyperedge to triangle
    graph.addEdge(interior, node1);
    graph.addEdge(interior, node2);
    graph.addEdge(interior, node3);
    // located in center of triangle
    interior->getData().setCoords((node1->getData().getCoords() +
                                   node2->getData().getCoords() +
                                   node3->getData().getCoords()) /
                                  3.);
    return std::move(interior);
  }

  //! Return reference underlying graph
  Graph& getGraph() const { return graph; }

  //! Get the coordinates of all neighbors of specified vertex and return them.
  const std::vector<Coordinates> getVerticesCoords(const GNode& node) const {
    std::vector<Coordinates> result;
    for (auto neighbour : getNeighbours(node)) {
      result.push_back(neighbour->getData().getCoords());
    }
    return result;
  }

  //! Remove edge node1->node2 or node2->node1 (whichever is found)
  void removeEdge(const GNode& node1, const GNode& node2) const {
    const EdgeIterator& edge1 = graph.findEdge(node1, node2);
    if (edge1.base() != edge1.end()) {
      graph.removeEdge(node1, edge1);
      return;
    }
    const EdgeIterator& edge2 = graph.findEdge(node2, node1);
    if (edge2.base() != edge2.end()) {
      graph.removeEdge(node2, edge2);
      return;
    }
    std::cerr << "Problem in removing an edge." << std::endl;
  }
};

#endif // GALOIS_CONNECTIVITYMANAGER_H
