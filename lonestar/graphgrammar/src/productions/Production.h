#ifndef GALOIS_PRODUCTION_H
#define GALOIS_PRODUCTION_H

#include "../model/ProductionState.h"

class Production {

public:
  //! constructor needs a connection manager wrapping the graph
  explicit Production(const ConnectivityManager& connManager)
      : connManager(connManager) {}

  virtual bool execute(ProductionState& pState,
                       galois::UserContext<GNode>& ctx) = 0;

protected:
  ConnectivityManager connManager;

  bool checkIfBrokenEdgeIsTheLongest(
      int brokenEdge, const std::vector<optional<EdgeIterator>>& edgesIterators,
      const std::vector<GNode>& vertices,
      const std::vector<NodeData>& verticesData) const {
    std::vector<double> lengths(4);
    Graph& graph = connManager.getGraph();
    for (int i = 0, j = 0; i < 3; ++i) {
      if (i != brokenEdge) {
        lengths[j++] = graph.getEdgeData(edgesIterators[i].get()).getLength();
      } else {
        const std::pair<int, int>& brokenEdgeVertices =
            getEdgeVertices(brokenEdge);
        GNode& hangingNode =
            connManager
                .findNodeBetween(vertices[brokenEdgeVertices.first],
                                 vertices[brokenEdgeVertices.second])
                .get();
        lengths[2] = graph
                         .getEdgeData(graph.findEdge(
                             vertices[brokenEdgeVertices.first], hangingNode))
                         .getLength();
        lengths[3] = graph
                         .getEdgeData(graph.findEdge(
                             vertices[brokenEdgeVertices.second], hangingNode))
                         .getLength();
      }
    }
    return !less(lengths[2] + lengths[3], lengths[0]) &&
           !less(lengths[2] + lengths[3], lengths[1]);
  }


  //! vertices of an edge
  //! assumption: edges connect vertices that have adjacent IDs
  std::pair<int, int> getEdgeVertices(int edge) const {
    return std::pair<int, int>{edge, (edge + 1) % 3};
  }

  //! get the vertex not connected to the edge
  //! assumption: edges are 0->1, 1->2, 2->0
  int getNeutralVertex(int edgeToBreak) const {
    return (edgeToBreak + 2) % 3;
  }

  void breakElementWithHangingNode(int edgeToBreak, ProductionState& pState,
                                   galois::UserContext<GNode>& ctx) const {
    GNode hangingNode = getHangingNode(edgeToBreak, pState);

    breakElementUsingNode(edgeToBreak, hangingNode, pState, ctx);

    hangingNode->getData().setHanging(false);
  }

  //! Break an edge that doesn't have a hanging node already on it
  void breakElementWithoutHangingNode(int edgeToBreak, ProductionState& pState,
                                      galois::UserContext<GNode>& ctx) const {
    // create new node + its edges
    GNode newNode = createNodeOnEdge(edgeToBreak, pState, ctx);
    // create the hyperedges that result from the break
    breakElementUsingNode(edgeToBreak, newNode, pState, ctx);
  }

  //! logging/debug function; print info to cout
  static void logg(const NodeData& interiorData,
                   const std::vector<NodeData>& verticesData) {
    std::cout << "interior: (" << interiorData.getCoords().toString()
              << "), neighbours: (";
    for (auto vertex : verticesData) {
      std::cout << vertex.getCoords().toString() + ", ";
    }
    std::cout << ") ";
  }

private:

  //! Creates a new node on an edge + its endpoints; does not
  //! create corresponding hyperedges
  GNode createNodeOnEdge(int edgeToBreak, ProductionState& pState,
                         galois::UserContext<GNode>& ctx) const {
    Graph& graph = connManager.getGraph();
    // edges of triangle
    const vector<galois::optional<EdgeData>>& edgesData = pState.getEdgesData();
    bool breakingOnBorder = edgesData[edgeToBreak].get().isBorder();
    int neutralVertex     = getNeutralVertex(edgeToBreak);

    //        const EdgeIterator &edge =
    //        pState.getEdgesIterators()[edgeToBreak].get();
    //        graph.removeEdge(*(graph.getEdgeData(edge).getSrc()), edge);

    //        auto edgePair =
    //        connManager.findSrc(pState.getEdgesIterators()[edgeToBreak].get());
    //        auto edgePair = connManager.findSrc(edgesData[edgeToBreak].get());
    //        graph.removeEdge(edgePair.first, edgePair.second);

    // remove original edge from graph
    const std::pair<int, int>& edgeVertices = getEdgeVertices(edgeToBreak);
    connManager.removeEdge(pState.getVertices()[edgeVertices.first],
                           pState.getVertices()[edgeVertices.second]);

    // new point is midway point; height comes from terrain 
    const Coordinates& newPointCoords = getNewPointCoords(
        pState.getVerticesData()[edgeVertices.first].getCoords(),
        pState.getVerticesData()[edgeVertices.second].getCoords(),
        pState.getZGetter());


    // create the new node, push to graph and worklist
    // note: border nodes are never hanging; hanging means it needs to be
    // broken on the other end
    NodeData newNodeData = NodeData{false, newPointCoords, !breakingOnBorder};
    GNode newNode        = graph.createNode(newNodeData);
    graph.addNode(newNode);
    ctx.push(newNode);

    // connect vertices in original triangle to new node
    for (int i = 0; i < 3; ++i) {
      auto vertexData = pState.getVerticesData()[i];
      // addition of the new edge
      auto edge       = graph.addEdge(newNode, pState.getVertices()[i]);

      graph.getEdgeData(edge).setBorder(i != neutralVertex ? breakingOnBorder
                                                           : false);
      // midpoint
      graph.getEdgeData(edge).setMiddlePoint(
          (newNodeData.getCoords().getX() + vertexData.getCoords().getX()) / 2.,
          (newNodeData.getCoords().getY() + vertexData.getCoords().getY()) / 2.,
          (newNodeData.getCoords().getZ() + vertexData.getCoords().getZ()) /
              2.);
      // distance
      graph.getEdgeData(edge).setLength(newNodeData.getCoords().dist(
          vertexData.getCoords(), pState.isVersion2D()));
    }
    return newNode;
  }

  //! Given a hanging node, create the hyperedges for the 2 resulting triangles
  void breakElementUsingNode(int edgeToBreak, GNode const& hangingNode,
                             const ProductionState& pState,
                             galois::UserContext<GNode>& ctx) const {
    const std::pair<int, int>& brokenEdgeVertices =
        getEdgeVertices(edgeToBreak);
    Graph& graph       = connManager.getGraph();
    int neutralVertex  = getNeutralVertex(edgeToBreak);
    // newly added hangingnode
    NodeData hNodeData = hangingNode->getData();
    double length      = 0;
    length             = hNodeData.getCoords().dist(
        pState.getVerticesData()[neutralVertex].getCoords(),
        pState.isVersion2D());
    // add edge between hanging node and node that it doesn't connect in
    // triangle
    // TODO might this already done in create node on edge?
    addEdge(graph, hangingNode, pState.getVertices()[neutralVertex], false,
            length,
            (hNodeData.getCoords() +
             pState.getVerticesData()[neutralVertex].getCoords()) /
                2);

    // create the 2 hyperedges that results from the two triangles
    connManager.createInterior(hangingNode, pState.getVertices()[neutralVertex],
                               pState.getVertices()[brokenEdgeVertices.first],
                               ctx);
    connManager.createInterior(hangingNode, pState.getVertices()[neutralVertex],
                               pState.getVertices()[brokenEdgeVertices.second],
                               ctx);

    // remove original hyperedge
    graph.removeNode(pState.getInterior());
  }

  //! Get the hanging node on a broken edge (i.e. midpoint typically)
  GNode getHangingNode(int edgeToBreak, const ProductionState& pState) const {
    const std::pair<int, int>& brokenEdgeVertices =
        getEdgeVertices(edgeToBreak);
    return connManager
        .findNodeBetween(pState.getVertices()[brokenEdgeVertices.first],
                         pState.getVertices()[brokenEdgeVertices.second])
        .get();
  }

  //! Adds an edge to the graph given all neccessary parameters
  void addEdge(Graph& graph, GNode const& node1, GNode const& node2,
               bool border, double length,
               const Coordinates& middlePoint) const {
    const EdgeIterator& newEdge = graph.addEdge(node1, node2);
    graph.getEdgeData(newEdge).setBorder(border);
    graph.getEdgeData(newEdge).setLength(length);
    graph.getEdgeData(newEdge).setMiddlePoint(middlePoint);
  }

  //! Find the halfway point of 2 coordinates + get its height using
  //! the provided zgetter function
  Coordinates getNewPointCoords(
      const Coordinates& coords1, const Coordinates& coords2,
      const std::function<double(double, double)>& zGetter) const {
    double x = (coords1.getX() + coords2.getX()) / 2.;
    double y = (coords1.getY() + coords2.getY()) / 2.;
    return {x, y, zGetter(x, y)};
  }
};

#endif // GALOIS_PRODUCTION_H
