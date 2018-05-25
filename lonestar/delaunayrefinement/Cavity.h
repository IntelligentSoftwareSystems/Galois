/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <vector>
#include <algorithm>

class Cavity {
  //! [STL vector using PerIterAllocTy]
  typedef std::vector<EdgeTuple,galois::PerIterAllocTy::rebind<EdgeTuple>::other> ConnTy;
  //! [STL vector using PerIterAllocTy]

  Tuple center;
  GNode centerNode;
  std::vector<GNode,galois::PerIterAllocTy::rebind<GNode>::other> frontier;
  // !the cavity itself
  PreGraph pre;
  // !what the new elements should look like
  PostGraph post;
  // the edge-relations that connect the boundary to the cavity
  ConnTy connections;
  Element* centerElement;
  Graph* graph;
  int dim;

  /**
   * find the node that is opposite the obtuse angle of the element
   */
  GNode getOpposite(GNode node) {
    assert(std::distance(graph->edge_begin(node), graph->edge_end(node)) == 3);
    Element& element = graph->getData(node, galois::MethodFlag::WRITE);
    Tuple elementTuple = element.getObtuse();
    Edge ObtuseEdge = element.getOppositeObtuse();
    for (Graph::edge_iterator ii = graph->edge_begin(node, galois::MethodFlag::WRITE),
        ee = graph->edge_end(node, galois::MethodFlag::WRITE); ii != ee; ++ii) {
      GNode neighbor = graph->getEdgeDst(ii);
      //Edge& edgeData = graph->getEdgeData(node, neighbor);
      Edge edgeData = element.getRelatedEdge(graph->getData(neighbor, galois::MethodFlag::WRITE));
      if (elementTuple != edgeData.getPoint(0) && elementTuple != edgeData.getPoint(1)) {
	return neighbor;
      }
    }
    GALOIS_DIE("unreachable");
    return node;
  }

  void expand(GNode node, GNode next) {
    Element& nextElement = graph->getData(next, galois::MethodFlag::WRITE);
    if ((!(dim == 2 && nextElement.dim() == 2 && next != centerNode))
        && nextElement.inCircle(center)) {
      // isMember says next is part of the cavity, and we're not the second
      // segment encroaching on this cavity
      if ((nextElement.dim() == 2) && (dim != 2)) {
	// is segment, and we are encroaching
	initialize(next);
	build();
      } else {
	if (!pre.containsNode(next)) {
	  pre.addNode(next);
	  frontier.push_back(next);
	}
      }
    } else {
      // not a member
      //Edge& edgeData = graph->getEdgeData(node, next);
      Edge edgeData = nextElement.getRelatedEdge(graph->getData(node, galois::MethodFlag::WRITE));
      EdgeTuple edge(node, next, edgeData);
      if (std::find(connections.begin(), connections.end(), edge) == connections.end()) {
	connections.push_back(edge);
      }
    }
  }

public:
  Cavity(Graph* g, galois::PerIterAllocTy& cnx)
    :frontier(cnx),
     pre(cnx),
     post(cnx),
     connections(cnx),
     graph(g)
  {}
  
  void initialize(GNode node) {
    pre.reset();
    post.reset();
    connections.clear();
    frontier.clear();
    centerNode = node;
    centerElement = &graph->getData(centerNode, galois::MethodFlag::WRITE);
    while (graph->containsNode(centerNode, galois::MethodFlag::WRITE) && centerElement->isObtuse()) {
      centerNode = getOpposite(centerNode);
      centerElement = &graph->getData(centerNode, galois::MethodFlag::WRITE);
    }
    center = centerElement->getCenter();
    dim = centerElement->dim();
    pre.addNode(centerNode);
    frontier.push_back(centerNode);
  }

  void build() {
    while (!frontier.empty()) {
      GNode curr = frontier.back();
      frontier.pop_back();
      for (Graph::edge_iterator ii = graph->edge_begin(curr, galois::MethodFlag::WRITE), 
	     ee = graph->edge_end(curr, galois::MethodFlag::WRITE); 
	   ii != ee; ++ii) {
	GNode neighbor = graph->getEdgeDst(ii);
	expand(curr, neighbor);
      }
    }
  }

  /**
   * Create the new cavity based on the data of the old one
   */
  void computePost() {
    if (centerElement->dim() == 2) { // we built around a segment
      GNode n1 = graph->createNode(Element(center, centerElement->getPoint(0)));
      GNode n2 = graph->createNode(Element(center, centerElement->getPoint(1)));

      post.addNode(n1);
      post.addNode(n2);
    }

    for (ConnTy::iterator ii = connections.begin(), ee = connections.end(); ii != ee; ++ii) {
      EdgeTuple tuple = *ii;
      Element newElement(center, tuple.data.getPoint(0), tuple.data.getPoint(1));
      GNode other = pre.containsNode(tuple.dst) ?  tuple.src : tuple.dst;
      Element& otherElement = graph->getData(other, galois::MethodFlag::WRITE);

      GNode newNode = graph->createNode(newElement); // XXX
      const Edge& otherEdge = newElement.getRelatedEdge(otherElement);
      post.addEdge(newNode, other, otherEdge);

      for (PostGraph::iterator ii = post.begin(), ee = post.end(); ii != ee; ++ii) {
        GNode node = *ii;
        Element& element = graph->getData(node, galois::MethodFlag::WRITE);
        if (element.isRelated(newElement)) {
          const Edge& edge = newElement.getRelatedEdge(element);
	  post.addEdge(newNode, node, edge);
        }
      }
      post.addNode(newNode);
    }
  }

  void update(GNode node, galois::UserContext<GNode>& ctx) {
    for (PreGraph::iterator ii = pre.begin(), ee = pre.end(); ii != ee; ++ii) 
      graph->removeNode(*ii, galois::MethodFlag::UNPROTECTED);
    
    //add new data
    for (PostGraph::iterator ii = post.begin(), ee = post.end(); ii != ee; ++ii) {
      GNode n = *ii;
      graph->addNode(n, galois::MethodFlag::UNPROTECTED);
      Element& element = graph->getData(n, galois::MethodFlag::UNPROTECTED);
      if (element.isBad()) {
        ctx.push(n);
      }
    }
    
    for (PostGraph::edge_iterator ii = post.edge_begin(), ee = post.edge_end(); ii != ee; ++ii) {
      EdgeTuple edge = *ii;
      graph->addEdge(edge.src, edge.dst, galois::MethodFlag::UNPROTECTED);
    }

    if (graph->containsNode(node, galois::MethodFlag::UNPROTECTED)) {
      ctx.push(node);
    }
  }
};
