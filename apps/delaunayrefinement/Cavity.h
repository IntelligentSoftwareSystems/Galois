/** Delaunay refinement -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * @author Milind Kulkarni <milind@purdue.edu>>
 */
#include <vector>
#include <algorithm>

class Cavity {
  Tuple center;
  GNode centerNode;
  Element* centerElement;
  int dim;
  std::vector<GNode,Galois::PerIterAllocTy::rebind<GNode>::other> frontier;
  // the cavity itself
  Subgraph pre;
  // what the new elements should look like
  Subgraph post;
  // the edge-relations that connect the boundary to the cavity
  typedef std::vector<Subgraph::tmpEdge,Galois::PerIterAllocTy::rebind<Subgraph::tmpEdge>::other> connTy;
  connTy connections;

  Graph* graph;

  /**
   * find the node that is opposite the obtuse angle of the element
   */
  GNode getOpposite(GNode node) {
    assert(std::distance(graph->edge_begin(node), graph->edge_end(node)) == 3);
    Element& element = graph->getData(node,Galois::ALL);
    Tuple elementTuple = element.getObtuse();
    Edge ObtuseEdge = element.getOppositeObtuse();
    bool found = false;
    GNode dst;
    for (Graph::edge_iterator ii = graph->edge_begin(node,Galois::ALL), ee = graph->edge_end(node,Galois::ALL); ii != ee; ++ii) {
      GNode neighbor = graph->getEdgeDst(ii);
      //Edge& edgeData = graph->getEdgeData(node, neighbor);
      Edge edgeData = element.getRelatedEdge(graph->getData(neighbor,Galois::ALL));
      if (elementTuple != edgeData.getPoint(0) && elementTuple != edgeData.getPoint(1)) {
	assert(!found);
	dst = neighbor;
        found = true;
      }
    }
    assert(found);
    return dst;
  }

  void expand(GNode node, GNode next) {
    Element& nextElement = graph->getData(next,Galois::ALL);
    if ((!(dim == 2 && nextElement.getDim() == 2 && next != centerNode)) && nextElement.inCircle(center)) {
      // isMember says next is part of the cavity, and we're not the second
      // segment encroaching on this cavity
      if ((nextElement.getDim() == 2) && (dim != 2)) {
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
      Edge edgeData = nextElement.getRelatedEdge(graph->getData(node,Galois::ALL));
      Subgraph::tmpEdge edge(node, next, edgeData);
      if (std::find(connections.begin(), connections.end(), edge) == connections.end()) {
	connections.push_back(edge);
      }
    }
  }


public:
  
  Cavity(Graph* g, Galois::PerIterAllocTy& cnx)
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
    frontier.clear();// = std::<GNode>();
    centerNode = node;
    centerElement = &graph->getData(centerNode,Galois::ALL);
    while (graph->containsNode(centerNode) && centerElement->isObtuse()) {
      centerNode = getOpposite(centerNode);
      centerElement = &graph->getData(centerNode,Galois::ALL);
    }
    center = centerElement->getCenter();
    dim = centerElement->getDim();
    pre.addNode(centerNode);
    frontier.push_back(centerNode);
  }

  void build() {
    while (!frontier.empty()) {
      GNode curr = frontier.back();
      frontier.pop_back();
      for (Graph::edge_iterator ii = graph->edge_begin(curr,Galois::ALL), 
	     ee = graph->edge_end(curr,Galois::ALL); 
	   ii != ee; ++ii) {
	GNode neighbor = graph->getEdgeDst(ii);
	expand(curr, neighbor); //VTune: Lots of work
      }
    }
  }

  /**
   * Create the new cavity based on the data of the old one
   */
  void update() {
    if (centerElement->getDim() == 2) { // we built around a segment
      Element ele1(center, centerElement->getPoint(0));
      GNode node1 = graph->createNode(ele1);
      post.addNode(node1);
      Element ele2(center, centerElement->getPoint(1));
      GNode node2 = graph->createNode(ele2);
      post.addNode(node2);
    }
    for (connTy::iterator ii = connections.begin(), ee = connections.end(); ii != ee; ++ii) {
      Subgraph::tmpEdge conn = *ii;
      Edge& edge = conn.data;
      Element new_element(center, edge.getPoint(0), edge.getPoint(1));
      GNode ne_node = graph->createNode(new_element);
      GNode ne_connection;
      if (pre.containsNode(conn.dst)) {
        ne_connection = conn.src;
      } else {
        ne_connection = conn.dst;
      }
      Element& ne_nodeData = graph->getData(ne_connection, Galois::ALL);
      const Edge& new_edge = new_element.getRelatedEdge(ne_nodeData);
      //boolean mod = 
      post.addEdge(Subgraph::tmpEdge(ne_node, ne_connection, new_edge));
      //assert mod;
      for (Subgraph::iterator ii = post.begin(), ee = post.end(); ii != ee; ++ii) {
        GNode node = *ii;
        Element& element = graph->getData(node, Galois::ALL);
        if (element.isRelated(new_element)) {
          const Edge& ele_edge = new_element.getRelatedEdge(element);
          //mod = 
	  post.addEdge(Subgraph::tmpEdge(ne_node, node, ele_edge));
          //assert mod;
        }
      }
      post.addNode(ne_node);
    }
  }
  
  Subgraph& getPre() {
    return pre;
  };
  Subgraph& getPost() {
    return post;
  };

  bool isMember(Element * n);
  
};

