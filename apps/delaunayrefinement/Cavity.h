/* 
 
   Lonestar DelaunayRefinement: Refinement of an initial, unrefined Delaunay
   mesh to eliminate triangles with angles < 30 degrees, using a
   variation of Chew's algorithm.
 
   Authors: Milind Kulkarni 
 
   Copyright (C) 2007, 2008 The University of Texas at Austin
 
   Licensed under the Eclipse Public License, Version 1.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
 
   http://www.eclipse.org/legal/epl-v10.html
 
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 
   File: Cavity.h
 
   Modified: February 12th, 2008 by Milind Kulkarni (initial version)
 
*/ 

#include <queue>
#include <set>

class Cavity {

  Tuple center;
  GNode centerNode;
  Element* centerElement;
  int dim;
  std::queue<GNode> frontier;
  // the cavity itself
  Subgraph pre;
  // what the new elements should look like
  Subgraph post;
  // the edge-relations that connect the boundary to the cavity
  std::set<Subgraph::tmpEdge> connections;

  Graph* graph;
	

  /**
   * find the node that is opposite the obtuse angle of the element
   */
  GNode getOpposite(GNode node) {
    int numOutNeighbors = graph->neighborsSize(node);
    if (numOutNeighbors != 3) {
      assert(0);
    }
    Element& element = node.getData();
    GNode dst;
    for (Graph::neighbor_iterator ii = graph->neighbor_begin(node), ee = graph->neighbor_end(node); ii != ee; ++ii) {
      GNode neighbor = *ii;
      Edge& edgeData = graph->getEdgeData(node, neighbor);
      Tuple elementTuple = element.getObtuse();
      if (elementTuple != edgeData.getPoint(0) && elementTuple != edgeData.getPoint(1)) {
	assert(dst.isNull());
	dst = neighbor;
      }
    }
    assert(!dst.isNull());
    return dst;
  }

  void expand(GNode node, GNode next) {
    Element& nextElement = next.getData();
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
	  frontier.push(next);
	}
      }
    } else {
      // not a member
      Edge& edgeData = graph->getEdgeData(node, next);
      Subgraph::tmpEdge edge(node, next, edgeData);
      if (!connections.count(edge)) {
	connections.insert(edge);
      }
    }
  }


 public:
  
 Cavity(Graph* g)
   :graph(g)
  {}
	
  void initialize(GNode node) {
    pre.reset();
    post.reset();
    connections.clear();
    frontier = std::queue<GNode>();
    centerNode = node;
    centerElement = &centerNode.getData();
    while (graph->containsNode(centerNode) && centerElement->isObtuse()) {
      centerNode = getOpposite(centerNode);
      centerElement = &centerNode.getData();
    }
    center = centerElement->getCenter();
    dim = centerElement->getDim();
    pre.addNode(centerNode);
    frontier.push(centerNode);
  }

  void build() {
    while (!frontier.empty()) {
      GNode curr = frontier.front();
      frontier.pop();
      for (Graph::neighbor_iterator ii = graph->neighbor_begin(curr), ee = graph->neighbor_end(curr); ii != ee; ++ii) {
	GNode neighbor = *ii;
	expand(curr, neighbor);
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
    for (std::set<Subgraph::tmpEdge>::iterator ii = connections.begin(), ee = connections.end(); ii != ee; ++ii) {
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
      Element& ne_nodeData = ne_connection.getData();
      const Edge& new_edge = new_element.getRelatedEdge(ne_nodeData);
      //boolean mod = 
      post.addEdge(Subgraph::tmpEdge(ne_node, ne_connection, new_edge));
      //assert mod;
      std::set<GNode>& postnodes = post.getNodes();
      for (std::set<GNode>::iterator ii = postnodes.begin(), ee = postnodes.end(); ii != ee; ++ii) {
        GNode node = *ii;
        Element& element = node.getData();
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

