/* 
 
   Lonestar DelaunayRefinement: Refinement of an initial, unrefined Delaunay
   mesh to eliminate triangles with angles < 30 degrees, using a
   variation of Chew's algorithm.
 
   Authors: Milind Kulkarni, Andrew Lenharth

   Copyright (C) 2007, 2008, 2011 The University of Texas at Austin
 
   Licensed under the Eclipse Public License, Version 1.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
 
   http://www.eclipse.org/legal/epl-v10.html
 
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 
   File: DelaunayRefinement.cpp
*/ 

#include <iostream>
#include <vector>
#include <stack>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <cassert>

#include "Element.h"

#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* name = "Delaunay Mesh Refinement";
static const char* description = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/delaunayrefinement.html";
static const char* help = "<input file>";

typedef Galois::Graph::FirstGraph<Element,void,false>            Graph;
typedef Galois::Graph::FirstGraph<Element,void,false>::GraphNode GNode;

#include "Mesh.h"

/**
 *  A sub-graph of the mesh. Used to store information about the original 
 *  and updated cavity  
 */
class Subgraph {
 public:
  struct tmpEdge {
    GNode src;
    GNode dst;
    Edge data;

    tmpEdge(GNode s, GNode d, const Edge& _d)
    :src(s), dst(d), data(_d)
    {}

    bool operator==(const tmpEdge& rhs) const {
      return src == rhs.src && dst == rhs.dst && data == data;
    }
  };

 private:
  // the nodes in the graph before updating
  typedef std::vector<GNode,Galois::PerIterMem::ItAllocTy::rebind<GNode>::other> nodesTy;

  nodesTy nodes;

  // the edges that connect the subgraph to the rest of the graph
  typedef std::vector<tmpEdge,Galois::PerIterMem::ItAllocTy::rebind<tmpEdge>::other> edgesTy;

  edgesTy edges;

 public:
  explicit Subgraph(Galois::PerIterMem* cnx) 
  : nodes(cnx->PerIterationAllocator), 
    edges(cnx->PerIterationAllocator)
  {}

  bool containsNode(GNode N) {
    return std::find(nodes.begin(), nodes.end(), N) != nodes.end();
  }

  void addNode(GNode n) {
    return nodes.push_back(n);
  }
 
  void addEdge(tmpEdge e) {
    return edges.push_back(e);
  }
  void addEdge(GNode src, GNode dst, const Edge& e) {
    return edges.push_back(tmpEdge(src,dst,e));
  }

  void reset() {
    nodes.clear();
    edges.clear();
  }

  typedef nodesTy::iterator iterator;

  iterator begin() {
    return nodes.begin();
  }

  iterator end() {
    return nodes.end();
  }

  typedef edgesTy::iterator edge_iterator;

  edge_iterator edge_begin() {
    return edges.begin();
  }

  edge_iterator edge_end() {
    return edges.end();
  }
};

class Cavity {
  Tuple center;
  GNode centerNode;
  Element* centerElement;
  int dim;
  std::vector<GNode,Galois::PerIterMem::ItAllocTy::rebind<GNode>::other> frontier;
  // the cavity itself
  Subgraph pre;
  // what the new elements should look like
  Subgraph post;
  // the edge-relations that connect the boundary to the cavity
  typedef std::vector<Subgraph::tmpEdge,Galois::PerIterMem::ItAllocTy::rebind<Subgraph::tmpEdge>::other> connTy;
  connTy connections;

  Graph* graph;

  /**
   * find the node that is opposite the obtuse angle of the element
   */
  GNode getOpposite(GNode node) {
    int numOutNeighbors = graph->neighborsSize(node, Galois::Graph::ALL);
    if (numOutNeighbors != 3) {
      assert(0);
    }
    Element& element = node.getData(Galois::Graph::ALL);
    Tuple elementTuple = element.getObtuse();
    Edge ObtuseEdge = element.getOppositeObtuse();
    GNode dst;
    for (Graph::neighbor_iterator ii = graph->neighbor_begin(node,Galois::Graph::ALL), ee = graph->neighbor_end(node,Galois::Graph::ALL); ii != ee; ++ii) {
      GNode neighbor = *ii;
      //Edge& edgeData = graph->getEdgeData(node, neighbor);
      Edge edgeData = element.getRelatedEdge(neighbor.getData(Galois::Graph::ALL));
      if (elementTuple != edgeData.getPoint(0) && elementTuple != edgeData.getPoint(1)) {
	assert(dst.isNull());
	dst = neighbor;
      }
    }
    assert(!dst.isNull());
    return dst;
  }

  void expand(GNode node, GNode next) {
    Element& nextElement = next.getData(Galois::Graph::ALL);
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
      Edge edgeData = nextElement.getRelatedEdge(node.getData(Galois::Graph::ALL));
      Subgraph::tmpEdge edge(node, next, edgeData);
      if (std::find(connections.begin(), connections.end(), edge) == connections.end()) {
	connections.push_back(edge);
      }
    }
  }

public:
  Cavity(Graph* g, Galois::PerIterMem* cnx)
    :frontier(cnx->PerIterationAllocator),
     pre(cnx),
     post(cnx),
     connections(cnx->PerIterationAllocator),
     graph(g)
  {}
  
  void initialize(GNode node) {
    pre.reset();
    post.reset();
    connections.clear();
    frontier.clear();// = std::<GNode>();
    centerNode = node;
    centerElement = &centerNode.getData(Galois::Graph::ALL);
    while (graph->containsNode(centerNode) && centerElement->isObtuse()) {
      centerNode = getOpposite(centerNode);
      centerElement = &centerNode.getData(Galois::Graph::ALL);
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
      for (Graph::neighbor_iterator ii = graph->neighbor_begin(curr,Galois::Graph::ALL), 
	     ee = graph->neighbor_end(curr,Galois::Graph::ALL); 
	   ii != ee; ++ii) {
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
      Element& ne_nodeData = ne_connection.getData(Galois::Graph::ALL);
      const Edge& new_edge = new_element.getRelatedEdge(ne_nodeData);
      //boolean mod = 
      post.addEdge(Subgraph::tmpEdge(ne_node, ne_connection, new_edge));
      //assert mod;
      for (Subgraph::iterator ii = post.begin(), ee = post.end(); ii != ee; ++ii) {
        GNode node = *ii;
        Element& element = node.getData(Galois::Graph::ALL);
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
  }

  Subgraph& getPost() {
    return post;
  }
};

Graph* mesh;

struct process {
  template<typename Context>
  void operator()(GNode item, Context& lwl) {
    if (!mesh->containsNode(item))
      return;
    
    item.getData(Galois::Graph::ALL); //lock

    Cavity cav(mesh, lwl.getPerIterAlloc());
    cav.initialize(item);
    cav.build();
    cav.update();
    
    for (Subgraph::iterator ii = cav.getPre().begin(),
	   ee = cav.getPre().end(); ii != ee; ++ii) 
      mesh->removeNode(*ii, Galois::Graph::NONE);
    
    //add new data
    for (Subgraph::iterator ii = cav.getPost().begin(),
	   ee = cav.getPost().end(); ii != ee; ++ii) {
      GNode node = *ii;
      mesh->addNode(node, Galois::Graph::ALL);
      Element& element = node.getData(Galois::Graph::ALL);
      if (element.isBad()) {
        lwl.push(node);
      }
    }
    
    for (Subgraph::edge_iterator ii = cav.getPost().edge_begin(),
	   ee = cav.getPost().edge_end(); ii != ee; ++ii) {
      Subgraph::tmpEdge edge = *ii;
      //bool ret = 
      mesh->addEdge(edge.src, edge.dst, Galois::Graph::ALL); //, edge.data);
      //assert ret;
    }
    if (mesh->containsNode(item)) {
      lwl.push(item);
    }
  }
};

int main(int argc, const char** argv) {
  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() != 1) {
    std::cout << "incorrect number of arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  mesh = new Graph();
  Mesh m;
  m.read(mesh, args[0]);
  std::vector<GNode> wl;
  int numbad = m.getBad(mesh, wl);

  std::cout << "configuration: " << mesh->size() << " total triangles, " << numbad << " bad triangles\n";

  Galois::Launcher::startTiming();
  using namespace GaloisRuntime::WorkList;
  Galois::for_each<LocalQueues<ChunkedLIFO<1024>, LIFO<> > >(wl.begin(), wl.end(), process());
  Galois::Launcher::stopTiming();
  
  if (!skipVerify) {
    if (!m.verify(mesh)) {
      std::cerr << "Refinement failed.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    
    int size = m.getBad(mesh, wl);
    if (size != 0) {
      std::cerr << size << " bad triangles remaining.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    std::cout << "Refinement OK\n";
  }
  return 0;
}
