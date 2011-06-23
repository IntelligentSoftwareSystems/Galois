// Subgraph -*- C++ -*-

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
 
 File: Subgraph.h
 
 Modified: February 12th, 2008 by Milind Kulkarni (initial version)
 
 */ 

#include <vector>
#include <algorithm>

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
  typedef std::vector<GNode,Galois::PerIterAllocTy::rebind<GNode>::other> nodesTy;

  nodesTy nodes;

  // the edges that connect the subgraph to the rest of the graph

  typedef std::vector<tmpEdge,Galois::PerIterAllocTy::rebind<tmpEdge>::other> edgesTy;

  edgesTy edges;

 public:
  explicit Subgraph(Galois::PerIterAllocTy& cnx) 
  : nodes(cnx), 
    edges(cnx)
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
