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

#include <set>

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

    bool operator<(const tmpEdge& rhs) const {
      if (src < rhs.src) return true;
      if (src > rhs.src) return false;
      if (dst < rhs.dst) return true;
      if (dst > rhs.dst) return false;
      return false;
    }
  };

 private:
  // the nodes in the graph before updating
  std::set<GNode> nodes;
  // the edges that connect the subgraph to the rest of the graph
  std::set<tmpEdge> edges;

 public:
  Subgraph() {}

  bool containsNode(GNode n) {
    return nodes.count(n);
  }

  bool addNode(GNode n) {
    return nodes.insert(n).second;
  }
 
  bool addEdge(tmpEdge e) {
    return edges.insert(e).second;
  }
  bool addEdge(GNode src, GNode dst, const Edge& e) {
    return edges.insert(tmpEdge(src,dst,e)).second;
  }

  std::set<GNode>& getNodes() {
    return nodes;
  }

  std::set<tmpEdge>& getEdges() {
    return edges;
  }

  void reset() {
    nodes.clear();
    edges.clear();
  }
};
