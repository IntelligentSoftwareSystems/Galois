/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef CAVITY_H
#define CAVITY_H

#include "Graph.h"

#include <vector>

//! A cavity which will be retrangulated
template <typename Alloc = std::allocator<char>>
class Cavity : private boost::noncopyable {
  typedef typename Alloc::template rebind<GNode>::other GNodeVectorAlloc;
  typedef std::vector<GNode, GNodeVectorAlloc> GNodeVector;
  typedef typename Alloc::template rebind<std::pair<GNode, int>>::other
      GNodeIntPairVectorAlloc;
  typedef std::vector<std::pair<GNode, int>, GNodeIntPairVectorAlloc>
      GNodeIntPairVector;

  struct InCircumcenter {
    const Graph& graph;
    Tuple tuple;
    InCircumcenter(const Graph& g, const Tuple& t) : graph(g), tuple(t) {}
    bool operator()(const GNode& n) const {
      Element& e = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      return e.inCircle(tuple);
    }
  };

  Searcher<Alloc> searcher;
  GNodeVector newNodes;
  GNodeIntPairVector outside;
  GNode center;
  Point* point;
  Graph& graph;
  const Alloc& alloc;

  //! Find triangles that border cavity but are not in the cavity
  void findOutside() {
    for (const auto& ii : searcher.inside) {
      for (auto jj : graph.edges(ii, galois::MethodFlag::UNPROTECTED)) {
        GNode n = graph.getEdgeDst(jj);
        // i.e., if (!e.boundary() && e.inCircle(point->t()))
        if (std::find(searcher.matches.begin(), searcher.matches.end(), n) !=
            searcher.matches.end())
          continue;

        int index = graph.getEdgeData(
            graph.findEdge(n, ii, galois::MethodFlag::UNPROTECTED));
        outside.push_back(std::make_pair(n, index));

        Element& e = graph.getData(n, galois::MethodFlag::UNPROTECTED);
        Point* p2  = e.getPoint(index);
        Point* p3  = e.getPoint((index + 1) % 3);

        p2->get(galois::MethodFlag::WRITE);
        p3->get(galois::MethodFlag::WRITE);
      }
    }
  }

  void addElements() {
    GNodeVector newNodes(alloc);

    // Create new nodes
    for (auto& ii : outside) {
      const GNode& n = ii.first;
      int& index     = ii.second;

      Element& e = graph.getData(n, galois::MethodFlag::UNPROTECTED);

      Point* p2 = e.getPoint(index);
      Point* p3 = e.getPoint((index + 1) % 3);

      Element newE(point, p2, p3);
      GNode newNode = graph.createNode(newE);
      graph.addNode(newNode, galois::MethodFlag::UNPROTECTED);

      point->addElement(newNode);
      p2->addElement(newNode);
      p3->addElement(newNode);

      graph.getEdgeData(
          graph.addEdge(newNode, n, galois::MethodFlag::UNPROTECTED)) = 1;
      graph.getEdgeData(
          graph.addEdge(n, newNode, galois::MethodFlag::UNPROTECTED)) = index;

      newNodes.push_back(newNode);
    }

    // Update new node connectivity
    for (unsigned i = 0; i < newNodes.size(); ++i) {
      const GNode& n1   = newNodes[i];
      const Element& e1 = graph.getData(n1, galois::MethodFlag::UNPROTECTED);
      for (unsigned j = i + 1; j < newNodes.size(); ++j) {
        if (i != j) {
          const GNode& n2 = newNodes[j];
          const Element& e2 =
              graph.getData(n2, galois::MethodFlag::UNPROTECTED);

          for (int x = 2; x >= 1; --x) {
            for (int y = 2; y >= 1; --y) {
              if (e1.getPoint(x) == e2.getPoint(y)) {
                int indexForNewNode                           = x & 2;
                int indexForNode                              = y & 2;
                graph.getEdgeData(graph.addEdge(
                    n1, n2, galois::MethodFlag::UNPROTECTED)) = indexForNewNode;
                graph.getEdgeData(graph.addEdge(
                    n2, n1, galois::MethodFlag::UNPROTECTED)) = indexForNode;
              }
            }
          }
        }
      }
    }
  }

  void removeElements() {
    for (auto ii : searcher.matches) {
      graph.removeNode(ii, galois::MethodFlag::UNPROTECTED);
    }
  }

public:
  Cavity(Graph& g, const Alloc& a = Alloc())
      : searcher(g, a), newNodes(a), outside(a), graph(g), alloc(a) {}

  void init(const GNode& c, Point* p) {
    center = c;
    point  = p;
  }

  void build() {
    assert(graph.getData(center).inCircle(point->t()));
    searcher.findAll(center, InCircumcenter(graph, point->t()));
    assert(!searcher.inside.empty());
    findOutside();
  }

  void update() {
    removeElements();
    addElements();
  }
};

#endif
