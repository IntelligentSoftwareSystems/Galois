/** -*- C++ -*-
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
 * @author Xin Sui <xinsui@cs.utexas.edu>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GRAPH_H
#define GRAPH_H

#include "Element.h"

#include "galois/optional.h"
#include "galois/graphs/Graph.h"

#include <vector>
#include <deque>

typedef galois::graphs::FirstGraph<Element,char,true> Graph;
typedef Graph::GraphNode GNode;

//! Factor out common graph traversals
template<typename Alloc=std::allocator<char> >
struct Searcher: private boost::noncopyable {
  typedef Alloc allocator_type;
  typedef typename Alloc::template rebind<GNode>::other GNodeVectorAlloc;
  typedef std::vector<GNode, GNodeVectorAlloc> GNodeVector;

  struct Marker {
    GNodeVector seen;
    Marker(Graph&, const Alloc& a): seen(a) { }
    void mark(GNode n) { 
      seen.push_back(n);
    }
    bool hasMark(GNode n) { 
      return std::find(seen.begin(), seen.end(), n) != seen.end();
    }
  };

  Graph& graph;
  GNodeVector matches, inside;
  const allocator_type& alloc;

  Searcher(Graph& g, const Alloc& a = allocator_type()): 
    graph(g), matches(a), inside(a), alloc(a) { }

  struct DetLess: public std::binary_function<GNode,GNode,bool> {
    Graph& g;
    DetLess(Graph& x): g(x) { }
    bool operator()(GNode a, GNode b) const {
      Element& e1 = g.getData(a, galois::MethodFlag::UNPROTECTED);
      Element& e2 = g.getData(b, galois::MethodFlag::UNPROTECTED);
      
      for (int i = 0; i < 3; ++i) {
        uintptr_t v1 = (i < 2 || !e1.boundary()) ? reinterpret_cast<uintptr_t>(e1.getPoint(i)) : 0;
        uintptr_t v2 = (i < 2 || !e2.boundary()) ? reinterpret_cast<uintptr_t>(e2.getPoint(i)) : 0;
        if (v1 < v2)
          return true;
        else if (v1 > v2)
          return false;
      }
      return false;
    }
  };
  
  void removeDupes(GNodeVector& v) {
    std::sort(v.begin(), v.end(), DetLess(graph));
    typename GNodeVector::iterator end = std::unique(v.begin(), v.end());
    v.resize(end - v.begin());
  }

  template<typename Pred>
  void find_(const GNode& start, const Pred& pred, bool all) {
    typedef galois::optional<GNode> SomeGNode;
    typedef typename Alloc::template rebind<std::pair<GNode,SomeGNode>>::other WorklistAlloc;
    typedef std::deque<std::pair<GNode,SomeGNode>, WorklistAlloc> Worklist;

    Worklist wl(alloc);
    wl.push_back(std::make_pair(start, SomeGNode()));

    Marker marker(graph, alloc);
    while (!wl.empty()) {
      GNode cur = wl.front().first;
      SomeGNode prev = wl.front().second;

      wl.pop_front();

      if (!graph.containsNode(cur, galois::MethodFlag::WRITE))
        continue;

      if (marker.hasMark(cur))
        continue;

      // NB(ddn): Technically this makes DelaunayTriangulation.cpp::Process not cautious
      if (!all)
        marker.mark(cur);

      bool matched = false;
      if (pred(cur)) {
        matched = true;
        matches.push_back(cur);
        if (all) {
          marker.mark(cur);
        }
        else
          break; // Found it
      } else {
        if (all && prev)
          inside.push_back(*prev);
      }

      // Search neighbors (a) when matched and looking for all or (b) when no match and looking
      // for first
      if (matched == all) {
        for (Graph::edge_iterator ii = graph.edge_begin(cur, galois::MethodFlag::WRITE),
            ee = graph.edge_end(cur, galois::MethodFlag::WRITE);
            ii != ee; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          wl.push_back(std::make_pair(dst, SomeGNode(cur)));
        }
      }
    }

    if (all) {
      removeDupes(matches);
      removeDupes(inside);
    }
  }

  //! Find the first occurance of element matching pred
  template<typename Pred>
  void findFirst(const GNode& start, const Pred& p) {
    find_(start, p, false);
  }
  
  //! Find all the elements matching pred (assuming monotonic predicate)
  template<typename Pred>
  void findAll(const GNode& start, const Pred& p) {
    find_(start, p, true);
    return;
  }
};

#endif
