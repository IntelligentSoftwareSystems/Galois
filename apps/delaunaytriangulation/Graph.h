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

#include "Galois/Graphs/Graph2.h"

#include <boost/optional.hpp>
#include <vector>
#include <deque>

#ifdef GALOIS_DET
#include <set>
#endif

typedef Galois::Graph::FirstGraph<Element,int,true> Graph;
typedef Graph::GraphNode GNode;

//! Factor out common graph traversals
template<typename Alloc=std::allocator<char> >
struct Searcher: private boost::noncopyable {
  typedef Alloc allocator_type;
  typedef std::vector<GNode, Alloc> GNodeVector;

  Graph& graph;
  GNodeVector matches, inside;
  const allocator_type& alloc;
  ElementMark mark;

  Searcher(Graph& g, const Alloc& a = allocator_type()): 
    graph(g), matches(a), inside(a), alloc(a) { }

  void useMark(long id, unsigned numSearch, unsigned numTry) {
    assert(numSearch < 2);
    mark = ElementMark(id + 1, (numTry << 1) | numSearch);
  }

  struct DetLess: public std::binary_function<GNode,GNode,bool> {
    Graph& g;
    DetLess(Graph& x): g(x) { }
    bool operator()(GNode a, GNode b) const {
      Element& e1 = g.getData(a, Galois::NONE);
      Element& e2 = g.getData(b, Galois::NONE);
      
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
#ifdef GALOIS_DET
    std::sort(v.begin(), v.end(), DetLess(graph));
#else
    std::sort(v.begin(), v.end(), std::less<GNode>());
#endif
    typename GNodeVector::iterator end = std::unique(v.begin(), v.end());
    v.resize(end - v.begin());
  }

  template<typename Pred>
  void find_(const GNode& start, const Pred& pred, bool all) {
    typedef boost::optional<GNode> SomeGNode;
    typedef std::deque<std::pair<GNode,SomeGNode>, Alloc> WorklistTy;

    WorklistTy wl(alloc);
    wl.push_back(std::make_pair(start, SomeGNode()));

#ifdef GALOIS_DET
    std::set<GNode> seen;
#endif
    while (!wl.empty()) {
      GNode cur = wl.front().first;
      SomeGNode prev = wl.front().second;

      wl.pop_front();

      if (!graph.containsNode(cur, Galois::CHECK_CONFLICT))
        continue;

#ifdef GALOIS_DET
      if (seen.find(cur) != seen.end())
        continue;
#else
      if (graph.getData(cur, Galois::NONE).getMark() == mark)
        continue;
#endif

      if (!all) {
        // NB(ddn): Technically this makes DelaunayTriangulation.cpp::Process not cautious
#ifdef GALOIS_DET
        seen.insert(cur);
#else
        graph.getData(cur, Galois::NONE).getMark().update(mark); // XXX
#endif
      }

      bool matched = false;
      if (pred(cur)) {
        matched = true;
        matches.push_back(cur);
        if (all) {
          // NB(ddn): Ditto
#ifdef GALOIS_DET
          seen.insert(cur);
#else
          graph.getData(cur, Galois::CHECK_CONFLICT).getMark().update(mark);
#endif
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
        for (Graph::edge_iterator ii = graph.edge_begin(cur, Galois::CHECK_CONFLICT),
            ee = graph.edge_end(cur, Galois::CHECK_CONFLICT);
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
    assert(mark != ElementMark());
    find_(start, p, false);
  }
  
  //! Find all the elements matching pred (assuming monotonic predicate)
  template<typename Pred>
  void findAll(const GNode& start, const Pred& p) {
    assert(mark != ElementMark());
    find_(start, p, true);
    return;
  }
};

#endif
