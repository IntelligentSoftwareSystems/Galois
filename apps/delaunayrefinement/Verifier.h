/** Delaunay triangulation verifier -*- C++ -*-
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
 */

#ifndef VERIFIER_H
#define VERIFIER_H

//#include "Galois/Galois.h"
#include "Galois/ParallelSTL/ParallelSTL.h"

#include <stack>
#include <set>
#include <iostream>

class Verifier : public Galois::Runtime::Lockable {
  struct inconsistent: public std::unary_function<GNode,bool> {
    Graphp graph;
    inconsistent() { }
    inconsistent(Graphp g): graph(g) { }

    bool operator()(const GNode& node) const {
      Element& e = graph->getData(node);

      size_t dist = std::distance(graph->edge_begin(node), graph->edge_end(node));
      if (e.dim() == 2) {
        if (dist != 1) {
          std::cerr << "Error: Segment " << e << " has " << dist << " relation(s)\n";
          return true;
        }
      } else if (e.dim() == 3) {
        if (dist != 3) {
          std::cerr << "Error: Triangle " << e << " has " << dist << " relation(s)\n";
          return true;
        }
      } else {
        std::cerr << "Error: Element with " << e.dim() << " edges\n";
        return true;
      }
      return false;
    }
    // // serialization functions
    // typedef int tt_has_serialize;
    // void serialize(Galois::Runtime::SerializeBuffer& s) const {
    //   gSerialize(s,graph);
    // }
    // void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    //   gDeserialize(s,graph);
    // }
  };

  struct not_delaunay: public std::unary_function<GNode,bool>, public Galois::Runtime::Lockable {
    Graphp graph;
    not_delaunay() { }
    not_delaunay(Graphp g): graph(g) { }

    bool operator()(const GNode& node) {
      Element& e1 = graph->getData(node);

      for (Graph::edge_iterator jj = graph->edge_begin(node),
          ej = graph->edge_end(node); jj != ej; ++jj) {
        const GNode& n = graph->getEdgeDst(jj);
        Element& e2 = graph->getData(n);
        if (e1.dim() == 3 && e2.dim() == 3) {
          Tuple t2;
          if (!getTupleT2OfRelatedEdge(e1, e2, t2)) {
            std::cerr << "missing tuple\n";
            return true;
          }
          if (e1.inCircle(t2)) {
            std::cerr << "Delaunay property violated: point " << t2 << " in element " << e1 << "\n";
            return true;
          }
        }
      }
      return false;
    }

    bool getTupleT2OfRelatedEdge(const Element& e1, const Element& e2, Tuple& t) {
      int e2_0 = -1;
      int e2_1 = -1;
      int phase = 0;

      for (int i = 0; i < e1.dim(); i++) {
        for (int j = 0; j < e2.dim(); j++) {
          if (e1.getPoint(i) != e2.getPoint(j)) 
            continue;

          if (phase == 0) {
            e2_0 = j;
            phase = 1;
            break;
          } 

          e2_1 = j;
          for (int k = 0; k < 3; k++) {
            if (k != e2_0 && k != e2_1) {
              t = e2.getPoint(k);
              return true;
            }
          }
        }
      }
      return false;
    }
    // serialization functions
    typedef int tt_has_serialize;
    void serialize(Galois::Runtime::SerializeBuffer& s) const {
      gSerialize(s,graph);
    }
    void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
      gDeserialize(s,graph);
    }
  };

  bool checkReachability(Graphp graph) {
    std::stack<GNode> remaining;
    std::set<GNode> found;
    remaining.push(*(graph->begin()));

    while (!remaining.empty()) {
      GNode node = remaining.top();
      remaining.pop();
      if (!found.count(node)) {
        if (!graph->containsNode(node)) {
          std::cerr << "Reachable node was removed from graph\n";
        }
        found.insert(node);
        int i = 0;
        for (Graph::edge_iterator ii = graph->edge_begin(node),
            ei = graph->edge_end(node); ii != ei; ++ii) {
          GNode n = graph->getEdgeDst(ii);
          assert(i < 3);
          assert(graph->containsNode(n));
          assert(node != n);
          ++i;
          remaining.push(n);
        }
      }
    }

    auto size = Galois::ParallelSTL::count_if_local(graph, [&](GNode) { return true; });
    if (found.size() != size) {
      std::cerr << "Error: Not all elements are reachable. ";
      std::cerr << "Found: " << found.size() << " needed: " << size << ".\n";
      return false;
    }
    return true;
  }

public:
  bool verify(Graphp g) {
    return checkReachability(g) && !Galois::ParallelSTL::count_if_local(g, inconsistent(g))
      && !Galois::ParallelSTL::count_if_local(g, not_delaunay(g));
  }
};

#endif
