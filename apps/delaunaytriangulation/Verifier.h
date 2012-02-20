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

#include "Graph.h"
#include "Point.h"

#include <stack>
#include <set>
#include <iostream>

class Verifier {
  bool checkConsistency(Graph* graph) {
    bool error = false;
    for (Graph::active_iterator ii = graph->active_begin(),
        ei = graph->active_end(); ii != ei; ++ii) {
      const GNode& node = *ii;
      Element& e = node.getData(Galois::NONE);

      if (e.dim() == 2) {
        if (graph->neighborsSize(node, Galois::NONE) != 1) {
          std::cerr << "Error: Segment " << e << " has " 
            << graph->neighborsSize(node, Galois::NONE) << " relation(s)\n";
          error = true;
        }
      } else if (e.dim() == 3) {
        if (graph->neighborsSize(*ii, Galois::NONE) != 3) {
          std::cerr << "Error: Triangle " << e << " has "
            << graph->neighborsSize(node, Galois::NONE) << " relation(s)\n";
          error = true;
        }
      } else {
        std::cerr << "Error: Element with " << e.dim() << " edges\n";
        error = true;
      }
    }
    if (error)
      return false;
    return true;
  }

  bool checkReachability(Graph* graph) {
    std::stack<GNode> remaining;
    std::set<GNode> found;
    remaining.push(*(graph->active_begin()));

    while (!remaining.empty()) {
      GNode node = remaining.top();
      remaining.pop();
      if (!found.count(node)) {
        if (!graph->containsNode(node)) {
          std::cerr << "Reachable node was removed from graph\n";
        }
        found.insert(node);
        int i = 0;
        for (Graph::neighbor_iterator ii = graph->neighbor_begin(node, Galois::NONE),
            ei = graph->neighbor_end(node, Galois::NONE); ii != ei; ++ii) {
          assert(i < 3);
          assert(graph->containsNode(*ii));
          assert(node != *ii);
          ++i;
          remaining.push(*ii);
        }
      }
    }

    if (found.size() != graph->size()) {
      std::cerr << "Error: Not all elements are reachable. ";
      std::cerr << "Found: " << found.size() << " needed: " << graph->size() << ".\n";
      return false;
    }
    return true;
  }

  bool checkDelaunayProperty(Graph* graph) {
    for (Graph::active_iterator ii = graph->active_begin(),
        ei = graph->active_end(); ii != ei; ++ii) {
      const GNode& node = *ii;
      Element& e1 = node.getData(Galois::NONE);

      for (Graph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::NONE),
          ej = graph->neighbor_end(node, Galois::NONE); jj != ej; ++jj) {
        const GNode& neighborNode = *jj;
        Element& e2 = neighborNode.getData(Galois::NONE);
        if (e1.dim() == 3 && e2.dim() == 3) {
          Tuple t2;
          if (!getTupleT2OfRelatedEdge(e1, e2, t2)) {
            std::cerr << "missing tuple\n";
            return false;
          }
          if (e1.inCircle(t2)) {
            std::cerr << "Delaunay property violated: point " << t2 << " in element " << e1 << "\n";
            return false;
          }
        }
      }
    }
    return true;
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
            t = e2.getPoint(k)->t();
            return true;
          }
        }
      }
    }
    return false;
  }

public:
  bool verify(Graph* graph) {
    return checkConsistency(graph) && checkReachability(graph) && checkDelaunayProperty(graph);
  }
};

#endif
