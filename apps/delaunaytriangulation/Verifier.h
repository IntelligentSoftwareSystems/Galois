/*
 * Verifier.h
 *
 *  Created on: May 11, 2011
 *      Author: xinsui
 */

#ifndef VERIFIER_H_
#define VERIFIER_H_
#include "Element.h"
#include "Tuple.h"
#include <stack>
#include <set>

class Verifier {
public:
  Verifier() { }
  bool verify(Graph* graph) {
    return checkConsistency(graph) && checkReachability(graph) && checkDelaunayProperty(graph);
  }

private:
  bool checkConsistency(Graph* graph){
    bool error = false;
    for (Graph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
      GNode node = *ii;

      Element& element = node.getData(Galois::Graph::NONE);

      if (element.getDim() == 2) {
        if (graph->neighborsSize(node, Galois::Graph::NONE) != 1) {
          std::cerr << "-> Segment " << element << " has " << graph->neighborsSize(node, Galois::Graph::NONE) << " relation(s)\n";
          error = true;
        }
      } else if (element.getDim() == 3) {
        if (graph->neighborsSize(node, Galois::Graph::NONE) != 3) {
          std::cerr << "-> Triangle " << element << " has " << graph->neighborsSize(node, Galois::Graph::NONE) << " relation(s)";
          error = true;
        }
      } else {
        std::cerr << "-> Figures with " << element.getDim() << " edges";
        error = true;
      }
    }
    if (error)
      return false;
    return true;
  }

  bool checkReachability(Graph* graph){
    std::stack<GNode> remaining;
    std::set<GNode> found;
    remaining.push(*(graph->active_begin()));

    while (!remaining.empty()) {
      GNode node = remaining.top();
      remaining.pop();
      if (!found.count(node)) {
        assert(graph->containsNode(node) && "Reachable node was removed from graph");
        found.insert(node);
        int i = 0;
        for (Graph::neighbor_iterator ii = graph->neighbor_begin(node, Galois::Graph::NONE), ee = graph->neighbor_end(node, Galois::Graph::NONE); ii != ee; ++ii) {
          assert(i < 3);
          assert(graph->containsNode(*ii));
          assert(node != *ii);
          ++i;
          //    if (!found.count(*ii))
          remaining.push(*ii);
        }
      }
    }

    if (found.size() != graph->size()) {
      std::cerr << "Not all elements are reachable. ";
      std::cerr << "Found: " << found.size() << " needed: " << graph->size() << ".\n";
      assert(0 && "Not all elements are reachable");
      return false;
    }
    return true;
  }

  bool checkDelaunayProperty(Graph* graph){
    for (Graph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
      GNode node = *ii;
      Element& e = node.getData(Galois::Graph::NONE);
      for (Graph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE), eejj = graph->neighbor_end(node, Galois::Graph::NONE); jj != eejj; ++jj) {
        GNode neighborNode = *jj;
        Element& e2 = neighborNode.getData(Galois::Graph::NONE);
        if (e.getDim() == 3 && e2.getDim() == 3) {
          const Tuple* t2 = getTupleT2OfRelatedEdge(e, e2);
          if (!t2) {
            std::cerr << "missing tuple\n";
            return false;
          }
          if (e.inCircle(*t2)) {
            std::cerr << "delaunay property violated: " 
              "point " << *t2 << " in element " << e << "\n";
            return false;
          }
        }
      }
    }
    return true;
  }

  const Tuple* getTupleT2OfRelatedEdge(Element& e1, Element& e2) {
    int e2_0 = -1;
    int e2_1 = -1;
    int phase = 0;

    for (int i = 0; i < e1.getDim(); i++) {
      for (int j = 0; j < e2.getDim(); j++) {
        if (e1.getPoint(i) == e2.getPoint(j)) {
          if (phase == 0) {
            e2_0 = j;
            phase = 1;
            break;
          } else {
            e2_1 = j;
            for (int k = 0; k < 3; k++) {
              if (k != e2_0 && k != e2_1) {
                 return &(e2.getPoint(k));
              }
            }
          }
        }
      }
    }
    return NULL;
  }
};

#endif /* VERIFIER_H_ */
