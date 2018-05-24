#ifndef VERIFIER_H
#define VERIFIER_H

#include "galois/Galois.h"
#include "galois/ParallelSTL.h"

#include <stack>
#include <set>
#include <iostream>

class Verifier {
  struct inconsistent: public std::unary_function<GNode,bool> {
    Graph& graph;
    inconsistent(Graph& g): graph(g) { }

    bool operator()(const GNode& node) const {
      Element& e = graph.getData(node);

      size_t dist = std::distance(graph.edge_begin(node), graph.edge_end(node));
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
  };

  struct not_delaunay: public std::unary_function<GNode,bool> {
    Graph& graph;
    not_delaunay(Graph& g): graph(g) { }

    bool operator()(const GNode& node) {
      Element& e1 = graph.getData(node);

      for (Graph::edge_iterator jj = graph.edge_begin(node),
          ej = graph.edge_end(node); jj != ej; ++jj) {
        const GNode& n = graph.getEdgeDst(jj);
        Element& e2 = graph.getData(n);
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
  };

  bool checkReachability(Graph& graph) {
    std::stack<GNode> remaining;
    std::set<GNode> found;
    remaining.push(*(graph.begin()));

    while (!remaining.empty()) {
      GNode node = remaining.top();
      remaining.pop();
      if (!found.count(node)) {
        if (!graph.containsNode(node)) {
          std::cerr << "Reachable node was removed from graph\n";
        }
        found.insert(node);
        int i = 0;
        for (Graph::edge_iterator ii = graph.edge_begin(node),
            ei = graph.edge_end(node); ii != ei; ++ii) {
          GNode n = graph.getEdgeDst(ii);
          assert(i < 3);
          assert(graph.containsNode(n));
          assert(node != n);
          ++i;
          remaining.push(n);
        }
      }
    }

    if (found.size() != graph.size()) {
      std::cerr << "Error: Not all elements are reachable. ";
      std::cerr << "Found: " << found.size() << " needed: " << graph.size() << ".\n";
      return false;
    }
    return true;
  }

public:
  bool verify(Graph& g) {
    return galois::ParallelSTL::find_if(g.begin(), g.end(), inconsistent(g)) == g.end()
      && galois::ParallelSTL::find_if(g.begin(), g.end(), not_delaunay(g)) == g.end()
      && checkReachability(g);
  }
};

#endif
