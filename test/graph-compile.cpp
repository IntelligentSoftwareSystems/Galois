#include <iostream>

#include "galois/graphs/Graph.h"

struct NoDefault {
  int x;
  explicit NoDefault(int x): x(x) { }
private:
  NoDefault();
};

template<typename GraphTy>
void check() {
  typedef typename GraphTy::GraphNode GNode;
  int v;

  GraphTy g;
  GNode n1 = g.createNode(1);
  GNode n2 = g.createNode(2);
  GNode n3 = g.createNode(3);
  GNode n4 = g.createNode(4);
  GNode n5 = g.createNode(5);
  g.addNode(n1);
  g.addNode(n2);
  g.addNode(n3);
  g.addNode(n4);
  g.addNode(n5);
  g.addMultiEdge(n1, n2, galois::MethodFlag::WRITE, v);
  g.addMultiEdge(n5, n2, galois::MethodFlag::WRITE, v);
  g.addMultiEdge(n2, n3, galois::MethodFlag::WRITE, v);
  g.addMultiEdge(n2, n4, galois::MethodFlag::WRITE, v);
  for(auto ii : g.edges(n2))
    std::cout << "o " << g.getData(g.getEdgeDst(ii)).x << "\n";
  for(auto ii : g.in_edges(n2))
    std::cout << "i " << g.getData(g.getEdgeDst(ii)).x << "\n";
  std::cout << "** removing 2->3\n";
  g.removeEdge(n2, g.findEdge(n2,n3));
  for(auto ii : g.edges(n2))
    std::cout << "o " << g.getData(g.getEdgeDst(ii)).x << "\n";
  for(auto ii : g.in_edges(n2))
    std::cout << "i " << g.getData(g.getEdgeDst(ii)).x << "\n";
  std::cout << "** removing 5->1\n";
  g.removeEdge(n5, g.findEdge(n5,n2));
  for(auto ii : g.edges(n2))
    std::cout << "o " << g.getData(g.getEdgeDst(ii)).x << "\n";
  for(auto ii : g.in_edges(n2))
    std::cout << "i " << g.getData(g.getEdgeDst(ii)).x << "\n";
  std::cout << "\n\n";
}

int main() {
  check<galois::graphs::MorphGraph<NoDefault,NoDefault,true> >();
  check<galois::graphs::MorphGraph<NoDefault,NoDefault,false> >();
  check<galois::graphs::MorphGraph<NoDefault,NoDefault,true,true> >();
  check<galois::graphs::MorphGraph<NoDefault,NoDefault,false,true> >();

  return 0;
}
