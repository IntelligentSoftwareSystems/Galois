#include "Galois/Graph/Graph.h"

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
  GNode n1 = g.createNode(v);
  GNode n2 = g.createNode(v);
  g.addNode(n1);
  g.addNode(n2);
  g.addMultiEdge(n1, n2, Galois::MethodFlag::ALL, v);
}

int main() {
  check<Galois::Graph::FirstGraph<NoDefault,NoDefault,true> >();
  check<Galois::Graph::FirstGraph<NoDefault,NoDefault,false> >();

  return 0;
}
