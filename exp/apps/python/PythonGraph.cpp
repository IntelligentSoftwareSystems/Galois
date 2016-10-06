#include "PythonGraph.h"
#include "Galois/Statistic.h"

Graph *createGraph() {
  Graph *g = new Graph();
  return g;
}

void deleteGraph(Graph *g) {
  delete g;
}

GNode createNode(Graph *g) {
  return g->createNode();
}

void addNode(Graph *g, const GNode& n) {
  g->addNode(n);
}

void addNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val) {
  g->getData(n)[key] = val;
}

void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key) {
  g->getData(n).erase(key);
}

edge_iterator addEdge(Graph *g, GNode src, GNode dst) {
  return g->addEdge(src, dst);
}

edge_iterator addMultiEdge(Graph *g, GNode src, GNode dst) {
  return g->addMultiEdge(src, dst, Galois::MethodFlag::WRITE);
}

void addEdgeAttr(Graph *g, edge_iterator e, const KeyAltTy key, const ValAltTy val) {
  g->getEdgeData(e)[key] = val;
}

void removeEdgeAttr(Graph *g, edge_iterator e, const KeyAltTy key) {
  g->getEdgeData(e).erase(key);
}

struct BFS {
  Graph& g;
  BFS(Graph& g): g(g) {}

  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {
    auto newDist = std::stoi(g.getData(n)["dist"]) + 1;
    auto newDistStr = std::to_string(newDist);
    for(auto e: g.edges(n)) {
      auto dst = g.getEdgeDst(e);
      auto& dstDistStr = g.getData(dst)["dist"];
      if(dstDistStr == "INFINITY" || std::stoi(dstDistStr) > newDist) {
        dstDistStr = newDistStr;
        ctx.push(dst);
      }
    }
  }
};

void analyzeBFS(Graph *g, int numThreads) {
  Galois::StatManager statManager;
  Galois::setActiveThreads(numThreads < 1 ? 1 : numThreads);

  Galois::StatTimer T;
  T.start();
  Galois::do_all_local(*g, [=] (GNode n) {(*g).getData(n)["dist"] = "INFINITY";});
  auto src = *(g->begin());
  g->getData(src)["dist"] = "0";
  Galois::for_each(src, BFS{*g});
  T.stop();
}

