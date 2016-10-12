#include "PythonGraph.h"
#include "Galois/Statistic.h"

#include <iostream>

Graph *createGraph() {
  Graph *g = new Graph();
  return g;
}

void deleteGraph(Graph *g) {
  delete g;
}

void printGraph(Graph* g) {
  for(auto n: *g) {
    std::cout << "node" << std::endl;
    for(auto i: g->getData(n)) {
      std::cout << "  " << i.first << ": " << i.second << std::endl;
    }
    for(auto e: g->edges(n)) {
      std::cout << "  edge" << std::endl;
      for(auto i: g->getEdgeData(e)) {
        std::cout << "    " << i.first << ": " << i.second << std::endl;
      }
    }
  }
}

GNode createNode(Graph *g) {
  return g->createNode();
}

void addNode(Graph *g, const GNode n) {
  g->addNode(n);
}

void addNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val) {
  g->getData(n)[key] = val;
}

void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key) {
  g->getData(n).erase(key);
}

Edge addEdge(Graph *g, GNode src, GNode dst) {
  auto ei = g->addEdge(src, dst, Galois::MethodFlag::WRITE);
  return {ei.base(), ei.end()};
}

void addEdgeAttr(Graph *g, Edge e, const KeyAltTy key, const ValAltTy val) {
  g->getEdgeData(edge_iterator(e.base, e.end))[key] = val;
}

void removeEdgeAttr(Graph *g, Edge e, const KeyAltTy key) {
  g->getEdgeData(edge_iterator(e.base, e.end)).erase(key);
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

