#include "PythonGraph.h"
#include "Galois/Statistic.h"

#include <iostream>

Graph *createGraph() {
  Graph *g = new Graph();
//  std::cout << "create graph at " << g << std::endl;
  return g;
}

void deleteGraph(Graph *g) {
//  std::cout << "delete graph at " << g << std::endl;
  delete g;
}

void printGraph(Graph* g) {
//  std::cout << "print graph at " << g << std::endl;
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
  GNode n = g->createNode();
//  std::cout << "create node at " << n << " for g at " << g << std::endl;
  return n;
}

void addNode(Graph *g, const GNode n) {
//  std::cout << "add node at " << n << " into g at " << g << std::endl;
  g->addNode(n);
}

void addNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val) {
  g->getData(n)[key] = val;
}

void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key) {
  g->getData(n).erase(key);
}

void addMultiEdge(Graph *g, GNode src, GNode dst, const ValAltTy id) {
//  std::cout << "add multiedge " << id << " from " << src << " to " << dst << " into g at " << g << std::endl;
  auto e = g->addMultiEdge(src, dst, Galois::MethodFlag::WRITE);
  g->getEdgeData(e)["galois_id"] = id;
}

void addEdgeAttr(Graph *g, GNode src, GNode dst, const ValAltTy id, const KeyAltTy key, const ValAltTy val) {
  if(std::string(key) == std::string("galois_id")) {
    return;
  }

  for(auto e: g->edges(src)) {
    if(g->getEdgeDst(e) == dst && g->getEdgeData(e)["galois_id"] == id) {
      g->getEdgeData(e)[key] = val;
      break;
    }
  }
}

void removeEdgeAttr(Graph *g, GNode src, GNode dst, const ValAltTy id, const KeyAltTy key) {
  if(std::string("galois_id") == std::string(key)) {
    return;
  }

  for(auto e: g->edges(src)) {
    if(g->getEdgeDst(e) == dst && g->getEdgeData(e)["galois_id"] == id) {
      g->getEdgeData(e).erase(key);
      break;
    }
  }
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

