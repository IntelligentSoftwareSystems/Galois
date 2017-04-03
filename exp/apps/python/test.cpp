#include "PythonGraph.h"
#include "AnalyzeBFS.h"
#include "SearchSubgraph.h"
#include "PageRank.h"
#include "Filter.h"
#include "Reachability.h"
#include "Coarsen.h"

#include <unordered_map>
#include <iostream>

struct GraphWrapper {
  Graph *g;
  std::string name;
  std::unordered_map<std::string, GNode> nodes;
  std::unordered_map<std::string, Edge> edges;
  std::unordered_map<GNode, std::string> inv_nodes;
};


GraphWrapper testGraphConstruction() {
  GraphWrapper gw;
  gw.g = createGraph();
  gw.name = "g";

  auto n = createNode(gw.g);
  addNode(gw.g, n);
  gw.nodes["n0"] = n;
  gw.inv_nodes[n] = "n0";
  setNodeAttr(gw.g, n, "color", "red");
  setNodeAttr(gw.g, n, "id", "node 0");

  n = createNode(gw.g);
  addNode(gw.g, n);
  gw.nodes["n1"] = n;
  gw.inv_nodes[n] = "n1";
  setNodeAttr(gw.g, n, "language", "english");
  setNodeAttr(gw.g, n, "garbage", "to_be_deleted");
  setNodeAttr(gw.g, n, "id", "node 1");

  n = createNode(gw.g);
  addNode(gw.g, n);
  gw.nodes["n2"] = n;
  gw.inv_nodes[n] = "n2";
  setNodeAttr(gw.g, n, "date", "Oct. 24, 2016");
  setNodeAttr(gw.g, n, "id", "node 2");
  assert("node 2" == getNodeAttr(gw.g, gw.nodes["n2"], "id"));

  // remove node attributes
  removeNodeAttr(gw.g, gw.nodes["n1"], "garbage");
  Attr& attrN1 = (gw.g)->getData(gw.nodes["n1"]).attr;
  assert(attrN1.find("garbage") == attrN1.end());

  // getNodeAllAttr
  AttrList attrL = getNodeAllAttr(gw.g, gw.nodes["n1"]);
  assert(attrN1.size() == attrL.num);
  for(auto i = 0; i < attrL.num; i++) {
    assert(attrN1.find(attrL.key[i]) != attrN1.end());
    assert(attrN1[attrL.key[i]] == attrL.value[i]);
  }
  deleteAttrList(attrL);

  // add edges and set edge attributes
  auto e = addEdge(gw.g, gw.nodes["n0"], gw.nodes["n1"]);
  gw.edges["e0"] = e;
  setEdgeAttr(gw.g, e, "weight", "3.0");
  setEdgeAttr(gw.g, e, "id", "edge 0: 0 -> 1");
  assert("edge 0: 0 -> 1" == getEdgeAttr(gw.g, gw.edges["e0"], "id"));

  // merge edges
  e = addEdge(gw.g, gw.nodes["n0"], gw.nodes["n1"]);
  gw.edges["e1"] = e;
  setEdgeAttr(gw.g, e, "place", "texas");
  setEdgeAttr(gw.g, e, "garbage", "discard");
  setEdgeAttr(gw.g, e, "id", "edge 1: 0 -> 1");

  // remove edge attributes
  removeEdgeAttr(gw.g, e, "garbage");
  removeEdgeAttr(gw.g, e, "galois_id");
  Attr& attrE1 = (gw.g)->getEdgeData((gw.g)->findEdge(gw.edges["e1"].src, gw.edges["e1"].dst));
  assert(attrE1.find("garbage") == attrE1.end());
  assert(attrE1.find("galois_id") == attrE1.end());

  // getEdgeAllAttr
  attrL = getEdgeAllAttr(gw.g, gw.edges["e1"]);
  assert(attrE1.size() == attrL.num);
  for(auto i = 0; i < attrL.num; i++) {
    assert(attrE1.find(attrL.key[i]) != attrE1.end());
    assert(attrE1[attrL.key[i]] == attrL.value[i]);
  }
  deleteAttrList(attrL);
 
  // add a self loop
  e = addEdge(gw.g, gw.nodes["n0"], gw.nodes["n0"]);
  gw.edges["e2"] = e;
  setEdgeAttr(gw.g, e, "id", "edge 2: 0 -> 0");

  // add a length-2 loop
  e = addEdge(gw.g, gw.nodes["n1"], gw.nodes["n0"]);
  gw.edges["e3"] = e;
  setEdgeAttr(gw.g, e, "id", "edge 3: 1 -> 0");

  // add a length-3 loop
  e = addEdge(gw.g, gw.nodes["n1"], gw.nodes["n2"]);
  gw.edges["e4"] = e;
  setEdgeAttr(gw.g, e, "id", "edge 4: 1 -> 2");
  e = addEdge(gw.g, gw.nodes["n2"], gw.nodes["n0"]);
  gw.edges["e5"] = e;
  setEdgeAttr(gw.g, e, "id", "edge 5: 2 -> 0");

  // find outgoing edges
  assert((gw.g)->findEdge(gw.edges["e1"].src, gw.edges["e1"].dst) != (gw.g)->edge_end(gw.edges["e1"].src));
  assert((gw.g)->findEdge(gw.edges["e2"].src, gw.edges["e2"].dst) != (gw.g)->edge_end(gw.edges["e2"].src));
  assert((gw.g)->findEdge(gw.edges["e3"].src, gw.edges["e3"].dst) != (gw.g)->edge_end(gw.edges["e3"].src));
  assert((gw.g)->findEdge(gw.edges["e4"].src, gw.edges["e4"].dst) != (gw.g)->edge_end(gw.edges["e4"].src));
  assert((gw.g)->findEdge(gw.edges["e5"].src, gw.edges["e5"].dst) != (gw.g)->edge_end(gw.edges["e5"].src));
  if (DIRECTED) {
    // non-existing edge
    assert((gw.g)->findEdge(gw.nodes["n2"], gw.nodes["n1"]) == (gw.g)->edge_end(gw.nodes["n2"]));
  } else {
    // implied by edge 4
    assert((gw.g)->findEdge(gw.nodes["n2"], gw.nodes["n1"]) != (gw.g)->edge_end(gw.nodes["n2"]));
  }

#if !(DIRECTED && !IN_EDGES)
  // find incoming edges
  assert((gw.g)->findInEdge(gw.edges["e1"].dst, gw.edges["e1"].src) != (gw.g)->in_edge_end(gw.edges["e1"].dst));
  assert((gw.g)->findInEdge(gw.edges["e2"].dst, gw.edges["e2"].src) != (gw.g)->in_edge_end(gw.edges["e2"].dst));
  assert((gw.g)->findInEdge(gw.edges["e3"].dst, gw.edges["e3"].src) != (gw.g)->in_edge_end(gw.edges["e3"].dst));
  assert((gw.g)->findInEdge(gw.edges["e4"].dst, gw.edges["e4"].src) != (gw.g)->in_edge_end(gw.edges["e4"].dst));
  assert((gw.g)->findInEdge(gw.edges["e5"].dst, gw.edges["e5"].src) != (gw.g)->in_edge_end(gw.edges["e5"].dst));
  if (DIRECTED) {
    // non-existing in_edge
    assert((gw.g)->findInEdge(gw.nodes["n1"], gw.nodes["n2"]) == (gw.g)->in_edge_end(gw.nodes["n1"]));
  } else {
    // the smae as edge 4
    assert((gw.g)->findInEdge(gw.nodes["n1"], gw.nodes["n2"]) != (gw.g)->in_edge_end(gw.nodes["n1"]));
  }
#endif

#if 0
  std::cout << "g before sorting edges by dst: " << std::endl;
  printGraph(gw.g);
  std::cout << "=====" << std::endl;
#endif

  // sort edges by dst
  (gw.g)->sortAllEdgesByDst();
  assert((gw.g)->findEdgeSortedByDst(gw.edges["e1"].src, gw.edges["e1"].dst) != (gw.g)->edge_end(gw.edges["e1"].src));
  assert((gw.g)->findEdgeSortedByDst(gw.edges["e2"].src, gw.edges["e2"].dst) != (gw.g)->edge_end(gw.edges["e2"].src));
  assert((gw.g)->findEdgeSortedByDst(gw.edges["e3"].src, gw.edges["e3"].dst) != (gw.g)->edge_end(gw.edges["e3"].src));
  assert((gw.g)->findEdgeSortedByDst(gw.edges["e4"].src, gw.edges["e4"].dst) != (gw.g)->edge_end(gw.edges["e4"].src));
  assert((gw.g)->findEdgeSortedByDst(gw.edges["e5"].src, gw.edges["e5"].dst) != (gw.g)->edge_end(gw.edges["e5"].src));
  if (!DIRECTED) {
    // the same as edge 4
    assert((gw.g)->findEdgeSortedByDst(gw.nodes["n2"], gw.nodes["n1"]) != (gw.g)->edge_end(gw.nodes["n2"]));
  } else {
    // non-existing edge
    assert((gw.g)->findEdgeSortedByDst(gw.nodes["n2"], gw.nodes["n1"]) == (gw.g)->edge_end(gw.nodes["n2"]));
  }

#if 0
  std::cout << "g after sorting edges by dst: " << std::endl;
  printGraph(gw.g);
  std::cout << "=====" << std::endl;
#endif

  return gw;
}

void deleteGraphWrapper(GraphWrapper& gw) {
  delete gw.g;
  gw.name.clear();
  gw.nodes.clear();
  gw.edges.clear();
  gw.inv_nodes.clear();
}

void testBFS(GraphWrapper& gw) {
  setNumThreads(1);
  analyzeBFS(gw.g, gw.nodes["n0"], "dist");
  assert("0" == getNodeAttr(gw.g, gw.nodes["n0"], "dist"));
  assert("1" == getNodeAttr(gw.g, gw.nodes["n1"], "dist"));
  assert("2" == getNodeAttr(gw.g, gw.nodes["n2"], "dist"));
}

void testPagerank(GraphWrapper& gw) {
  setNumThreads(2);
  NodeDouble *pr = analyzePagerank(gw.g, 10, 0.01, "pr");
  assert(pr[0].n == gw.nodes["n0"] && pr[1].n == gw.nodes["n1"] && pr[2].n == gw.nodes["n2"]);
  assert(pr[0].v >= pr[1].v && pr[1].v >= pr[2].v);
  for (auto i = 3; i < 10; ++i) {
    assert(nullptr == pr[i].n && 0.0 == pr[i].v);
  }
  deleteNodeDoubles(pr);
}

void testReachability(GraphWrapper& gw) {
  setNumThreads(2);

  NodeList src = filterNode(gw.g, "color", "red");
  assert(1 == src.num);
  assert(gw.nodes["n0"] == src.nodes[0]);

  NodeList dst = filterNode(gw.g, "id", "node 2");
  assert(1 == dst.num);
  assert(gw.nodes["n2"] == dst.nodes[0]);

  int hop = 1;
  NodeList fromL = findReachableFrom(gw.g, dst, hop);
  assert(2 == fromL.num);
  assert(fromL.nodes[0] != fromL.nodes[1]);
  assert(fromL.nodes[0] == gw.nodes["n1"] || fromL.nodes[1] == gw.nodes["n1"]);
  assert(fromL.nodes[0] == gw.nodes["n2"] || fromL.nodes[1] == gw.nodes["n2"]);

  NodeList toL = findReachableTo(gw.g, src, hop);
  assert(2 == toL.num);
  assert(toL.nodes[0] != toL.nodes[1]);
  assert(toL.nodes[0] == gw.nodes["n0"] || toL.nodes[1] == gw.nodes["n0"]);
  assert(toL.nodes[0] == gw.nodes["n1"] || toL.nodes[1] == gw.nodes["n1"]);

  hop = 2;
  NodeList reach = findReachableBetween(gw.g, src, dst, hop);
  assert(3 == reach.num);
  assert(reach.nodes[0] != reach.nodes[1] && reach.nodes[0] != reach.nodes[2] && reach.nodes[1] != reach.nodes[2]);
  assert(reach.nodes[0] == gw.nodes["n0"] || reach.nodes[1] == gw.nodes["n0"] || reach.nodes[2] == gw.nodes["n0"]);
  assert(reach.nodes[0] == gw.nodes["n1"] || reach.nodes[1] == gw.nodes["n1"] || reach.nodes[2] == gw.nodes["n1"]);
  assert(reach.nodes[0] == gw.nodes["n2"] || reach.nodes[1] == gw.nodes["n2"] || reach.nodes[2] == gw.nodes["n2"]);

  deleteNodeList(src);
  deleteNodeList(dst);
  deleteNodeList(fromL);
  deleteNodeList(toL);
  deleteNodeList(reach);
}

void testSearchSubgraph(GraphWrapper& gw) {
  // set up subgraph as a length-3 cycle
  GraphWrapper gw2;
  gw2.g = createGraph();
  gw2.name = "subgraph";

  auto n = createNode(gw2.g);
  addNode(gw2.g, n);
  gw2.nodes["s.n0"] = n;
  gw2.inv_nodes[n] = "s.n0";
  n = createNode(gw2.g);
  addNode(gw2.g, n);
  gw2.nodes["s.n1"] = n;
  gw2.inv_nodes[n] = "s.n1";
  n = createNode(gw2.g);
  addNode(gw2.g, n);
  gw2.nodes["s.n2"] = n;
  gw2.inv_nodes[n] = "s.n2";

  auto e = addEdge(gw2.g, gw2.nodes["s.n0"], gw2.nodes["s.n1"]);
  gw2.edges["s.n0 -> s.n1"] = e;
  e = addEdge(gw2.g, gw2.nodes["s.n1"], gw2.nodes["s.n2"]);
  gw2.edges["s.n1 -> s.n2"] = e;
  e = addEdge(gw2.g, gw2.nodes["s.n2"], gw2.nodes["s.n0"]);
  gw2.edges["s.n2 -> s.n0"] = e;

  setNumThreads(3);

  std::unordered_map<std::string, std::string> sol[3] = { 
    { {"s.n0", "n0"}, {"s.n1", "n1"}, {"s.n2", "n2"} }, 
    { {"s.n0", "n1"}, {"s.n1", "n2"}, {"s.n2", "n0"} }, 
    { {"s.n0", "n2"}, {"s.n1", "n0"}, {"s.n2", "n1"} }
  };

  NodePair *result = searchSubgraphUllmann(gw.g, gw2.g, 10);
  for (auto i = 3; i < 10; ++i) {
    for (auto j = 0; j < 3; ++j) {
      // empty solutions
      assert(nullptr == result[i*3+j].nQ);
      assert(nullptr == result[i*3+j].nD);
    }
  }

  std::unordered_map<std::string, std::string> res_ull[3];
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      res_ull[i][gw2.inv_nodes[result[i*3+j].nQ]] = gw.inv_nodes[result[i*3+j].nD];
    }
  }
  assert(res_ull[0] != res_ull[1] && res_ull[0] != res_ull[2] && res_ull[1] != res_ull[2]);
  assert(res_ull[0] == sol[0] || res_ull[0] == sol[1] || res_ull[0] == sol[2]);
  deleteGraphMatches(result);

  result = searchSubgraphVF2(gw.g, gw2.g, 10);
  for (auto i = 3; i < 10; ++i) {
    for (auto j = 0; j < 3; ++j) {
      assert(nullptr == result[i*3+j].nQ);
      assert(nullptr == result[i*3+j].nD);
    }
  }

  std::unordered_map<std::string, std::string> res_vf2[3];
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      res_vf2[i][gw2.inv_nodes[result[i*3+j].nQ]] = gw.inv_nodes[result[i*3+j].nD];
    }
  }
  assert(res_vf2[0] != res_vf2[1] && res_vf2[0] != res_vf2[2] && res_vf2[1] != res_vf2[2]);
  assert(res_vf2[0] == sol[0] || res_vf2[0] == sol[1] || res_vf2[0] == sol[2]);
  deleteGraphMatches(result);

  deleteGraphWrapper(gw2);
}

void testCoarsening() {
  std::vector<std::string> colors = {"red", "green", "blue"};

  GraphWrapper fgw;
  fgw.g = createGraph();
  fgw.name = "fg";

  for (int i = 0; i < 10; ++i) {
    auto n = createNode(fgw.g);
    addNode(fgw.g, n);
    std::string nName = "f" + std::to_string(i);
    fgw.nodes[nName] = n;
    fgw.inv_nodes[n] = nName;
  }

  for (int i = 0; i < 9; i++) {
    std::string nName = "f" + std::to_string(i);
    setNodeAttr(fgw.g, fgw.nodes[nName], "color", const_cast<ValAltTy>(colors[i%3].c_str()));
  }

  addEdge(fgw.g, fgw.nodes["f0"], fgw.nodes["f5"]);
  addEdge(fgw.g, fgw.nodes["f1"], fgw.nodes["f6"]);
  addEdge(fgw.g, fgw.nodes["f2"], fgw.nodes["f7"]);
  addEdge(fgw.g, fgw.nodes["f3"], fgw.nodes["f9"]);
  addEdge(fgw.g, fgw.nodes["f4"], fgw.nodes["f7"]);
  addEdge(fgw.g, fgw.nodes["f5"], fgw.nodes["f2"]);
  addEdge(fgw.g, fgw.nodes["f5"], fgw.nodes["f6"]);
  addEdge(fgw.g, fgw.nodes["f8"], fgw.nodes["f7"]);
  addEdge(fgw.g, fgw.nodes["f9"], fgw.nodes["f4"]);

#if 0
  std::cout << fgw.name << ": " << std::endl;
  printGraph(fgw.g);
#endif

  GraphWrapper cgw;
  cgw.g = createGraph();
  cgw.name = "cg";
  setNumThreads(3);
  coarsen(fgw.g, cgw.g, "color");

#if 0
  std::cout << cgw.name << ": " << std::endl;
  printGraph(cgw.g);
#endif

  NodeList nl = getAllNodes(cgw.g);
  assert(3 == nl.num);
  for (auto i = 0; i < 3; ++i) {
    std::string nName = getNodeAttr(cgw.g, nl.nodes[i], "color");
    cgw.inv_nodes[nl.nodes[i]] = nName;
    cgw.nodes[nName] = nl.nodes[i];
  }

  // nodes of cg are of different colors
  assert(getNodeAttr(cgw.g, nl.nodes[0], "color") != getNodeAttr(cgw.g, nl.nodes[1], "color"));
  assert(getNodeAttr(cgw.g, nl.nodes[0], "color") != getNodeAttr(cgw.g, nl.nodes[2], "color"));
  assert(getNodeAttr(cgw.g, nl.nodes[1], "color") != getNodeAttr(cgw.g, nl.nodes[2], "color"));

  Edge cge[4] = { 
    {cgw.nodes["red"], cgw.nodes["blue"]},
    {cgw.nodes["green"], cgw.nodes["red"]},
    {cgw.nodes["blue"], cgw.nodes["red"]},
    {cgw.nodes["blue"], cgw.nodes["green"]}
  };
  EdgeList el = getAllEdges(cgw.g);
  assert(4 == el.num);
  for (auto i = 0; i < 4; ++i) {
    // edges of cg connect nodes of different colors
    assert(el.edges[i].src != el.edges[i].dst);
    // no repeated edges
    for (auto j = i+1; j < 4; ++j) {
      assert(el.edges[i] != el.edges[j]);
    }
    // must be in cge
    assert(el.edges[i] == cge[0] || el.edges[i] == cge[1] || el.edges[i] == cge[2] || el.edges[i] == cge[3]);
  }

  deleteNodeList(nl);
  deleteEdgeList(el);
  deleteGraphWrapper(fgw);
  deleteGraphWrapper(cgw);
}

int main(int argc, char *argv[]) {
  std::cout << "testGraphConstruction()...";
  auto gw = testGraphConstruction();
  std::cout << " pass" << std::endl;

  std::cout << "testBFS()...";
  testBFS(gw);
  std::cout << " pass" << std::endl;

  std::cout << "testPagerank()...";
  testPagerank(gw);
  std::cout << " pass" << std::endl;

  std::cout << "testReachability()...";
  testReachability(gw);
  std::cout << " pass" << std::endl;

  std::cout << "testSearchSubgraph()...";
  testSearchSubgraph(gw);
  std::cout << " pass" << std::endl;

  deleteGraphWrapper(gw);

  std::cout << "testCoarsening()...";
  testCoarsening();
  std::cout << " pass" << std::endl;

  return 0;
}

