#include "PythonGraph.h"
#include "AnalyzeBFS.h"
#include "SearchSubgraph.h"
#include "PageRank.h"
#include "Filter.h"
#include "Reachability.h"
#include "Coarsen.h"

#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
  std::vector<GNode> nodes;
  std::vector<Edge> edges;

  Graph *g = createGraph();

  // add nodes and set node attributes
  nodes.push_back(createNode(g));
  addNode(g, nodes[0]);
  setNodeAttr(g, nodes[0], "color", "red");
  setNodeAttr(g, nodes[0], "id", "node 0");

  nodes.push_back(createNode(g));
  addNode(g, nodes[1]);
  setNodeAttr(g, nodes[1], "language", "english");
  setNodeAttr(g, nodes[1], "garbage", "to_be_deleted");
  setNodeAttr(g, nodes[1], "id", "node 1");

  nodes.push_back(createNode(g));
  addNode(g, nodes[2]);
  setNodeAttr(g, nodes[2], "date", "Oct. 24, 2016");
  setNodeAttr(g, nodes[2], "id", "node 2");
  printGraph(g);
  std::cout << "=====" << std::endl;

  // remove node attributes
  removeNodeAttr(g, nodes[1], "garbage");
  AttrList attrL = getNodeAllAttr(g, nodes[1]);
  for(auto i = 0; i < attrL.num; i++) {
    std::cout << attrL.key[i] << ": " << attrL.value[i] << std::endl;
  }
  deleteAttrList(attrL);
  std::cout << "=====" << std::endl;

  // add edges and set edge attributes
  edges.push_back(addEdge(g, nodes[0], nodes[1]));
  setEdgeAttr(g, edges[0], "weight", "3.0");
  setEdgeAttr(g, edges[0], "id", "edge 0: 0 -> 1");
  printGraph(g);
  std::cout << "=====" << std::endl;

  // merge edges
  edges.push_back(addEdge(g, nodes[0], nodes[1]));
  setEdgeAttr(g, edges[1], "place", "texas");
  setEdgeAttr(g, edges[1], "garbage", "discard");
  setEdgeAttr(g, edges[1], "id", "edge 1: 0 -> 1");
  printGraph(g);
  std::cout << "=====" << std::endl;

  // remove edge attributes
  removeEdgeAttr(g, edges[1], "garbage");
  removeEdgeAttr(g, edges[1], "galois_id");
  printGraph(g);
  std::cout << "=====" << std::endl;

  // add a self loop
  edges.push_back(addEdge(g, nodes[0], nodes[0]));
  setEdgeAttr(g, edges[2], "id", "edge 2: 0 -> 0");

  // add a length-2 loop
  edges.push_back(addEdge(g, nodes[1], nodes[0]));
  setEdgeAttr(g, edges[3], "id", "edge 3: 1 -> 0");
  printGraph(g);
  std::cout << "=====" << std::endl;

  // add a length-3 loop
  edges.push_back(addEdge(g, nodes[1], nodes[2]));
  setEdgeAttr(g, edges[4], "id", "edge 4: 1 -> 2");
  edges.push_back(addEdge(g, nodes[2], nodes[0]));
  setEdgeAttr(g, edges[5], "id", "edge 5: 2 -> 0");
  printGraph(g);
  std::cout << "=====" << std::endl;

  // node filter
  setNumThreads(2);
  NodeList l = filterNode(g, "id", "node 1");
  if (0 == l.num) 
    std::cout << "no nodes whose id is \"node 1\"\n=====" << std::endl;
  else {
    std::cout << "nodes whose id is \"node 1\":\n";
    for (auto i = 0; i < l.num; ++i) {
      std::cout << l.nodes[i] << " ";
    }
    std::cout << "\n=====" << std::endl;
    deleteNodeList(l);
  }
  
  // BFS analysis
  setNumThreads(1);
  analyzeBFS(g, nodes[0], "dist");
  printGraph(g);
  std::cout << "=====" << std::endl;

  // reachability
  setNumThreads(2);
  NodeList src = createNodeList(1);
  src.nodes[0] = nodes[2];
  std::cout << "src node(s):\n";
  printNodeList(src);
  NodeList dst = createNodeList(1);
  dst.nodes[0] = nodes[1];
  std::cout << "dst node(s):\n";
  printNodeList(dst);
  int hop = 1;
  NodeList fromL = findReachableFrom(g, dst, hop);
  std::cout << "nodes reachable to dst within " << hop << " step(s):\n";
  printNodeList(fromL);
  NodeList toL = findReachableTo(g, src, hop);
  std::cout << "nodes reachable from src within " << hop << " step(s):\n";
  printNodeList(toL);
  hop = 2;
  NodeList reach = findReachableBetween(g, src, dst, hop);
  std::cout << "nodes reachable from src to dst within " << hop << " step(s):\n";
  printNodeList(reach);
  std::cout << "=====" << std::endl;
  deleteNodeList(reach); 
  deleteNodeList(src);
  deleteNodeList(dst);

  // find outgoing edges
  if(g->findEdge(nodes[0], nodes[1]) != g->edge_end(nodes[0]))
    std::cout << "edge 1 exists" << std::endl;

  if(g->findEdge(nodes[0], nodes[0]) != g->edge_end(nodes[0]))
    std::cout << "edge 2 exists" << std::endl;

  if(g->findEdge(nodes[1], nodes[0]) != g->edge_end(nodes[1]))
    std::cout << "edge 3 exists" << std::endl;

  if(g->findEdge(nodes[1], nodes[2]) != g->edge_end(nodes[1]))
    std::cout << "edge 4 exists" << std::endl;

  if(g->findEdge(nodes[2], nodes[0]) != g->edge_end(nodes[2]))
    std::cout << "edge 5 exists" << std::endl;

  if(g->findEdge(nodes[2], nodes[1]) != g->edge_end(nodes[2]))
    std::cout << ((DIRECTED) ? "non-existing edge!" : "implied by edge 4") << std::endl;
  std::cout << "=====" << std::endl;

#if !(DIRECTED && !IN_EDGES)
  // find incoming edges
  if(g->findInEdge(nodes[1], nodes[0]) != g->in_edge_end(nodes[1]))
    std::cout << "in_edge 1 exists" << std::endl;

  if(g->findInEdge(nodes[0], nodes[0]) != g->in_edge_end(nodes[0]))
    std::cout << "in_edge 2 exists" << std::endl;

  if(g->findInEdge(nodes[0], nodes[1]) != g->in_edge_end(nodes[0]))
    std::cout << "in_edge 3 exists" << std::endl;

  if(g->findInEdge(nodes[2], nodes[1]) != g->in_edge_end(nodes[2]))
    std::cout << "in_edge 4 exists" << std::endl;

  if(g->findInEdge(nodes[0], nodes[2]) != g->in_edge_end(nodes[0]))
    std::cout << "in_edge 5 exists" << std::endl;

  if(g->findInEdge(nodes[1], nodes[2]) != g->in_edge_end(nodes[1]))
    std::cout << ((DIRECTED) ? "non-existing in_edge!" : "the same as edge 4") << std::endl;
  std::cout << "=====" << std::endl;
#endif

  // sort edges by dst
  g->sortAllEdgesByDst();
  printGraph(g);
  std::cout << "=====" << std::endl;

  // find edges after sorting edges
  auto e = g->findEdgeSortedByDst(nodes[0], nodes[1]);
  if(e != g->edge_end(nodes[0])) {
    std::cout << "find edge 1 after sorting edges" << std::endl;
    for(auto i: g->getEdgeData(e))
      std::cout << "  " << i.first << ": " << i.second << std::endl;
  }

  e = g->findEdgeSortedByDst(nodes[0], nodes[0]);
  if(e != g->edge_end(nodes[0])) {
    std::cout << "find edge 2 after sorting edges" << std::endl;
    for(auto i: g->getEdgeData(e))
      std::cout << "  " << i.first << ": " << i.second << std::endl;
  }

  e = g->findEdgeSortedByDst(nodes[1], nodes[0]);
  if(e != g->edge_end(nodes[1])) {
    std::cout << "find edge 3 after sorting edges" << std::endl;
    for(auto i: g->getEdgeData(e))
      std::cout << "  " << i.first << ": " << i.second << std::endl;
  }

  e = g->findEdgeSortedByDst(nodes[1], nodes[2]);
  if(e != g->edge_end(nodes[1])) {
    std::cout << "find edge 4 after sorting edges" << std::endl;
    for(auto i: g->getEdgeData(e))
      std::cout << "  " << i.first << ": " << i.second << std::endl;
  }

  e = g->findEdgeSortedByDst(nodes[2], nodes[0]);
  if(e != g->edge_end(nodes[2])) {
    std::cout << "find edge 5 after sorting edges" << std::endl;
    for(auto i: g->getEdgeData(e))
      std::cout << "  " << i.first << ": " << i.second << std::endl;
  }

  e = g->findEdgeSortedByDst(nodes[2], nodes[1]);
  if(e != g->edge_end(nodes[2])) {
    std::cout << ((!DIRECTED) ? "the same as edge 4" : "find non-existing edge after sorting edges") << std::endl;
    for(auto i: g->getEdgeData(e))
      std::cout << "  " << i.first << ": " << i.second << std::endl;
  } 
  std::cout << "=====" << std::endl;

  // subgraph isomorphism
  Graph *g2 = createGraph();
  nodes.push_back(createNode(g2));
  addNode(g2, nodes[3]);
  nodes.push_back(createNode(g2));
  addNode(g2, nodes[4]);
  nodes.push_back(createNode(g2));
  addNode(g2, nodes[5]);
  edges.push_back(addEdge(g2, nodes[3], nodes[4]));
  edges.push_back(addEdge(g2, nodes[4], nodes[5]));
  edges.push_back(addEdge(g2, nodes[5], nodes[3]));
  printGraph(g2);
  std::cout << "=====" << std::endl;

  setNumThreads(3);
  NodePair *result = searchSubgraphUllmann(g, g2, 10);
  deleteGraphMatches(result);
  std::cout << "=====" << std::endl;

  result = searchSubgraphVF2(g, g2, 10);
  deleteGraphMatches(result);
  deleteGraph(g2);
  std::cout << "=====" << std::endl;

  // pagerank
  setNumThreads(2);
  NodeDouble *pr = analyzePagerank(g, 10, 0.01, "pr");
  deleteNodeDoubles(pr);
  std::cout << "=====" << std::endl;

  deleteGraph(g);
  nodes.clear();
  edges.clear();

  // coarsen
  std::vector<std::string> colors = {"red", "green", "blue"};
  Graph *fg = createGraph();
  for (int i = 0; i < 10; ++i) {
    nodes.push_back(createNode(fg));
    addNode(fg, nodes[i]);
  }
  for (int i = 0; i < 9; i++) {
    setNodeAttr(fg, nodes[i], "color", const_cast<ValAltTy>(colors[i%3].c_str()));
  }
  edges.push_back(addEdge(fg, nodes[0], nodes[5]));
  edges.push_back(addEdge(fg, nodes[1], nodes[6]));
  edges.push_back(addEdge(fg, nodes[2], nodes[7]));
  edges.push_back(addEdge(fg, nodes[3], nodes[9]));
  edges.push_back(addEdge(fg, nodes[4], nodes[7]));
  edges.push_back(addEdge(fg, nodes[5], nodes[2]));
  edges.push_back(addEdge(fg, nodes[5], nodes[6]));
  edges.push_back(addEdge(fg, nodes[8], nodes[7]));
  edges.push_back(addEdge(fg, nodes[9], nodes[4]));
  printGraph(fg);
  std::cout << "coarsening..." << std::endl;

  setNumThreads(3);
  Graph *cg = createGraph(); 
  coarsen(fg, cg, "color");
  printGraph(cg);

  deleteGraph(fg);
  deleteGraph(cg);
  nodes.clear();
  edges.clear();

  return 0;
}

