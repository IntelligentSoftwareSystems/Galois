/** Parallel MST application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Rashid Kaleem <rashid@cs.utexas.edu>
 */


#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Timer.h"
#include "Galois/Queue.h"
#include "Galois/UserContext.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/FileGraph.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <limits>
#include <algorithm>
#include <vector>
#include <iostream>
 
#define BORUVKA_DEBUG 0 

static const char* name = "Parallel MST";
static const char* description = "Computes the Minimal Spanning Tree using combination of "
  "Boruvka's and Prim's algorithm\n";
static const char* url = 0;
static const char* help = "<input file>";

struct Node;

typedef Galois::Graph::FirstGraph<Node*, int, true> Graph;
typedef Graph::GraphNode GNode;
typedef Galois::GAccumulator<double> MstWeight;

Graph graph;
int NumNodes;
int NumEdges;

struct HeapItem {
  GNode node;
  int weight;
  HeapItem() { }
  HeapItem(GNode n): node(n), weight(std::numeric_limits<int>::max()) { }
  HeapItem(GNode n, int w): node(n), weight(w) { }
  bool operator<(const HeapItem& other) const {
    return weight < other.weight;
  }
};

//typedef Galois::PairingHeap<HeapItem, std::less<HeapItem>, Galois::PerIterAllocTy::rebind<HeapItem>::other> Heap;
typedef Galois::PairingHeap<HeapItem> Heap;

struct Node {
  GNode parent;
  Heap::Handle handle;
  int id;
  bool present;
  Node(Heap::Handle h): handle(h), present(true) { }
};

struct PrimProcess {
  typedef Galois::PerIterAllocTy::rebind<Node>::other NodeAlloc;
  MstWeight* m_mst_weight;

  PrimProcess(MstWeight* w): m_mst_weight(w) { }
  
  void expandNeighborhood(const GNode& src) {
    
  }

  
  void operator()(const GNode& root, Galois::UserContext<GNode>& ctx) {
    //Heap heap(5, ctx.getPerIterAlloc());
    Heap heap;
    NodeAlloc nodeAlloc(ctx.getPerIterAlloc());

    int i = 0;
    for (Graph::active_iterator ii = graph.active_begin(), ei = graph.active_end(); ii != ei; ++ii) {
      Heap::Handle h;
      if (*ii != root)
        h = heap.add(HeapItem(*ii));
      else
        h = heap.add(HeapItem(*ii, 0));
      NodeAlloc::pointer node = nodeAlloc.allocate(1);
      nodeAlloc.construct(node, Node(h));
      ii->getData() = node;
      node->id = i++;
    }

    std::pair<bool,HeapItem> retval = heap.pollMin();
    while (retval.first) {
      GNode src = retval.second.node;
      //std::cout << " Got (" << retval.second.weight << "," << src.getData()->id << ")\n";
      for (Graph::neighbor_iterator dst = graph.neighbor_begin(src),
          edst = graph.neighbor_end(src); dst != edst; ++dst) {
        Node* node = dst->getData();
        //std::cout << " Seeing " << node->id << " " 
        //  << heap.value(node->handle).node.getData()->id << " " 
        //  << heap.value(node->handle).weight << " " << "\n";
        if (node->present) {
          int w = graph.getEdgeData(src, *dst);
          HeapItem item = heap.value(node->handle);
          if (w < item.weight) {
            node->parent = src;
            //std::cout << " Decrease (" << item.weight << "," << item.node.getData()->id << ") to " 
            //  << w << "\n";
            heap.decreaseKey(node->handle, HeapItem(item.node, w));
          }
        }
      }
      
      src.getData()->present = false;

      retval = heap.pollMin();
    }

    for (Graph::active_iterator ii = graph.active_begin(), ei = graph.active_end(); ii != ei; ++ii) {
      if (*ii != root)
        *m_mst_weight += graph.getEdgeData(*ii, ii->getData()->parent);
      // Automatically reclaimed, but don't leave dangling pointers around
      ii->getData() = NULL;
    }
  }
};



struct BoruvkaProcess {
  MstWeight* m_mst_weight;
  BoruvkaProcess(MstWeight* w): m_mst_weight(w) { }

  void expandNeighborhood(const GNode& src) {
    for (Graph::neighbor_iterator ii = graph.neighbor_begin(src, Galois::ALL), 
        ei = graph.neighbor_end(src, Galois::ALL); ii != ei; ++ii) {
      graph.getEdgeData(src, *ii);
    }
  }

  std::pair<GNode, int> findMin(const GNode& src, Galois::MethodFlag flag) {
    int minWeight = std::numeric_limits<int>::max();
    GNode minNode;

    for (Graph::neighbor_iterator dst = graph.neighbor_begin(src, flag), 
        edst = graph.neighbor_end(src, flag); dst != edst; ++dst) {
      
      int w = graph.getEdgeData(src, *dst, flag);
      if (w < minWeight){
        minNode = *dst;
        minWeight = w;
      }
    }

    return std::make_pair(minNode, minWeight);
  }

  typedef std::pair<GNode, int> Edge;

  struct EdgeLess : public std::binary_function<const Edge&, const Edge&, bool> {
    bool operator()(const Edge& a, const Edge& b) {
      return a.first != b.first ? a.first < b.first : a.second < b.second;
    }
  };

  GNode collapseEdge(GNode& a, GNode& b, Galois::UserContext<GNode>& ctx, Galois::MethodFlag flag) {
    typedef std::vector<Edge, Galois::PerIterAllocTy::rebind<Edge>::other> Vector;

    Vector edges(ctx.getPerIterAlloc());

    for (Graph::neighbor_iterator dst = graph.neighbor_begin(a, flag),
        edst = graph.neighbor_end(a, flag); dst != edst; ++dst) {
      if (*dst != b)
        edges.push_back(std::make_pair(*dst, graph.getEdgeData(a, *dst, flag)));
    }

    for (Graph::neighbor_iterator dst = graph.neighbor_begin(b, flag),
        edst = graph.neighbor_end(b, flag); dst != edst; ++dst) {
      if (*dst != a)
        edges.push_back(std::make_pair(*dst, graph.getEdgeData(b, *dst, flag)));
    }

    graph.removeNode(a, flag);
    graph.removeNode(b, flag);

    std::sort(edges.begin(), edges.end(), EdgeLess());

    GNode n = graph.createNode(Graph::node_type());
    graph.addNode(n);
    GNode last;
    for (Vector::iterator ii = edges.begin(), ei = edges.end(); ii != ei; ++ii) {
      if (ii->first == last)
        continue;

      graph.addEdge(n, ii->first, ii->second, flag);
      graph.addEdge(ii->first, n, ii->second, flag);

      last = ii->first;
    }

    return n;
  }

  void operator()(GNode& src, Galois::UserContext<GNode>& ctx) {
    expandNeighborhood(src);

    if (!graph.containsNode(src))
      return;

    std::pair<GNode,int> minp = findMin(src, Galois::NONE);

    if (minp.second == std::numeric_limits<int>::max()) {
      graph.removeNode(src, Galois::NONE);
      return;
    }

    expandNeighborhood(minp.first);
    
    GNode rep = collapseEdge(src, minp.first, ctx, Galois::ALL); //Galois::NONE);
    *m_mst_weight += minp.second;
    ctx.push(rep);
  }
};

static void makeGraph(const char* input) {
  typedef Galois::Graph::LC_FileGraph<int, int> ReaderGraph;
  typedef ReaderGraph::GraphNode ReaderGNode;

  ReaderGraph reader;
  reader.structureFromFile(input);
  reader.emptyNodeData();

  // Assign ids to ReaderGNodes
  NumNodes = 0;
  for (ReaderGraph::active_iterator ii = reader.active_begin(),
      ee = reader.active_end(); ii != ee; ++ii, ++NumNodes) {
    ReaderGNode src = *ii;
    reader.getData(src) = NumNodes;
  }

  // Create dense map between ids and GNodes
  std::vector<GNode> nodes;
  nodes.resize(NumNodes);
  for (int i = 0; i < NumNodes; ++i) {
    GNode src = graph.createNode(Graph::node_type());
    graph.addNode(src);
    nodes[i] = src;
  }

  // Create edges
  NumEdges = 0;
  for (ReaderGraph::active_iterator ii = reader.active_begin(),
      ei = reader.active_end(); ii != ei; ++ii) {
    ReaderGNode rsrc = *ii;
    int rsrcId = reader.getData(rsrc);
    for (ReaderGraph::neighbor_iterator jj = reader.neighbor_begin(rsrc),
        ej = reader.neighbor_end(rsrc); jj != ej; ++jj) {
      ReaderGNode rdst = *jj;
      int rdstId = reader.getData(rdst);
      int w = reader.getEdgeData(rsrc, rdst);
      GNode gsrc = nodes[rsrcId];
      GNode gdst = nodes[rdstId];
      if (gsrc.hasNeighbor(gdst)) {
        int& ww = graph.getEdgeData(gsrc, gdst);
        if (ww > w) {
          ww = w;
        }
      } else if (gsrc != gdst) {
        graph.addEdge(gsrc, gdst, w);
        graph.addEdge(gdst, gsrc, w);
        NumEdges += 2;
      }
    }
  }

  std::cout << "Read " << NumNodes << " nodes and " << NumEdges << " edges\n";
}


int main(int argc, const char **argv) {
  using namespace GaloisRuntime::WorkList;

  std::vector<const char*> args = parse_command_line(argc, argv, help);
  if (args.size() < 1) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }

  printBanner(std::cout, name, description, url);
  const char* inputfile = args[0];
  makeGraph(inputfile);

  {
    Galois::StatTimer T("prim");
    T.start();
    MstWeight w;
    std::vector<GNode> start;
    for (Graph::active_iterator ii = graph.active_begin(), ei = graph.active_end(); ii != ei; ++ii) {
      start.push_back(*ii);
      break;
    }
    Galois::for_each(start.begin(), start.end(), PrimProcess(&w));
    std::cout << "MST Weight is " << w.get() << "\n";
    T.stop();
  }

  Galois::StatTimer T;
  T.start();
  MstWeight w;
  Galois::for_each(graph.active_begin(), graph.active_end(), BoruvkaProcess(&w));
  std::cout << "MST Weight is " << w.get() << "\n";
  T.stop();
  

  return 0;
}
