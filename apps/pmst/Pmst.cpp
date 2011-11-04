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
#include "Galois/Runtime/DebugWorkList.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include "Exp/PriorityScheduling/WorkListTL.h"

#include <limits>
#include <algorithm>
#include <vector>
#include <iostream>
 
#define BORUVKA_DEBUG 0 

static const char* name = "Parallel MST";
static const char* description = "Computes the Minimal Spanning Tree using combination of "
  "Boruvka's and Prim's algorithm\n";
static const char* url = 0;
static const char* help = "[-algo N] [-wl string] <input file>";

typedef int Weight;
typedef Galois::GAccumulator<size_t> MstWeight;

struct Prims {
  struct Node;

  typedef Galois::Graph::FirstGraph<Node*, Weight, true> Graph;
  typedef Graph::GraphNode GraphNode;

  struct HeapItem {
    GraphNode node;
    Weight weight;
    HeapItem() { }
    HeapItem(GraphNode n): node(n), weight(std::numeric_limits<int>::max()) { }
    HeapItem(GraphNode n, Weight w): node(n), weight(w) { }
    bool operator<(const HeapItem& other) const {
      return weight < other.weight;
    }
  };

  typedef Galois::PairingHeap<HeapItem, std::less<HeapItem>,
          Galois::PerIterAllocTy::rebind<HeapItem>::other> Heap;

  struct Node {
    GraphNode parent;
    Heap::Handle handle;
    int id;
    bool present;
    Node(Heap::Handle h): handle(h), present(true) { }
  };

  typedef Galois::PerIterAllocTy::rebind<Node>::other NodeAlloc;

  std::string name() { return std::string("Serial Prim"); }

  void expandNeighborhood(Graph& g, const GraphNode& src) {
    
  }

  void doIt(Graph& g, const GraphNode& root, Galois::UserContext<GraphNode>& ctx, 
      MstWeight& mstWeight) {
    //Heap heap(5, ctx.getPerIterAlloc());
    Heap heap(ctx.getPerIterAlloc());
    NodeAlloc nodeAlloc(ctx.getPerIterAlloc());

    int i = 0;
    for (Graph::active_iterator ii = g.active_begin(), ei = g.active_end(); ii != ei; ++ii) {
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
      GraphNode src = retval.second.node;
      //std::cout << " Got (" << retval.second.weight << "," << src.getData()->id << ")\n";
      for (Graph::neighbor_iterator dst = g.neighbor_begin(src),
          edst = g.neighbor_end(src); dst != edst; ++dst) {
        Node* node = dst->getData();
        //std::cout << " Seeing " << node->id << " " 
        //  << heap.value(node->handle).node.getData()->id << " " 
        //  << heap.value(node->handle).weight << " " << "\n";
        if (node->present) {
          const Weight& w = g.getEdgeData(src, dst);
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

    for (Graph::active_iterator ii = g.active_begin(), ei = g.active_end(); ii != ei; ++ii) {
      if (*ii != root)
        mstWeight += g.getEdgeData(*ii, ii->getData()->parent);
      // Automatically reclaimed, but don't leave dangling pointers around
      ii->getData() = NULL;
    }
  }

  struct Process {
    Prims& parent;
    Graph& g;
    MstWeight& mstWeight;
    Process(Prims& p, Graph& _g, MstWeight& m): parent(p), g(_g), mstWeight(m) { }
    void operator()(const GraphNode& root, Galois::UserContext<GraphNode>& ctx) {
      parent.doIt(g, root, ctx, mstWeight);
    }
  };

  void operator()(Graph& g, MstWeight& w) {
    std::vector<GraphNode> start;
    for (Graph::active_iterator ii = g.active_begin(), ei = g.active_end(); ii != ei; ++ii) {
      start.push_back(*ii);
      break;
    }

    Galois::setMaxThreads(1);
    Galois::for_each(start.begin(), start.end(), Process(*this, g, w));
  }
};


struct Boruvkas {

  typedef Galois::Graph::FirstGraph<unsigned, Weight, true> Graph;
  typedef Graph::GraphNode GraphNode;
  typedef std::pair<GraphNode, Weight> Edge;
  typedef std::vector<Edge, Galois::PerIterAllocTy::rebind<Edge>::other> EdgeList;

  struct EdgeLess : public std::binary_function<const Edge&, const Edge&, bool> {
    bool operator()(const Edge& a, const Edge& b) {
      return a.first != b.first ? a.first < b.first : a.second < b.second;
    }
  };

  struct Indexer: public std::unary_function<const GraphNode&,unsigned> {
    unsigned operator()(const GraphNode& n) const {
      return n.getData(Galois::NONE);
    }
  };

  struct Less: public std::binary_function<const GraphNode&,const GraphNode&,bool> {
    bool operator()(const GraphNode& lhs, const GraphNode& rhs) const {
      return lhs.getData(Galois::NONE) < rhs.getData(Galois::NONE);
    }
  };
  struct Greater: public std::binary_function<const GraphNode&,const GraphNode&,bool> {
    bool operator()(const GraphNode& lhs, const GraphNode& rhs) const {
      return lhs.getData(Galois::NONE) > rhs.getData(Galois::NONE);
    }
  };

  std::string name() { return std::string("Boruvka's"); }

  void expandNeighborhood(Graph& g, const GraphNode& src) {
    for (Graph::neighbor_iterator ii = g.neighbor_begin(src, Galois::ALL), 
        ei = g.neighbor_end(src, Galois::ALL); ii != ei; ++ii) {
      //graph.getEdgeData(src, *ii);
    }
  }

  std::pair<GraphNode, Weight> findMin(Graph& g, const GraphNode& src, Galois::MethodFlag flag) {
    Weight minWeight = std::numeric_limits<Weight>::max();
    GraphNode minNode;

    for (Graph::neighbor_iterator dst = g.neighbor_begin(src, flag), 
        edst = g.neighbor_end(src, flag); dst != edst; ++dst) {
      
      const Weight& w = g.getEdgeData(src, dst, flag);
      if (w < minWeight){
        minNode = *dst;
        minWeight = w;
      }
    }

    return std::make_pair(minNode, minWeight);
  }

  GraphNode collapseEdge(Graph& g, GraphNode& a, GraphNode& b,
      Galois::UserContext<GraphNode>& ctx, Galois::MethodFlag flag) {
    EdgeList edges(ctx.getPerIterAlloc());

    for (Graph::neighbor_iterator dst = g.neighbor_begin(a, flag),
        edst = g.neighbor_end(a, flag); dst != edst; ++dst) {
      if (*dst != b) {
        edges.push_back(std::make_pair(*dst, g.getEdgeData(a, dst, flag)));
        --dst->getData(flag);
      }
    }

    for (Graph::neighbor_iterator dst = g.neighbor_begin(b, flag),
        edst = g.neighbor_end(b, flag); dst != edst; ++dst) {
      if (*dst != a) {
        edges.push_back(std::make_pair(*dst, g.getEdgeData(b, dst, flag)));
        --dst->getData(flag);
      }
    }

    g.removeNode(a, flag);
    g.removeNode(b, flag);

    std::sort(edges.begin(), edges.end(), EdgeLess());

    GraphNode n = g.createNode(Graph::node_type());
    g.addNode(n);
    GraphNode last;
    unsigned numNeighbors = 0;
    for (EdgeList::iterator ii = edges.begin(), ei = edges.end(); ii != ei; ++ii) {
      if (ii->first == last)
        continue;

      g.addMultiEdge(n, ii->first, ii->second, flag);
      g.addMultiEdge(ii->first, n, ii->second, flag);
      --ii->first.getData(flag);
      numNeighbors++;

      last = ii->first;
    }

    n.getData(flag) = numNeighbors;

    return n;
  }

  struct Process {
    Boruvkas& parent;
    Graph& g;
    MstWeight& mstWeight;

    Process(Boruvkas& p, Graph& _g, MstWeight& m): parent(p), g(_g), mstWeight(m) { }

    void operator()(GraphNode& src, Galois::UserContext<GraphNode>& ctx) {
      parent.expandNeighborhood(g, src);

      if (!g.containsNode(src))
        return;

      std::pair<GraphNode,Weight> minp = parent.findMin(g, src, Galois::NONE);

      if (minp.second == std::numeric_limits<Weight>::max()) {
        g.removeNode(src, Galois::NONE);
        return;
      }

      parent.expandNeighborhood(g, minp.first);
      
      GraphNode rep = parent.collapseEdge(g, src, minp.first, ctx, Galois::NONE);
      mstWeight += minp.second;
      ctx.push(rep);
    }
  };

  void operator()(Graph& g, MstWeight& w) {
    for (Graph::active_iterator ii = g.active_begin(), ei = g.active_end(); ii != ei; ++ii) {
      ii->getData() = std::distance(g.neighbor_begin(*ii), g.neighbor_end(*ii));
    }

    using namespace GaloisRuntime::WorkList;
    typedef dChunkedLIFO<16> IChunk;
    typedef OrderedByIntegerMetric<Indexer, IChunk> OBIM;

    Exp::StartWorklistExperiment<OBIM,dChunkedLIFO<16>, ChunkedLIFO<16>,Indexer,Less,Greater>()(
        std::cout, g.active_begin(), g.active_end(), Process(*this, g, w));
  }
};

template<typename Graph>
void makeGraph(const char* in, Graph& g) {
  typedef typename Graph::GraphNode GraphNode;
  typedef Galois::Graph::LC_FileGraph<int, Weight> ReaderGraph;
  typedef ReaderGraph::GraphNode ReaderGNode;

  ReaderGraph reader;
  reader.structureFromFile(in);
  reader.emptyNodeData();

  // Assign ids to ReaderGNodes
  size_t numNodes = 0;
  for (ReaderGraph::active_iterator ii = reader.active_begin(),
      ee = reader.active_end(); ii != ee; ++ii, ++numNodes) {
    ReaderGNode src = *ii;
    reader.getData(src) = numNodes;
  }

  // Create dense map between ids and GNodes
  std::vector<GraphNode> nodes;
  nodes.resize(numNodes);
  for (size_t i = 0; i < numNodes; ++i) {
    GraphNode src = g.createNode(typename Graph::node_type());
    g.addNode(src);
    nodes[i] = src;
  }

  // Create edges
  size_t numEdges = 0;
  for (ReaderGraph::active_iterator ii = reader.active_begin(),
      ei = reader.active_end(); ii != ei; ++ii) {
    ReaderGNode rsrc = *ii;
    int rsrcId = reader.getData(rsrc);
    for (ReaderGraph::neighbor_iterator jj = reader.neighbor_begin(rsrc),
        ej = reader.neighbor_end(rsrc); jj != ej; ++jj) {
      ReaderGNode rdst = *jj;
      int rdstId = reader.getData(rdst);
      const Weight& w = reader.getEdgeData(rsrc, rdst);
      GraphNode gsrc = nodes[rsrcId];
      GraphNode gdst = nodes[rdstId];
      if (gsrc.hasNeighbor(gdst)) {
        Weight& ww = g.getEdgeData(gsrc, gdst);
        if (ww > w) {
          ww = w;
        }
      } else if (gsrc != gdst) {
        g.addMultiEdge(gsrc, gdst, w);
        g.addMultiEdge(gdst, gsrc, w);
        numEdges += 2;
      }
    }
  }

  std::cout << "Read " << numNodes << " nodes and " << numEdges << " edges\n";
}

template<typename Algo>
void start(const char* in) {
  typedef typename Algo::Graph Graph;
  MstWeight w;
  Graph g;
  Algo algo;
  makeGraph<Graph>(in, g);

  std::cout << "Using " << algo.name() << "\n";
  Galois::StatTimer T;
  T.start();
  algo(g, w);
  T.stop();
  std::cout << "MST Weight is " << w.get() << "\n";
}

int main(int argc, const char **argv) {
  int algo = 0;
  std::vector<const char*> args = parse_command_line(argc, argv, help);
  Exp::parse_worklist_command_line(args);

  for (std::vector<const char*>::iterator ii = args.begin(), ei = args.end(); ii != ei; ++ii) {
    if (strcmp(*ii, "-algo") == 0 && ii + 1 != ei) {
      algo = atoi(ii[1]);
      ii = args.erase(ii);
      ii = args.erase(ii);
      --ii;
      ei = args.end();
    }
  }

  if (args.size() < 1) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }

  const char* in = args[0];

  printBanner(std::cout, name, description, url);

  switch (algo) {
    case 1: start<Prims>(in); break;
    case 0:
    default: start<Boruvkas>(in); break;
  }

  return 0;
}
