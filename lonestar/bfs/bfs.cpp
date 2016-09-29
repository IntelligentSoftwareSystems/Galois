/** Breadth-first search -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Breadth-first search.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Statistic.h"
#include "Galois/StatTimer.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>

namespace cll = llvm::cl;

static const char* name = "Breadth-first Search";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a direction reversing algorithm";
static const char* url = "breadth_first_search";

static cll::opt<std::string> filename(cll::Positional, 
                                      cll::desc("<input graph>"), 
                                      cll::Required);

static cll::opt<std::string> transposeGraphName(cll::Positional,
                                                cll::desc("<transpose graph>"),
                                                cll::Required);

static cll::opt<unsigned int> source("startNode",
                                     cll::desc("Node to start search from"),
                                     cll::init(0));

static cll::opt<bool> dumpResult("dump", 
                                 cll::desc("Dump result"),
                                 cll::init(false));

constexpr Galois::MethodFlag unflag = Galois::MethodFlag::UNPROTECTED;

template<typename Graph, typename NodeBag>
struct BackwardProcess {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_push;
  
  using GNode = typename Graph::GraphNode;

  Graph& graph;
  NodeBag& nextBag;
  Galois::GAccumulator<size_t>& count;
  unsigned newDist; 
  BackwardProcess(Graph& g, NodeBag& n, Galois::GAccumulator<size_t>& c, int d): graph(g), nextBag(n), count(c), newDist(d) { }
  
  void operator()(const GNode& n) const {
    auto& sdata = graph.getData(n, unflag);
    if (sdata <= newDist)
      return;
    
    for (auto ii : graph.in_edges(n, unflag)) {
      GNode dst = graph.getInEdgeDst(ii);
      auto& ddata = graph.getData(dst, unflag);
      
      if (ddata + 1 == newDist) {
        sdata = newDist;
        nextBag.push(n);
        count += 1
          + std::distance(graph.edge_begin(n, unflag),
                          graph.edge_end(n, unflag));
        break;
      }
    }
  }
};


template<typename Graph, typename NodeBag>
struct ForwardProcess {
  typedef int tt_does_not_need_aborts;
  
  Graph& graph;
  NodeBag& nextBag;
  Galois::GAccumulator<size_t>& count;
  unsigned newDist;
  
  using GNode = typename Graph::GraphNode;
  
  ForwardProcess(Graph& g, NodeBag& n, Galois::GAccumulator<size_t>& c, unsigned d):
    graph(g), nextBag(n), count(c), newDist(d) { }
  
  void operator()(const GNode& n) {
    for (auto ii : graph.edges(n, unflag))
      (*this)(ii);
  }
  void operator()(typename Graph::edge_iterator ii) {
    GNode dst = graph.getEdgeDst(ii);
    auto& ddata = graph.getData(dst, unflag);
    
    unsigned oldDist;
    while (true) {
      oldDist = ddata;
      if (oldDist <= newDist)
        return;
      if (__sync_bool_compare_and_swap(&ddata, oldDist, newDist)) {
        nextBag.push(dst);
        count += 1 
          + std::distance(graph.edge_begin(dst, unflag),
                          graph.edge_end(dst, unflag));
        break;
      }
    }
  }
};


template<typename Graph, typename NodeBag>
BackwardProcess<Graph, NodeBag> mkBackward(Graph& g, NodeBag& n, Galois::GAccumulator<size_t>& c, int d) {
  return BackwardProcess<Graph, NodeBag>(g,n,c,d);
}

template<typename Graph, typename NodeBag>
ForwardProcess<Graph, NodeBag> mkForward(Graph& g, NodeBag& n, Galois::GAccumulator<size_t>& c, int d) {
  return ForwardProcess<Graph, NodeBag>(g,n,c,d);
}


int main(int argc, char** argv) {
  //Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  typedef typename Galois::Graph::LC_CSR_Graph<unsigned,void>
    ::template with_no_lockable<true>::type
    ::template with_numa_alloc<true>::type 
    InnerGraph;
  typedef typename Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;
  typedef Galois::Runtime::InsertBag<GNode> NodeBag;
  typedef std::pair<GNode,unsigned> WorkItem;
  typedef Galois::Runtime::InsertBag<WorkItem> WorkItemBag;

  Graph graph;
  int next = 0;
  int newDist = 1;
  int numForward = 0;
  int numBackward = 0;
  NodeBag bags[2];
  Galois::GAccumulator<size_t> count;

  {
    Galois::StatTimer TL("LoadTime", Galois::start_now);
    Galois::Graph::readGraph(graph, filename, transposeGraphName);
  }

  {
    Galois::StatTimer TI("InitTime", Galois::start_now);
    Galois::do_all_local(graph, [&graph] (GNode n) { graph.getData(n) = ~0; });
    graph.getData(source) = 0;
  }

  {
    Galois::StatTimer T("Time", Galois::start_now);
    if (std::distance(graph.edge_begin(source), graph.edge_end(source)) + 1 > (long) graph.sizeEdges() / 20) {
      Galois::do_all_local(graph, mkBackward(graph, bags[next], count, newDist), Galois::loopname("bfs_backward"), Galois::do_all_steal<true>());
      numBackward += 1;
    } else {
      auto r = graph.out_edges(source);
      Galois::do_all(r.begin(), r.end(), 
                     mkForward(graph, bags[next], count, newDist), Galois::loopname("bfs_forward"), Galois::do_all_steal<true>());
      numForward += 1;
    }
    
    while (!bags[next].empty()) {
      size_t nextSize = count.reduce();
      count.reset();
      int cur = next;
      next ^= 1;
      newDist++;
      if (nextSize > graph.sizeEdges() / 20) {
        //std::cout << "Dense " << nextSize << "\n";
        Galois::do_all_local(graph, mkBackward(graph, bags[next], count, newDist), Galois::loopname("bfs_backward"), Galois::do_all_steal<true>());
        numBackward += 1;
      } else { //if (numForward < 10 && numBackward == 0) {
        //std::cout << "Sparse " << nextSize << "\n";
        Galois::do_all_local(bags[cur], mkForward(graph, bags[next], count, newDist), Galois::loopname("bfs_forward"), Galois::do_all_steal<true>());
        numForward += 1;
      } //else {
      //   //std::cout << "Async " << nextSize << "\n";
      //   WorkItemBag asyncBag;
      //   Galois::do_all_local(bags[cur], [&asyncBag, newDist] (GNode n) { asyncBag.push(WorkItem(n, newDist)); } );
      //   Galois::for_each_local(asyncBag, mkAsync(graph));
      //   break;
      // }
      bags[cur].clear();
    }
    T.stop();
    std::cout << "Runtime: " << T.get() << "ms\n";;
  }
  
  int retval = 0;
  if (!skipVerify) {
    Galois::StatTimer TV("VerifyTime", Galois::start_now);
    // if (verify<true>(graph, source)) {
    //   std::cout << "Verification successful\n";
    // } else {
    //   std::cout << "Verification failed\n";
    //   retval = 1;
    // }
  }

  if (dumpResult) {
    std::cout << "{\"source\": " << source << ", \"values\" : [";
    bool first = true;
    for (auto ii : graph) {
      if (!first)
        std::cout << ", ";
      std::cout << graph.getData(ii);
      first = false;
    }
    std::cout << "]}\n";
  }

  Galois::Runtime::printStats();

  return retval;
}
