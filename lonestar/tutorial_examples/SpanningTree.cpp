/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/UnionFind.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <algorithm>
#include <iostream>

namespace cll = llvm::cl;

const char* name = "Spanning Tree Algorithm";
const char* desc = "Computes the spanning forest of a graph";

enum Algo { demo, asynchronous, blockedasync };

static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo>
    algo("algo", cll::desc("Choose an algorithm:"),
         cll::values(clEnumVal(demo, "Demonstration algorithm"),
                     clEnumVal(asynchronous, "Asynchronous"),
                     clEnumVal(blockedasync, "Blocked Asynchronous")),
         cll::init(blockedasync));

struct Node : public galois::UnionFindNode<Node> {
  Node() : galois::UnionFindNode<Node>(const_cast<Node*>(this)) {}
  Node* component() { return find(); }
  void setComponent(Node* n) { m_component = n; }
};

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << &n << "]";
  return os;
}

typedef galois::graphs::LC_Linear_Graph<Node, void>::with_numa_alloc<true>::type
    Graph;
typedef Graph::GraphNode GNode;
typedef std::pair<GNode, GNode> Edge;

struct BlockedWorkItem {
  GNode src;
  Graph::edge_iterator start;
};

template <bool MakeContinuation, int Limit>
auto specialized_process(Graph& graph, galois::InsertBag<Edge>& mst)
    -> decltype(auto) {
  return
      [&](const GNode& src, const Graph::edge_iterator& start, auto& pusher) {
        Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
        int count   = 1;
        for (Graph::edge_iterator
                 ii = start,
                 ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
             ii != ei; ++ii, ++count) {
          GNode dst   = graph.getEdgeDst(ii);
          Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
          if (sdata.merge(&ddata)) {
            mst.push(std::make_pair(src, dst));
            if (Limit == 0 || count != Limit) {
              continue;
            }
          }

          if (MakeContinuation || (Limit != 0 && count == Limit)) {
            BlockedWorkItem item = {src, ii + 1};
            pusher.push(item);
            break;
          }
        }
      };
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, nullptr, nullptr);

  Graph graph;

  galois::InsertBag<Edge> mst;

  galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();
  galois::graphs::readGraph(graph, inputFilename);
  std::cout << "Num nodes: " << graph.size() << "\n";
  Tinitial.stop();

  //! Normalize component by doing find with path compression
  auto Normalize = [&](const GNode& src) {
    Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    sdata.setComponent(sdata.findAndCompress());
  };

  // galois::preAlloc(numThreads + graph.size() /
  // galois::runtime::MM::hugePageSize * 60);
  galois::reportPageAlloc("MeminfoPre");
  galois::StatTimer T;
  T.start();
  switch (algo) {
  /**
   * Construct a spanning forest via a modified BFS algorithm. Intended as a
   * simple introduction to the Galois system and not intended to particularly
   * fast. Restrictions: graph must be strongly connected. In this case, the
   * spanning tree is over the undirected graph created by making the directed
   * graph symmetric.
   */
  case demo: {
    Graph::iterator ii = graph.begin(), ei = graph.end();
    if (ii != ei) {
      Node* root = &graph.getData(*ii);
      galois::for_each(
          galois::iterate({*ii}),
          [&](GNode src, auto& ctx) {
            for (auto ii : graph.edges(src, galois::MethodFlag::WRITE)) {
              GNode dst   = graph.getEdgeDst(ii);
              Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              if (ddata.component() == root)
                continue;
              ddata.setComponent(root);
              mst.push(std::make_pair(src, dst));
              ctx.push(dst);
            }
          },
          galois::loopname("DemoAlgo"),
          galois::wl<galois::worklists::PerSocketChunkFIFO<32>>());
    }
  } break;

  case asynchronous:
    /**
     * Like asynchronous connected components algorithm.
     */
    {
      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
            for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
              GNode dst   = graph.getEdgeDst(ii);
              Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              if (sdata.merge(&ddata)) {
                mst.push(std::make_pair(src, dst));
              }
            }
          },
          galois::loopname("Merge"), galois::steal());
      galois::do_all(galois::iterate(graph), Normalize,
                     galois::loopname("Normalize"));
    }
    break;

  case blockedasync:
    /**
     * Improve performance of async algorithm by following machine topology.
     */
    {
      galois::InsertBag<BlockedWorkItem> items;
      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            Graph::edge_iterator start =
                graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
            if (galois::substrate::ThreadPool::getSocket() == 0) {
              specialized_process<true, 0>(graph, mst)(src, start, items);
            } else {
              specialized_process<true, 1>(graph, mst)(src, start, items);
            }
          },
          galois::loopname("Initialize"));
      galois::for_each(
          galois::iterate(items),
          [&](const BlockedWorkItem& i, auto& ctx) {
            specialized_process<true, 0>(graph, mst)(i.src, i.start, ctx);
          },
          galois::loopname("Merge"), galois::disable_conflict_detection(),
          galois::wl<galois::worklists::PerSocketChunkFIFO<128>>());
      //! Normalize component by doing find with path compression
      galois::do_all(galois::iterate(graph), Normalize,
                     galois::loopname("Normalize"));
    }
    break;

  default:
    std::cerr << "Unknown algo: " << algo << "\n";
  }
  T.stop();
  galois::reportPageAlloc("MeminfoPost");

  /* Verification Routines */
  auto is_bad_graph = [&](const GNode& n) {
    Node& me = graph.getData(n);
    for (auto ii : graph.edges(n)) {
      GNode dst  = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      if (me.component() != data.component()) {
        std::cerr << "not in same component: " << me << " and " << data << "\n";
        return true;
      }
    }
    return false;
  };

  auto is_bad_mst = [&](const Edge& e) {
    return graph.getData(e.first).component() !=
           graph.getData(e.second).component();
  };

  auto checkAcyclic = [&]() {
    galois::GAccumulator<unsigned> roots;
    galois::do_all(galois::iterate(graph), [&](const GNode& n) {
      Node& data = graph.getData(n);
      if (data.component() == &data)
        roots += 1;
    });
    unsigned numRoots = roots.reduce();
    unsigned numEdges = std::distance(mst.begin(), mst.end());
    if (graph.size() - numRoots != numEdges) {
      std::cerr << "Generated graph is not a forest. "
                << "Expected " << graph.size() - numRoots << " edges but "
                << "found " << numEdges << "\n";
      return false;
    }
    std::cout << "Num trees: " << numRoots << "\n";
    std::cout << "Tree edges: " << numEdges << "\n";
    return true;
  };

  auto verify = [&]() {
    if (galois::ParallelSTL::find_if(graph.begin(), graph.end(),
                                     is_bad_graph) == graph.end()) {
      if (galois::ParallelSTL::find_if(mst.begin(), mst.end(), is_bad_mst) ==
          mst.end()) {
        return checkAcyclic();
      }
    }
    return false;
  };

  if (!skipVerify && !verify()) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}
