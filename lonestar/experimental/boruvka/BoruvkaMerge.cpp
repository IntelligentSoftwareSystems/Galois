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

#include "UnionFind.h"

#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>

namespace cll = llvm::cl;

static const char* name = "Boruvka's Minimum Spanning Tree Algorithm";
static const char* desc = "Computes a minimum weight spanning tree of a graph";
static const char* url  = "mst";

static cll::opt<std::string>
    inputfile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<bool>
    verify_via_kruskal("verify",
                       cll::desc("Verify MST result via Serial Kruskal"),
                       cll::Optional, cll::init(false));

///////////////////////////////////////////////////////////////////////////////////////
typedef int NodeDataType;
typedef int EdgeDataType;

typedef galois::graphs::MorphGraph<NodeDataType, EdgeDataType, false> Graph;
typedef Graph::GraphNode GNode;

using Counter = galois::GAccumulator<size_t>;

#define BORUVKA_DEBUG

///////////////////////////////////////////////////////////////////////////////////////

std::ostream&
operator<<(std::ostream& out,
           std::tuple<NodeDataType, NodeDataType, EdgeDataType>& etpl) {
  out << "(" << std::get<0>(etpl) << ", " << std::get<1>(etpl) << ", "
      << std::get<2>(etpl) << ")";
  return out;
}

EdgeDataType runBodyParallel(Graph& graph) {

  auto indexer = [&](const GNode& n) {
    return std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                         graph.edge_begin(n, galois::MethodFlag::UNPROTECTED));
  };

  using namespace galois::worklists;
  typedef PerSocketChunkFIFO<64> PSchunk;
  typedef OrderedByIntegerMetric<decltype(indexer), PSchunk> OBIM;

  Counter MSTWeight;

  galois::StatTimer T;
  T.start();

  galois::for_each(
      galois::iterate(graph),

      [&graph, &MSTWeight](const GNode& src, auto& lwl) {
        if (graph.containsNode(src) == false)
          return;
        graph.getData(src, galois::MethodFlag::WRITE);
        GNode minNeighbor = 0;
#ifdef BORUVKA_DEBUG
        std::cout << "Processing " << graph.getData(src) << std::endl;
#endif
        EdgeDataType minEdgeWeight = std::numeric_limits<EdgeDataType>::max();
        // Acquire locks on neighborhood.
        for (auto dit : graph.edges(src, galois::MethodFlag::WRITE)) {
          graph.getData(graph.getEdgeDst(dit));
        }
        // Find minimum neighbor
        for (auto e_it : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
          EdgeDataType w =
              graph.getEdgeData(e_it, galois::MethodFlag::UNPROTECTED);
          assert(w >= 0);
          auto dst = graph.getEdgeDst(e_it);

          if (dst != src && w < minEdgeWeight) {
            minNeighbor   = graph.getEdgeDst(e_it);
            minEdgeWeight = w;
          }
        }
        // If there are no outgoing neighbors.
        if (minEdgeWeight == std::numeric_limits<EdgeDataType>::max()) {
          graph.removeNode(src, galois::MethodFlag::UNPROTECTED);
          return;
        }
#ifdef BORUVKA_DEBUG
        auto tpl = std::make_tuple(
            graph.getData(src, galois::MethodFlag::UNPROTECTED),
            graph.getData(minNeighbor, galois::MethodFlag::UNPROTECTED),
            minEdgeWeight);
        std::cout << " Boruvka edge added: " << tpl << std::endl;
#endif
        // Acquire locks on neighborhood of min neighbor.
        for (auto e_it : graph.edges(minNeighbor, galois::MethodFlag::WRITE)) {
          graph.getData(graph.getEdgeDst(e_it));
        }
        assert(minEdgeWeight >= 0);
        // update MST weight.
        MSTWeight += minEdgeWeight;

        using DstEdgePair = std::pair<GNode, EdgeDataType>;
        using DstEdgeSet  = galois::gstl::Set<DstEdgePair>;

        DstEdgeSet toAdd;

        for (auto dit :
             graph.edges(minNeighbor, galois::MethodFlag::UNPROTECTED)) {

          GNode dstNode = graph.getEdgeDst(dit);
          int edgeWeight =
              graph.getEdgeData(dit, galois::MethodFlag::UNPROTECTED);

          if (dstNode != src) { // Do not add the edge being contracted

            Graph::edge_iterator dup_edge =
                graph.findEdge(src, dstNode, galois::MethodFlag::UNPROTECTED);

            if (dup_edge !=
                graph.edge_end(src, galois::MethodFlag::UNPROTECTED)) {

              EdgeDataType dup_wt =
                  graph.getEdgeData(dup_edge, galois::MethodFlag::UNPROTECTED);
              graph.getEdgeData(dup_edge, galois::MethodFlag::UNPROTECTED) =
                  std::min<EdgeDataType>(edgeWeight, dup_wt);
              assert(std::min<EdgeDataType>(edgeWeight, dup_wt) >= 0);

            } else {
              toAdd.insert(DstEdgePair(dstNode, edgeWeight));
              assert(edgeWeight >= 0);
            }
          }
        }

        graph.removeNode(minNeighbor, galois::MethodFlag::UNPROTECTED);

        for (const auto& p : toAdd) {
          graph.getEdgeData(graph.addEdge(
              src, p.first, galois::MethodFlag::UNPROTECTED)) = p.second;
        }

        lwl.push(src);
      },
      galois::wl<OBIM>(indexer), galois::loopname("Main"));

  T.stop();

  return MSTWeight.reduce();
}

void makeGraph(Graph& graph) {

  galois::graphs::readGraph(graph, inputfile.c_str());

  int id = 0;

  size_t numEdges = 0;

  for (GNode n : graph) {
    graph.getData(n) = id++;

    numEdges += std::distance(graph.edge_begin(n), graph.edge_end(n));
  }
  numEdges /= 2;

  // TODO: sort graph by edge data

  std::cout << inputfile << " read with nodes = " << graph.size()
            << ", edges = " << numEdges << std::endl;
}

////////////////////////////Kruskal////////////////////////////////////////////////
EdgeDataType runKruskal(Graph& graph) {

  struct KEdgeTuple {
    NodeDataType src;
    NodeDataType dst;
    EdgeDataType wt;
  };

  auto kEdgeCmp = [](const KEdgeTuple& o1, const KEdgeTuple& o2) {
    return (o1.wt == o2.wt) ? o1.src < o2.src : o1.wt < o2.wt;
  };

  std::vector<KEdgeTuple> kruskalEdges;

  for (GNode src : graph) {

    auto sd = graph.getData(src);

    for (auto e : graph.edges(src)) {
      GNode dst = graph.getEdgeDst(e);
      auto dd   = graph.getData(dst);

      auto wt = graph.getEdgeData(e);

      kruskalEdges.push_back(KEdgeTuple{sd, dd, wt});
    }
  }

  size_t num_nodes = graph.size();

  std::sort(kruskalEdges.begin(), kruskalEdges.end(), kEdgeCmp);

  UnionFind<NodeDataType, -1> uf(num_nodes);

  size_t mst_size = 0;

  EdgeDataType mst_sum = 0;

  for (size_t i = 0; i < kruskalEdges.size(); ++i) {

    const KEdgeTuple& e = kruskalEdges[i];
    NodeDataType src    = uf.uf_find(e.src);
    NodeDataType dst    = uf.uf_find(e.dst);

    if (src != dst) {
      uf.uf_union(src, dst);
      mst_sum += e.wt;
      mst_size++;

#ifdef BORUVKA_DEBUG
      auto tpl = std::make_tuple(e.src, e.dst, e.wt);
      std::cout << "Kruskal Edge added: " << tpl << std::endl;
#endif
    }

    if (mst_size >= num_nodes - 1)
      return mst_sum;
  }
  return -1;
}

//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph;

  makeGraph(graph);

  EdgeDataType krusk_wt = 0;

  if (!skipVerify) { // run kruskal before Boruvka morphs the graph into 1 node
    krusk_wt = runKruskal(graph);
  }

  galois::preAlloc(galois::runtime::numPagePoolAllocTotal() * 10);
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  T.start();

  EdgeDataType mst_wt = runBodyParallel(graph);

  T.stop();

  std::cout << "Boruvka MST Result is " << mst_wt << "\n";

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    if (krusk_wt != mst_wt) {
      std::cerr
          << "ERROR: Boruvka's mst weight doesn't match Kruskal's mst weight"
          << std::endl;
      std::cerr << "Boruvka's weight = " << mst_wt
                << ", Kruskal's weight = " << krusk_wt << std::endl;

      std::abort();
    }
  }

  return 0;
}
