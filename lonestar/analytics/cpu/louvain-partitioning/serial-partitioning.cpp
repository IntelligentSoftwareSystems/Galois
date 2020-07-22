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
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"
#include "Lonestar/BFS_SSSP.h"

#include "llvm/Support/CommandLine.h"
#include "louvain-partitioning.h"
#include <iostream>
#include <deque>
#include <type_traits>
#include <map>

constexpr static const unsigned CHUNK_SIZE      = 256U;
constexpr static const ptrdiff_t EDGE_TILE_SIZE = 256;

using BFS = BFS_SSSP<Graph, unsigned int, false, EDGE_TILE_SIZE>;

using Dist           = BFS::Dist;
using OutEdgeRangeFn = BFS::OutEdgeRangeFn;

struct NodePushWrap {

  template <typename C>
  void operator()(C& cont, const GNode& n, const char* const) const {
    (*this)(cont, n);
  }

  template <typename C>
  void operator()(C& cont, const GNode& n) const {
    cont.push(n);
  }
};

// check if moving node n to part p improves the edge cut
bool checkIfFeasible(GNode n, Graph& graph, uint32_t p) {

  uint32_t zeros = 0;
  uint32_t ones  = 0;

  for (auto edge : graph.edges(n)) {

    auto dst = graph.getEdgeDst(edge);

    if (dst == n || graph.getData(dst).getDegree() <= 1)
      continue;

    uint32_t part = graph.getData(dst).getPart();
    if (part == 0)
      zeros += graph.getEdgeData(edge).weight;
    else
      ones += graph.getEdgeData(edge).weight;
  }

  if (p == 1) {
    if (zeros > ones)
      return false;
    else
      return true;
  } else {
    if (ones > zeros)
      return false;
    else
      return true;
  }
}

template <bool CONCURRENT, typename T, typename P, typename R>
void syncAlgoRefine(Graph& graph, uint32_t p, const P& pushWrap,
                    const R& edgeRange) {

  using Cont = typename std::conditional<CONCURRENT, galois::InsertBag<T>,
                                         galois::SerStack<T>>::type;
  using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                         galois::StdForEach>::type;

  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

  Loop loop;

  auto curr = std::make_unique<Cont>();
  auto next = std::make_unique<Cont>();

  Dist nextLevel = 0U;

  for (auto n : graph)
    if (graph.getData(n).getPart() == p) {
      graph.getData(n, flag).distance = 0U;
      pushWrap(*next, n, "parallel");
    }

  assert(!next->empty());

  while (!next->empty()) {

    std::swap(curr, next);
    next->clear();
    ++nextLevel;

    loop(
        galois::iterate(*curr),
        [&](const T& item) {
          for (auto e : edgeRange(item)) {
            auto dst = graph.getEdgeDst(e);

            if (dst == item)
              continue;

            auto& dstData = graph.getData(dst, flag);

            // only add dst if it belongs to the other partition
            if (dstData.getPart() == (1 - p)) {

              // check if moving dst to part p improves the edge cut or not
              bool check = checkIfFeasible(dst, graph, p);
              if (!check)
                continue;

              dstData.distance = nextLevel;
              pushWrap(*next, dst);
              dstData.setPart(p);
            }
          }
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("extendBoundary"));
  }
}

template <bool CONCURRENT>
void runAlgoRefine(Graph& graph, uint32_t p) {

  syncAlgoRefine<CONCURRENT, GNode>(graph, p, NodePushWrap(),
                                    OutEdgeRangeFn{graph});
}

void extendBoundary(Graph& graph, uint32_t p) { runAlgoRefine<true>(graph, p); }

uint64_t serial_partition(Graph& graph, const GNode& source,
                          uint64_t nodes_moved, uint64_t threshold) {

  uint64_t count = nodes_moved;
  count += graph.getData(source).getWeight();

  std::multimap<int64_t, GNode> myMap;
  galois::LargeArray<int64_t> gain;

  gain.allocateBlocked(graph.size() + 1);

  GNode curr = source;

  while (count < threshold || gain[curr] >= 0) {

    for (auto edge : graph.edges(curr)) {
      auto dst = graph.getEdgeDst(edge);

      // node is already in partition 1
      if (graph.getData(dst).getPart() == 1)
        continue;

      // calculate gain for dst
      int64_t g = 0;
      for (auto e1 : graph.edges(dst)) {
        auto dst1 = graph.getEdgeDst(e1);
        if (dst1 == dst)
          continue;

        int64_t wt = graph.getEdgeData(e1).weight;

        if (graph.getData(dst1).getPart() == 0)
          g -= wt;
        else
          g += wt;
      }

      gain[dst] = g;

      myMap.insert(std::make_pair(g, dst));
    }

    // find curr node
    while (myMap.begin() != myMap.end()) {
      curr      = myMap.rbegin()->second;
      int64_t g = myMap.rbegin()->first;

      // check if this node is in partition 0, and if so mark it as the next
      // node to process
      if (graph.getData(curr).getPart() == 0 && gain[curr] == g)
        break;

      auto it = myMap.end();
      it--;
      myMap.erase(it);
    }

    // no nodes left to move to partition 1 in this component
    if (myMap.begin() == myMap.end())
      break;

    graph.getData(curr).setPart(1);
    count += graph.getData(curr).getWeight();
  }

  gain.destroy();
  gain.deallocate();

  return count;
}

// contsruct a bi-partition
void partition(Graph& graph, double tol) {

  GNode source;
  galois::GAccumulator<uint64_t> nodes;

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    graph.getData(n).distance = BFS::DIST_INFINITY;
    graph.getData(n).setPart(0);
    nodes += graph.getData(n).getWeight();
  });

  uint64_t nodes_moved = 0;
  uint64_t threshold   = (double)nodes.reduce() * (1.0f / (2.0f + tol));

  for (auto n : graph) {
    if (graph.getData(n).getPart() == 0) {
      source                         = n;
      graph.getData(source).distance = 0;
      graph.getData(source).setPart(1);
      nodes_moved = serial_partition(graph, source, nodes_moved, threshold);
      if (nodes_moved >= threshold)
        break;
    }
  }
}
