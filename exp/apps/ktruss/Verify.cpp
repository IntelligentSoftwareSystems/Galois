/** Verification for k-truss -*- C++ -*-
 * @example Verify.cpp
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
 * Verify whether an edgelist from an undirected graph is a k-truss
 *
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <deque>
#include <algorithm>
#include <fstream>

namespace cll = llvm::cl;

static const char* name = "Verify K-truss";
static const char* desc = nullptr;
static const char* url = nullptr;

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> trussFile("trussFile", cll::desc("edgelist to for the trusses"), cll::Required);
static cll::opt<unsigned int> trussNum("trussNum", cll::desc("verify trussNum-trusses"), cll::Required);

static const unsigned int valid = 0x0;
static const unsigned int removed = 0x1;

// edge weight: (# triangles supported << 1) | removal
//   set LSB of an edge weight to indicate the removal of the edge.
//   << 1 to track # triangles an edge supports, 
//   >> 1 when computing edge supports
typedef Galois::Graph::LC_CSR_Graph<void, unsigned int>
  ::template with_numa_alloc<true>::type 
  ::template with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

typedef std::pair<GNode, GNode> Edge;
typedef Galois::InsertBag<Edge> EdgeVec;

template<typename Graph>
void initialize(Graph& g) {
  g.sortAllEdgesByDst();

  // initializa all edges to removed
  Galois::do_all_local(
    g, 
    [&g] (typename Graph::GraphNode N) { 
      for (auto e: g.edges(N)) {
        g.getEdgeData(e) = removed;
      }
    },
    Galois::do_all_steal<true>()
  );
}

// TODO: can we read in edges in parallel?
template<typename Graph>
void readTruss(Graph& g, EdgeVec& w) {
  std::ifstream edgelist(trussFile);
  if (!edgelist.is_open()) {
    std::string errMsg = "Failed to open " + trussFile;
    GALOIS_DIE(errMsg);
  }

  unsigned int n1, n2;
  while (edgelist >> n1 >> n2) {
    g.getEdgeData(g.findEdgeSortedByDst(n1, n2)) = valid;
    g.getEdgeData(g.findEdgeSortedByDst(n2, n1)) = valid;
    if (n1 < n2) {
      w.push_back(std::make_pair(n1, n2));
    } else {
      w.push_back(std::make_pair(n2, n1));
    }
  }
}

template<typename Graph>
void printGraph(Graph& g) {
  for (auto n: g) {
    std::cout << "node " << n << std::endl;
    for (auto e: g.edges(n)) {
      auto d = g.getEdgeDst(e);
      if (d >= n) continue;
      std::cout << "  edge to " << d << ((g.getEdgeData(e) & removed) ? " removed" : "") << std::endl;
    }
  }
}

template<typename G>
size_t countValidNodes(G& g) {
  Galois::GAccumulator<size_t> numNodes;

  Galois::do_all_local(g, 
    [&g, &numNodes] (GNode n) {
      for (auto e: g.edges(n)) {
        if (!(g.getEdgeData(e) & removed)) {
          numNodes += 1;
          break;
        }
      }
    },
    Galois::do_all_steal<true>()
  );

  return numNodes.reduce();
}

template<typename G>
size_t countValidEqual(G& g, typename G::GraphNode src, typename G::GraphNode dst) {
  size_t retval = 0;
  auto srcI = g.edge_begin(src, Galois::MethodFlag::UNPROTECTED), 
    srcE = g.edge_end(src, Galois::MethodFlag::UNPROTECTED), 
    dstI = g.edge_begin(dst, Galois::MethodFlag::UNPROTECTED), 
    dstE = g.edge_end(dst, Galois::MethodFlag::UNPROTECTED);

  while (true) {
    // find the first valid edge
    while (srcI != srcE && (g.getEdgeData(srcI) & removed)) {
      ++srcI;
    }
    while (dstI != dstE && (g.getEdgeData(dstI) & removed)) {
      ++dstI;
    }

    if (srcI == srcE || dstI == dstE) {
      return retval;
    }

    // check for intersection
    auto sN = g.getEdgeDst(srcI), dN = g.getEdgeDst(dstI);
    if (sN < dN) {
      ++srcI;
    } else if (dN < sN) {
      ++dstI;
    } else {
      retval += 1;
      ++srcI;
      ++dstI;
    }
  }
  return retval;
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if (2 > trussNum) {
    std::cerr << "trussNum >= 2" << std::endl;
    return -1;
  }

  Graph g;
  EdgeVec work, unsupported;

  Galois::Graph::readGraph(g, filename);
  initialize(g);
  readTruss(g, work);
//  printGraph(g);

  std::cout << "Read " << g.size() << " nodes" << std::endl;
  std::cout << "Valid truss nodes: " << countValidNodes(g) << std::endl;
  std::cout << "Valid truss edges: " << std::distance(work.begin(), work.end()) << std::endl;

  Galois::do_all_local(work, 
    [&g, &unsupported] (Edge e) {
       if (countValidEqual(g, e.first, e.second) < trussNum-2) {
         unsupported.push_back(e);
       }
    },
    Galois::do_all_steal<true>()
  );

  if (0 == std::distance(unsupported.begin(), unsupported.end())) {
    std::cout << "Verification successful" << std::endl;
  } else {
    for (auto e: unsupported) {
      std::cerr << "(" << e.first << ", " << e.second << ") unsupported" << std::endl;
    }
    GALOIS_DIE("Verification failed!");
  }

  return 0;
}
