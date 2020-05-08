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

#ifndef GRAPH_COLORING_BASE_H
#define GRAPH_COLORING_BASE_H

#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/DoAllWrap.h"
#include "galois/graphs/Util.h"
#include "galois/graphs/Graph.h"
#include "galois/PerThreadContainer.h"
// #include "galois/Graph/FileGraph.h"

#include "galois/runtime/Profile.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <cstdio>

enum HeuristicType {
  FIRST_FIT,
  BY_ID,
  RANDOM,
  MIN_DEGREE,
  MAX_DEGREE,
};

namespace cll = llvm::cl;

static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);

static cll::opt<HeuristicType> heuristic(
    cll::desc("choose heuristic"),
    cll::values(clEnumVal(FIRST_FIT, "first fit, no priority"),
                clEnumVal(BY_ID, "order by ID modulo some constant"),
                clEnumVal(RANDOM, "uniform random within some small range"),
                clEnumVal(MIN_DEGREE, "order by min degree first"),
                clEnumVal(MAX_DEGREE, "order by max degree first"),
                clEnumValEnd),
    cll::init(FIRST_FIT));

static cll::opt<bool> useParaMeter("parameter",
                                   cll::desc("use parameter executor"),
                                   cll::init(false));

static const char* const name = "Graph Coloring";
static const char* const desc =
    "Greedy coloring of graphs with minimal number of colors";
static const char* const url = "graph-coloring";

template <typename G>
class GraphColoringBase : private boost::noncopyable {

protected:
  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  typedef galois::PerThreadVector<unsigned> PerThrdColorVec;
  typedef typename G::GraphNode GN;
  typedef typename G::node_data_type NodeData;

  G graph;
  PerThrdColorVec perThrdColorVec;

  void readGraph(void) {
    galois::graphs::readGraph(graph, filename);

    const size_t numNodes = graph.size();
    galois::GAccumulator<size_t> numEdges;

    galois::StatTimer t_init("initialization time: ");

    t_init.start();
    galois::on_each(
        [&](const unsigned tid, const unsigned numT) {
          size_t num_per = (numNodes + numT - 1) / numT;
          size_t beg     = tid * num_per;
          size_t end     = std::min(numNodes, (tid + 1) * num_per);

          auto it_beg = graph.begin();
          std::advance(it_beg, beg);

          auto it_end = it_beg;
          std::advance(it_end, (end - beg));

          for (; it_beg != it_end; ++it_beg) {
            // graph.getData (*it_beg, galois::MethodFlag::UNPROTECTED) =
            // NodeData (beg++);
            auto* ndptr =
                &(graph.getData(*it_beg, galois::MethodFlag::UNPROTECTED));
            ndptr->~NodeData();
            new (ndptr) NodeData(beg++);

            size_t deg = std::distance(
                graph.edge_begin(*it_beg, galois::MethodFlag::UNPROTECTED),
                graph.edge_end(*it_beg, galois::MethodFlag::UNPROTECTED));

            numEdges.update(deg);
          }
        },
        galois::loopname("initialize"));

    // color 0 is reserved as uncolored value
    // therefore, we put in at least 1 entry to handle the
    // corner case when a node with no neighbors is being colored
    for (unsigned i = 0; i < perThrdColorVec.numRows(); ++i) {
      perThrdColorVec.get(i).resize(1, 0);
    }

    t_init.stop();

    std::printf("Graph read with %zd nodes and %zd edges\n", numNodes,
                numEdges.reduceRO());
  }

  void colorNode(GN src) {

    auto& sd = graph.getData(src, galois::MethodFlag::UNPROTECTED);

    auto& forbiddenColors = perThrdColorVec.get();
    std::fill(forbiddenColors.begin(), forbiddenColors.end(), unsigned(-1));

    for (typename G::edge_iterator
             e     = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
             e_end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
         e != e_end; ++e) {

      GN dst   = graph.getEdgeDst(e);
      auto& dd = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      if (forbiddenColors.size() <= dd.color) {
        forbiddenColors.resize(dd.color + 1, unsigned(-1));
      }
      // std::printf ("Neighbor %d has color %d\n", dd.id, dd.color);

      forbiddenColors[dd.color] = sd.id;
    }

    bool colored = false;
    for (size_t i = 1; i < forbiddenColors.size(); ++i) {
      if (forbiddenColors[i] != sd.id) {
        sd.color = i;
        colored  = true;
        break;
      }
    }

    if (!colored) {
      sd.color = forbiddenColors.size();
    }

    // std::printf ("Node %d assigned color %d\n", sd.id, sd.color);
  }

  template <typename F>
  void assignPriorityHelper(const F& nodeFunc) {
    galois::do_all_choice(
        galois::runtime::makeLocalRange(graph),
        [&](GN node) { nodeFunc(node); }, "assign-priority",
        galois::chunk_size<DEFAULT_CHUNK_SIZE>());
  }

  static const unsigned MAX_LEVELS = 100;
  static const unsigned SEED       = 10;

  struct RNG {
    std::uniform_int_distribution<unsigned> dist;
    std::mt19937 eng;

    RNG(void) : dist(0, MAX_LEVELS), eng() { this->eng.seed(SEED); }

    unsigned operator()(void) { return dist(eng); }
  };

  void assignPriority(void) {

    auto byId = [&](GN node) {
      auto& nd    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      nd.priority = nd.id % MAX_LEVELS;
    };

    galois::substrate::PerThreadStorage<RNG> perThrdRNG;

    auto randPri = [&](GN node) {
      auto& rng   = *(perThrdRNG.getLocal());
      auto& nd    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      nd.priority = rng();
    };

    auto minDegree = [&](GN node) {
      auto& nd = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      nd.priority =
          std::distance(graph.edge_begin(node, galois::MethodFlag::UNPROTECTED),
                        graph.edge_end(node, galois::MethodFlag::UNPROTECTED));
    };

    const size_t numNodes = graph.size();
    auto maxDegree        = [&](GN node) {
      auto& nd = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      nd.priority =
          numNodes -
          std::distance(graph.edge_begin(node, galois::MethodFlag::UNPROTECTED),
                        graph.edge_end(node, galois::MethodFlag::UNPROTECTED));
    };

    galois::StatTimer t_priority("priority assignment time: ");

    t_priority.start();

    switch (heuristic) {
    case FIRST_FIT:
      // do nothing
      break;

    case BY_ID:
      assignPriorityHelper(byId);
      break;

    case RANDOM:
      assignPriorityHelper(randPri);
      break;

    case MIN_DEGREE:
      assignPriorityHelper(minDegree);
      break;

    case MAX_DEGREE:
      assignPriorityHelper(maxDegree);
      break;

    default:
      std::abort();
    }

    t_priority.stop();
  }

  void verify(void) {
    if (skipVerify) {
      return;
    }

    galois::StatTimer t_verify("verification time: ");

    t_verify.start();

    galois::GReduceLogicalOR foundError;
    galois::GReduceMax<unsigned> maxColor;

    galois::do_all_choice(
        galois::runtime::makeLocalRange(graph),
        [&](GN src) {
          auto& sd = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          if (sd.color == 0) {
            std::fprintf(stderr, "ERROR: src %d found uncolored\n", sd.id);
            foundError.update(true);
          }
          for (typename G::edge_iterator
                   e = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
                   e_end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
               e != e_end; ++e) {

            GN dst   = graph.getEdgeDst(e);
            auto& dd = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            if (sd.color == dd.color) {
              foundError.update(true);
              std::fprintf(stderr,
                           "ERROR: nodes %d and %d have the same color\n",
                           sd.id, dd.id);
            }
          }

          maxColor.update(sd.color);
        },
        "check-coloring", galois::chunk_size<DEFAULT_CHUNK_SIZE>());

    std::printf("Graph colored with %d colors\n", maxColor.reduce());

    t_verify.stop();

    if (foundError.reduceRO()) {
      GALOIS_DIE("ERROR! verification failed!\n");
    } else {
      printf("OK! verification succeeded!\n");
    }
  }

  virtual void colorGraph(void) = 0;

public:
  void run(int argc, char* argv[]) {
    LonestarStart(argc, argv, name, desc, url);
    galois::StatManager sm;

    readGraph();

    galois::preAlloc(galois::getActiveThreads() +
                     2 * sizeof(NodeData) * graph.size() /
                         galois::runtime::pagePoolSize());
    galois::reportPageAlloc("MeminfoPre");

    galois::StatTimer t;

    t.start();
    colorGraph();
    t.stop();

    galois::reportPageAlloc("MeminfoPost");

    verify();
  }
};
#endif // GRAPH_COLORING_BASE_H
