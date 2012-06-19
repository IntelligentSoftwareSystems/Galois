/** Maximal independent set application -*- C++ -*-
 * @file
 *
 * A simple spanning tree algorithm to demostrate the Galois system.
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 */
#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "boost/optional.hpp"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

namespace cll = llvm::cl;

const char* name = "Maximal Independent Set";
const char* desc = "Compute a maximal independent set (not maximum) of nodes in a graph";
const char* url = NULL;

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

enum DetAlgo {
  nondet,
  detBase,
  detPrefix,
  detDisjoint
};

#ifdef GALOIS_USE_DET
static cll::opt<DetAlgo> detAlgo(cll::desc("Deterministic algorithm:"),
    cll::values(
      clEnumVal(nondet, "Non-deterministic"),
      clEnumVal(detBase, "Base execution"),
      clEnumVal(detPrefix, "Prefix execution"),
      clEnumVal(detDisjoint, "Disjoint execution"),
      clEnumValEnd), cll::init(detBase));
#endif

enum MatchFlag {
  UNMATCHED, OTHER_MATCHED, MATCHED
};

struct Node {
  MatchFlag flag; 
  Node() : flag(UNMATCHED) { }
};

typedef Galois::Graph::LC_Linear_Graph<Node,void> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

template<int Version=detBase>
struct Process {
  typedef int tt_does_not_need_parallel_push;
  typedef int tt_needs_per_iter_alloc; // For LocalState

  struct LocalState {
    bool mod;
    LocalState(Process<Version>* self, Galois::PerIterAllocTy& alloc): mod(false) { }
  };

  bool build(GNode src) {
    for (Graph::edge_iterator ii = graph.edge_begin(src),
        ei = graph.edge_end(src); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      if (data.flag != UNMATCHED)
        return false;
    }

    Node& me = graph.getData(src, Galois::NONE);
    if (me.flag != UNMATCHED)
      return false;
    return true;
  }

  void modify(GNode src) {
    Node& me = graph.getData(src, Galois::NONE);
    for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE),
        ei = graph.edge_end(src, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Galois::NONE);
      data.flag = OTHER_MATCHED;
    }

    me.flag = MATCHED;
  }

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
    bool* modp;
#ifdef GALOIS_USE_DET
    if (Version == detDisjoint) {
      bool used;
      LocalState* localState = (LocalState*) ctx.getLocalState(used);
      modp = &localState->mod;
      if (used) {
        if (*modp)
          modify(src);
        return;
      }
    }
#endif

    if (Version == detDisjoint) {
      *modp = build(src);
    } else {
      bool mod = build(src);
      if (Version == detPrefix)
        return;
      else
        graph.getData(src, Galois::WRITE); // Failsafe point
      if (mod)
        modify(src);
    }
  }
};

struct is_bad {
  bool operator()(GNode n) const {
    Node& me = graph.getData(n);
    if (me.flag == MATCHED) {
      for (Graph::edge_iterator ii = graph.edge_begin(n),
          ei = graph.edge_end(n); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& data = graph.getData(dst);
        if (dst != n && data.flag == MATCHED) {
          std::cerr << "double match\n";
          return true;
        }
      }
    } else if (me.flag == UNMATCHED) {
      bool ok = false;
      for (Graph::edge_iterator ii = graph.edge_begin(n),
          ei = graph.edge_end(n); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& data = graph.getData(dst);
        if (data.flag != UNMATCHED) {
          ok = true;
        }
      }
      if (!ok) {
        std::cerr << "not maximal\n";
        return true;
      }
    }
    return false;
  }
};

struct is_matched {
  bool operator()(const GNode& n) const {
    return graph.getData(n).flag == MATCHED;
  }
};

bool verify() {
  return Galois::find_if(graph.begin(), graph.end(), is_bad()) == graph.end();
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  graph.structureFromFile(filename.c_str());

  Galois::StatTimer T;
  T.start();
  using namespace GaloisRuntime::WorkList;
  typedef BulkSynchronous<dChunkedLIFO<256> > BSWL;
#ifdef GALOIS_USE_EXP
  typedef BulkSynchronousInline<> BSInline;
#else
  typedef BSWL BSInline;
#endif

#ifdef GALOIS_USE_DET
  switch (detAlgo) {
    case nondet: 
      Galois::for_each(graph.begin(), graph.end(), Process<>()); break;
    case detBase:
      Galois::for_each_det<false>(graph.begin(), graph.end(), Process<>()); break;
    case detPrefix:
      Galois::for_each_det<false>(graph.begin(), graph.end(), Process<detPrefix>(), Process<>());
      break;
    case detDisjoint:
      Galois::for_each_det<true>(graph.begin(), graph.end(), Process<detDisjoint>()); break;
    default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
  }
#else
  //Galois::for_each<BSInline>(graph.begin(), graph.end(), Process());
  Galois::for_each(graph.begin(), graph.end(), Process());
#endif
  T.stop();

  std::cout << "Cardinality of maximal independent set: " 
    << Galois::count_if(graph.begin(), graph.end(), is_matched()) 
    << "\n";

  if (!skipVerify && !verify()) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}

