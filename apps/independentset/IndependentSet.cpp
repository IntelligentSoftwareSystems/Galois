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
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

const char* name = "Maximal Independent Set";
const char* desc = "Computes a maximal independent set (not maximum) of nodes in a graph";
const char* url = "independent_set";

enum Algo {
  serial,
  nondet,
  detBase,
  detPrefix,
  detDisjoint,
  orderedBase,
};

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(nondet, "Non-deterministic"),
      clEnumVal(detBase, "Base deterministic execution"),
      clEnumVal(detPrefix, "Prefix deterministic execution"),
      clEnumVal(detDisjoint, "Disjoint deterministic execution"),
      clEnumVal(orderedBase, "Base ordered execution"),
      clEnumValEnd), cll::init(nondet));

enum MatchFlag {
  UNMATCHED, OTHER_MATCHED, MATCHED
};

struct Node {
  unsigned int id;
  MatchFlag flag; 
  MatchFlag pendingFlag; 
  Node() : flag(UNMATCHED), pendingFlag(UNMATCHED) { }
};

typedef Galois::Graph::LC_InlineEdge_Graph<Node,void>
  ::with_numa_alloc<true>::type
  ::with_compressed_node_ptr<true>::type Graph;

typedef Graph::GraphNode GNode;

Graph graph;

//! Basic operator for default and deterministic scheduling
template<int Version=detBase>
struct Process {
  typedef int tt_does_not_need_parallel_push;
  typedef int tt_needs_per_iter_alloc; // For LocalState

  struct LocalState {
    bool mod;
    LocalState(Process<Version>& self, Galois::PerIterAllocTy& alloc): mod(false) { }
  };
  typedef LocalState GaloisDeterministicLocalState;
  static_assert(Galois::has_deterministic_local_state<Process>::value, "Oops");

  template<Galois::MethodFlag Flag>
  bool build(GNode src) {
    Node& me = graph.getData(src, Flag);
    if (me.flag != UNMATCHED)
      return false;

    for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::MethodFlag::NONE),
        ei = graph.edge_end(src, Galois::MethodFlag::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Flag);
      if (data.flag == MATCHED)
        return false;
    }

    return true;
  }

  void modify(GNode src) {
    Node& me = graph.getData(src, Galois::MethodFlag::NONE);
    for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::MethodFlag::NONE),
        ei = graph.edge_end(src, Galois::MethodFlag::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Galois::MethodFlag::NONE);
      data.flag = OTHER_MATCHED;
    }

    me.flag = MATCHED;
  }

  //! Serial operator
  void operator()(GNode src) {
    if (build<Galois::MethodFlag::NONE>(src))
      modify(src);
  }

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
    bool* modp;
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

    if (Version == detDisjoint) {
      *modp = build<Galois::MethodFlag::ALL>(src);
    } else {
      bool mod = build<Galois::MethodFlag::ALL>(src);
      if (Version == detPrefix)
        return;
      else
        graph.getData(src, Galois::MethodFlag::WRITE); // Failsafe point
      if (mod)
        modify(src);
    }
  }
};

template<bool prefix>
struct OrderedProcess {
  typedef int tt_does_not_need_parallel_push;

  Process<> process;

  template<typename C>
  void operator()(GNode src, C& ctx) {
    (*this)(src);
  }

  void operator()(GNode src) {
    if (prefix) {
      graph.edge_begin(src, Galois::MethodFlag::ALL);
    } else {
      if (process.build<Galois::MethodFlag::NONE>(src))
        process.modify(src);
    }
  }
};

struct Compare {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, Galois::MethodFlag::NONE).id < graph.getData(b, Galois::MethodFlag::NONE).id;
  }
};

struct GaloisAlgo {
  void operator()() {
#ifdef GALOIS_USE_EXP
    typedef Galois::WorkList::BulkSynchronousInline<> WL;
#else
    typedef Galois::WorkList::dChunkedFIFO<256> WL;
#endif

    switch (algo) {
      case nondet: 
        Galois::for_each<WL>(graph.begin(), graph.end(), Process<>());
        break;
      case detBase:
        Galois::for_each_det(graph.begin(), graph.end(), Process<>());
        break;
      case detPrefix:
        Galois::for_each_det(graph.begin(), graph.end(), Process<detPrefix>(), Process<>());
        break;
      case detDisjoint:
        Galois::for_each_det(graph.begin(), graph.end(), Process<detDisjoint>());
        break;
      case orderedBase:
        Galois::for_each_ordered(graph.begin(), graph.end(), Compare(),
            OrderedProcess<true>(), OrderedProcess<false>());
        break;
      default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
    }
  }
};

struct SerialAlgo {
  void operator()() {
    std::for_each(graph.begin(), graph.end(), Process<>());
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
  return Galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad()) == graph.end();
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::Graph::readGraph(graph, filename);

  unsigned int id = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, ++id)
    graph.getData(*ii).id = id;
  
  // XXX Test if this matters
  Galois::preAlloc(numThreads + (graph.size() * sizeof(Node) * numThreads / 8) / Galois::Runtime::MM::pageSize);
  Galois::reportPageAlloc("MeminfoPre");
  Galois::StatTimer T;
  T.start();
  switch (algo) {
    case serial: SerialAlgo()(); break;
    default: GaloisAlgo()(); break;
  }
  T.stop();
  Galois::reportPageAlloc("MeminfoPost");

  std::cout << "Cardinality of maximal independent set: " 
    << Galois::ParallelSTL::count_if(graph.begin(), graph.end(), is_matched()) 
    << "\n";

  if (!skipVerify && !verify()) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}
