/** Maximal independent set application -*- C++ -*-
 * @file
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"


#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <type_traits>

const char* name = "Maximal Independent Set";
const char* desc = "Computes a maximal independent set (not maximum) of nodes in a graph";
const char* url = "independent_set";

enum Algo {
  serial,
  pull,
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
      clEnumVal(pull, "Pull-based (deterministic)"),
      clEnumVal(nondet, "Non-deterministic"),
      clEnumVal(detBase, "Base deterministic execution"),
      clEnumVal(detPrefix, "Prefix deterministic execution"),
      clEnumVal(detDisjoint, "Disjoint deterministic execution"),
      clEnumVal(orderedBase, "Base ordered execution"),
      clEnumValEnd), cll::init(nondet));

enum MatchFlag: char {
  UNMATCHED, OTHER_MATCHED, MATCHED
};

struct Node {
  MatchFlag flag; 
  Node() : flag(UNMATCHED) { }
};


struct SerialAlgo {
  typedef Galois::Graph::LC_CSR_Graph<Node,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  void operator()(Graph& graph) {
    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      if (findUnmatched(graph, *ii))
        match(graph, *ii);
    }
  }

  bool findUnmatched(Graph& graph, GNode src) {
    Node& me = graph.getData(src);
    if (me.flag != UNMATCHED)
      return false;

    for (auto ii : graph.edges(src)) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      if (data.flag == MATCHED)
        return false;
    }

    return true;
  }

  void match(Graph& graph, GNode src) {
    Node& me = graph.getData(src);
    for (auto ii : graph.edges(src)) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      data.flag = OTHER_MATCHED;
    }

    me.flag = MATCHED;
  }
};

//! Basic operator for default and deterministic scheduling
template<int Version=detBase>
struct Process {
  typedef typename Galois::Graph::LC_CSR_Graph<Node,void>
    ::template with_numa_alloc<true>::type Graph;

  typedef typename Graph::GraphNode GNode;

  struct LocalState {
    bool mod;
    LocalState(Process<Version>& self, Galois::PerIterAllocTy& alloc): mod(false) { }
  };

  struct DeterministicId {
    uintptr_t operator()(const GNode& x) const {
      return x;
    }
  };

  typedef std::tuple<
    Galois::does_not_need_push<>,
    Galois::needs_per_iter_alloc<>,
    Galois::has_deterministic_id<DeterministicId>,
    Galois::has_deterministic_local_state<LocalState>
  > function_traits;

  Graph& graph;

  Process(Graph& g): graph(g) { }

  template<Galois::MethodFlag Flag>
  bool build(GNode src) {
    Node& me = graph.getData(src, Flag);
    if (me.flag != UNMATCHED)
      return false;

    for (auto ii : graph.edges(src, Galois::MethodFlag::UNPROTECTED)) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Flag);
      if (data.flag == MATCHED)
        return false;
    }

    return true;
  }

  void modify(GNode src) {
    Node& me = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
    for (auto ii : graph.edges(src, Galois::MethodFlag::UNPROTECTED)) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
      data.flag = OTHER_MATCHED;
    }

    me.flag = MATCHED;
  }

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
    bool* modp;
    if (Version == detDisjoint) {
      LocalState* localState = (LocalState*) ctx.getLocalState();
      modp = &localState->mod;
      if (!ctx.isFirstPass()) {
        if (*modp)
          modify(src);
        return;
      }
    }

    if (Version == detDisjoint && ctx.isFirstPass ()) {
      *modp = build<Galois::MethodFlag::WRITE>(src);
    } else {
      bool mod = build<Galois::MethodFlag::WRITE>(src);
      if (Version == detPrefix) {
        return;
      } else {
        graph.getData(src, Galois::MethodFlag::WRITE);
        ctx.cautiousPoint(); // Failsafe point
      }
      if (mod)
        modify(src);
    }
  }
};

template<bool prefix>
struct OrderedProcess {
  typedef int tt_does_not_need_push;

  typedef typename Process<>::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  Graph& graph;
  Process<> process;

  OrderedProcess(Graph& g): graph(g), process(g) { }

  template<typename C>
  void operator()(GNode src, C& ctx) {
    (*this)(src);
  }

  void operator()(GNode src) {
    if (prefix) {
      graph.edge_begin(src, Galois::MethodFlag::WRITE);
    } else {
      if (process.build<Galois::MethodFlag::UNPROTECTED>(src))
        process.modify(src);
    }
  }
};

template<typename Graph>
struct Compare {
  typedef typename Graph::GraphNode GNode;
  Graph& graph;

  Compare(Graph& g): graph(g) { }
  
  bool operator()(const GNode& a, const GNode& b) const {
    return &graph.getData(a, Galois::MethodFlag::UNPROTECTED)< &graph.getData(b, Galois::MethodFlag::UNPROTECTED);
  }
};


template<Algo algo>
struct DefaultAlgo {
  typedef typename Process<>::Graph Graph;

  void operator()(Graph& graph) {
    typedef Galois::WorkList::Deterministic<> DWL;

    typedef Galois::WorkList::BulkSynchronous<> WL;
    //    typedef Galois::WorkList::dChunkedFIFO<256> WL;

    switch (algo) {
      case nondet: 
        Galois::for_each(graph.begin(), graph.end(), Process<>(graph), Galois::wl<WL>());
        break;
      case detBase:
        Galois::for_each(graph.begin(), graph.end(), Process<>(graph), Galois::wl<DWL>());
        break;
      case detPrefix:
        Galois::for_each(graph.begin(), graph.end(), Process<>(graph),
            Galois::wl<DWL>(),
            Galois::make_trait_with_args<Galois::has_neighborhood_visitor>(Process<detPrefix>(graph))
            );
        break;
      case detDisjoint:
        Galois::for_each(graph.begin(), graph.end(), Process<detDisjoint>(graph), Galois::wl<DWL>());
        break;
      case orderedBase:
        Galois::for_each_ordered(graph.begin(), graph.end(), Compare<Graph>(graph),
            OrderedProcess<true>(graph), OrderedProcess<false>(graph));
        break;
      default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
    }
  }
};

struct PullAlgo {
  typedef Galois::Graph::LC_CSR_Graph<Node,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  struct Pull {
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    typedef Galois::InsertBag<GNode> Bag;

    Graph& graph;
    Bag& matched;
    Bag& otherMatched;
    Bag& next;
    Galois::GAccumulator<size_t>& numProcessed;

    void operator()(GNode src, Galois::UserContext<GNode>&) const {
      operator()(src);
    }

    void operator()(GNode src) const {
      numProcessed += 1;
      //Node& n = graph.getData(src, Galois::MethodFlag::UNPROTECTED);

      MatchFlag f = MATCHED;
      for (auto edge : graph.out_edges(src, Galois::MethodFlag::UNPROTECTED)) {
        GNode dst = graph.getEdgeDst(edge);
        if (dst >= src) {
          continue; 
        } 
        
        Node& other = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
        if (other.flag == MATCHED) {
          f = OTHER_MATCHED;
          break;
        } else if (other.flag == UNMATCHED) {
          f = UNMATCHED;
        }
      }

      if (f == UNMATCHED) {
        next.push_back(src);
      } else if (f == MATCHED) {
        matched.push_back(src);
      } else {
        otherMatched.push_back(src);
      }
    }
  };

  template<MatchFlag F>
  struct Take {
    Graph& graph;
    Galois::GAccumulator<size_t>& numTaken;

    void operator()(GNode src) const {
      Node& n = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
      numTaken += 1;
      n.flag = F;
    }
  };

  void operator()(Graph& graph) {
    Galois::Statistic rounds("Rounds");
    Galois::GAccumulator<size_t> numProcessed;
    Galois::GAccumulator<size_t> numTaken;

    typedef Galois::InsertBag<GNode> Bag;
    Bag bags[2];
    Bag *cur = &bags[0];
    Bag *next = &bags[1];
    Bag matched;
    Bag otherMatched;
    uint64_t size = graph.size();
    uint64_t delta = graph.size() / 25;

    Graph::iterator ii = graph.begin();
    Graph::iterator ei = graph.begin();

    while (size > 0) {
      Pull pull { graph, matched, otherMatched, *next, numProcessed };
      Take<MATCHED> takeMatched { graph, numTaken };
      Take<OTHER_MATCHED> takeOtherMatched { graph, numTaken };

      numProcessed.reset();

      if (!cur->empty()) {
        typedef Galois::WorkList::StableIterator<> WL;
        //Galois::for_each_local(*cur, pull, Galois::wl<WL>());
        Galois::do_all_local(*cur, pull);
      }

      size_t numCur = numProcessed.reduce();
      std::advance(ei, std::min(size, delta) - numCur);

      if (ii != ei)
        Galois::do_all(ii, ei, pull);
      ii = ei;

      numTaken.reset();

      Galois::do_all_local(matched, takeMatched);
      Galois::do_all_local(otherMatched, takeOtherMatched);

      cur->clear();
      matched.clear();
      otherMatched.clear();
      std::swap(cur, next);
      rounds += 1;
      size -= numTaken.reduce();
    }
  }
};

template<typename Graph>
struct is_bad {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_type Node;
  Graph& graph;

  is_bad(Graph& g): graph(g) { }

  bool operator()(GNode n) const {
    Node& me = graph.getData(n);
    if (me.flag == MATCHED) {
      for (auto ii : graph.edges(n)) {
        GNode dst = graph.getEdgeDst(ii);
        Node& data = graph.getData(dst);
        if (dst != n && data.flag == MATCHED) {
          std::cerr << "double match\n";
          return true;
        }
      }
    } else if (me.flag == UNMATCHED) {
      bool ok = false;
      for (auto ii : graph.edges(n)) {
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

template<typename Graph>
struct is_matched {
  typedef typename Graph::GraphNode GNode;
  Graph& graph;
  is_matched(Graph& g): graph(g) { }

  bool operator()(const GNode& n) const {
    return graph.getData(n).flag == MATCHED;
  }
};

template<typename Graph>
bool verify(Graph& graph) {
  return Galois::ParallelSTL::find_if(
      graph.begin(), graph.end(), is_bad<Graph>(graph))
    == graph.end();
}

template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  Algo algo;
  Graph graph;
  Galois::Graph::readGraph(graph, filename);

  // Galois::preAlloc(numThreads + (graph.size() * sizeof(Node) * numThreads / 8) / Galois::Runtime::MM::hugePageSize);
  // Tighter upper bound
  if (std::is_same<Algo, DefaultAlgo<nondet> >::value) {
    Galois::preAlloc(numThreads + 8*graph.size()/Galois::Runtime::pagePoolSize());
  } else {
    Galois::preAlloc(numThreads + 64*(sizeof(GNode) + sizeof(Node))*graph.size()/Galois::Runtime::pagePoolSize());
  }

  Galois::reportPageAlloc("MeminfoPre");
  Galois::StatTimer T;
  T.start();
  algo(graph);
  T.stop();
  Galois::reportPageAlloc("MeminfoPost");

  std::cout << "Cardinality of maximal independent set: " 
    << Galois::ParallelSTL::count_if(graph.begin(), graph.end(), is_matched<Graph>(graph)) 
    << "\n";

  if (!skipVerify && !verify(graph)) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);
  
  switch (algo) {
    case serial: run<SerialAlgo>(); break;
    case nondet: run<DefaultAlgo<nondet> >(); break;
    case detBase: run<DefaultAlgo<detBase> >(); break;
    case detPrefix: run<DefaultAlgo<detPrefix> >(); break;
    case detDisjoint: run<DefaultAlgo<detDisjoint> >(); break;
    case orderedBase: run<DefaultAlgo<orderedBase> >(); break;
    case pull: run<PullAlgo>(); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }
  return 0;
}
