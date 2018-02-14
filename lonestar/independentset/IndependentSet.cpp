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
 * @author Michael Jiayuan He <hejy@cs.utexas.edu> (prio implementation)
 */
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "galois/runtime/Profile.h"
#include "llvm/Support/CommandLine.h"


#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <random>
#include <math.h>

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
  prio,
  prio2
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
      clEnumVal(prio, "prio algo with priority based on degree"),
      clEnumVal(prio2, "prio algo based on Martin's GPU ECL-MIS algorithm"),
      clEnumValEnd), cll::init(nondet));

enum MatchFlag: char {
  UNMATCHED, OTHER_MATCHED, MATCHED
};

struct Node {
  MatchFlag flag; 
  Node() : flag(UNMATCHED) { }
};

struct prioFlag {
  char in:1;
  char prio:6;
  char decided:1;
};

struct prioNode {
  MatchFlag flag; 
  float prio;
  prioNode() : flag(UNMATCHED), prio(0) { }
};

struct prioNode2 {
  unsigned char flag; // 1 bit matched, 3 bits prio of degree, 3 bits random prio, 1 bit undecided
  prioNode2() : flag((unsigned char)0x01) { }
};


struct SerialAlgo {
  typedef galois::graphs::LC_CSR_Graph<Node,void>
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
  typedef typename galois::graphs::LC_CSR_Graph<Node,void>
    ::template with_numa_alloc<true>::type Graph;

  typedef typename Graph::GraphNode GNode;

  struct LocalState {
    bool mod;
    LocalState(Process<Version>& self, galois::PerIterAllocTy& alloc): mod(false) { }
  };

  struct DeterministicId {
    uintptr_t operator()(const GNode& x) const {
      return x;
    }
  };

  typedef std::tuple<
    galois::no_pushes,
    galois::per_iter_alloc,
    galois::det_id<DeterministicId>,
    galois::local_state<LocalState>
  > function_traits;

  Graph& graph;

  Process(Graph& g): graph(g) { }

  template<galois::MethodFlag Flag>
  bool build(GNode src) {
    Node& me = graph.getData(src, Flag);
    if (me.flag != UNMATCHED)
      return false;

    for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Flag);
      if (data.flag == MATCHED)
        return false;
    }

    return true;
  }

  void modify(GNode src) {
    Node& me = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      data.flag = OTHER_MATCHED;
    }

    me.flag = MATCHED;
  }

  void operator()(GNode src, galois::UserContext<GNode>& ctx) {
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
      *modp = build<galois::MethodFlag::WRITE>(src);
    } else {
      bool mod = build<galois::MethodFlag::WRITE>(src);
      if (Version == detPrefix) {
        return;
      } else {
        graph.getData(src, galois::MethodFlag::WRITE);
        ctx.cautiousPoint(); // Failsafe point
      }
      if (mod)
      {
        modify(src);
      }
    }
  }
};

template<bool prefix>
struct OrderedProcess {
  // typedef int tt_does_not_need_push;

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
      graph.edge_begin(src, galois::MethodFlag::WRITE);
    } else {
      if (process.build<galois::MethodFlag::UNPROTECTED>(src))
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
    return &graph.getData(a, galois::MethodFlag::UNPROTECTED)< &graph.getData(b, galois::MethodFlag::UNPROTECTED);
  }
};


template<Algo algo>
struct DefaultAlgo {
  typedef typename Process<>::Graph Graph;

  void operator()(Graph& graph) {
    typedef galois::worklists::Deterministic<> DWL;

    typedef galois::worklists::BulkSynchronous<typename galois::worklists::dChunkedFIFO<256> > WL;
        //typedef galois::worklists::dChunkedFIFO<256> WL;

    switch (algo) {
      case nondet: 
        galois::for_each(galois::iterate(graph), Process<>(graph), galois::loopname("Main"), galois::wl<WL>());
        break;
      case detBase:
        galois::for_each(galois::iterate(graph), Process<>(graph), galois::loopname("Main"), galois::wl<DWL>());
        break;
      case detPrefix:
        galois::for_each(galois::iterate(graph), Process<>(graph),
            galois::loopname("Main"), galois::wl<DWL>(),
            galois::make_trait_with_args<galois::neighborhood_visitor>(Process<detPrefix>(graph))
            );
        break;
      case detDisjoint:
        galois::for_each(galois::iterate(graph), Process<detDisjoint>(graph), galois::wl<DWL>());
        break;
      case orderedBase:
        galois::for_each_ordered(graph.begin(), graph.end(), Compare<Graph>(graph),
            OrderedProcess<true>(graph), OrderedProcess<false>(graph));
        break;
      default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
    }
  }
};

struct PullAlgo {
  typedef galois::graphs::LC_CSR_Graph<Node,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  struct Pull {
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    typedef galois::InsertBag<GNode> Bag;

    Graph& graph;
    Bag& matched;
    Bag& otherMatched;
    Bag& next;
    galois::GAccumulator<size_t>& numProcessed;

    void operator()(GNode src, galois::UserContext<GNode>&) const {
      operator()(src);
    }

    void operator()(GNode src) const {
      numProcessed += 1;
      Node& n = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      if(n.flag == OTHER_MATCHED)
        return;

      MatchFlag f = MATCHED;
      for (auto edge : graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
        GNode dst = graph.getEdgeDst(edge);
        if (dst >= src) {
          continue; 
        } 
        
        Node& other = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
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
      //std::cout<<src<< " " << f <<std::endl;
    }
  };

  template<MatchFlag F>
  struct Take {
    Graph& graph;
    galois::GAccumulator<size_t>& numTaken;

    void operator()(GNode src) const {
      Node& n = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      numTaken += 1;
      n.flag = F;
    }
  };

  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> rounds;
    galois::GAccumulator<size_t> numProcessed;
    galois::GAccumulator<size_t> numTaken;

    typedef galois::InsertBag<GNode> Bag;
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
        // typedef galois::worklists::StableIterator<> WL;
        //galois::for_each(*cur, pull, galois::wl<WL>());
        galois::do_all(galois::iterate(*cur), pull, galois::loopname("pull-0"));
      }

      size_t numCur = numProcessed.reduce();
      std::advance(ei, std::min(size, delta) - numCur);

      if (ii != ei)
        galois::do_all(galois::iterate(ii, ei), pull, galois::loopname("pull-1"));
      ii = ei;

      numTaken.reset();

      galois::do_all(galois::iterate(matched), takeMatched, galois::loopname("takeMatched"));
      galois::do_all(galois::iterate(otherMatched), takeOtherMatched, galois::loopname("takeOtherMatched"));

      cur->clear();
      matched.clear();
      otherMatched.clear();
      std::swap(cur, next);
      rounds += 1;
      size -= numTaken.reduce();
      //std::cout<<size<<std::endl;
      //break;
    }

    galois::runtime::reportStat_Single("IndependentSet-PullAlgo", "rounds", rounds.reduce());
  }
};

struct PrioAlgo {
  typedef galois::graphs::LC_CSR_Graph<prioNode,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  struct Init_perthread {
    galois::substrate::PerThreadStorage<std::mt19937* >& generator;

    void operator()(auto tid, auto numThreads) {
      *(generator.getLocal()) = new std::mt19937(clock() + tid);
    }
  };

  struct Init_prio{
    Graph& graph;
    galois::substrate::PerThreadStorage<std::mt19937* >& generator;

    int IntRand(const int &min, const int &max, std::mt19937* generator) const {
      std::uniform_int_distribution<int > dist(min, max);
      return dist(*generator);
    }

    unsigned char charRand(const int &min, const int &max, std::mt19937* generator) const {
      std::uniform_int_distribution<unsigned char > dist(min, max);
      return dist(*generator);
    }

    void operator()(GNode src) const {
      prioNode& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      int outdegree = std::distance(graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
 


      /*unsigned char prio = (unsigned char)outdegree & 7;
      prio = ~prio; 
      prio = prio & 7;

      nodedata.flag |= (prio << 4);

      nodedata.flag |= (charRand(0, 7, *(generator.getLocal())) << 1);*/


      nodedata.prio = /*(float) IntRand(0, graph.size(), *(generator.getLocal())) / (float) graph.size() +*/ (-1.0) * outdegree;

      /*float random_num = (float) (rand() % graph.size()) / (float) graph.size(); //0 < random_num < 1

      float prio_low = 1/ (float)outdegree;

      float prio_high =  1/ (float)(outdegree + 1);

      nodedata.prio = prio_low + random_num * (prio_high - prio_low); //prio_low < prio < prio_high*/

    }
  };


  struct Execute{
    Graph& graph;
    bool& unmatched;
    void operator()(GNode src) const{
      prioNode& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      if(nodedata.flag != UNMATCHED)
        return;

      for (auto edge : graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
        GNode dst = graph.getEdgeDst(edge);

        prioNode& other = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        if (other.flag == UNMATCHED) {
            if (nodedata.prio > other.prio) {
                continue;
            }
            else if (nodedata.prio == other.prio) {
                if(src > dst)
                    continue;
                else if(src == dst)
                {
                    //std::cout<<"node id:" << src << " equal to "<< dst << std::endl;
                    nodedata.flag = OTHER_MATCHED;
                    return;
                }
                else {
                    unmatched = true;
                    return;
                }
            }
            else {
                unmatched = true;
                return;
            }
        } else if (other.flag == OTHER_MATCHED) {
            continue;
        }
        else if (other.flag == MATCHED) {
            nodedata.flag = OTHER_MATCHED;
            return;
        }
      }
      nodedata.flag = MATCHED;
      
    }
  };

  struct VerifyChange{
    Graph& graph;

    void operator()(GNode src) const{
      prioNode& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      if(nodedata.flag == (unsigned char)0xfe){
        nodedata.flag = MATCHED;
      }
      else if(nodedata.flag == (unsigned char)0x00)
      {
        nodedata.flag = OTHER_MATCHED;
      }
      else 
        std::cout<<"error in verify_change!"<<std::endl;

    }
  };

  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> rounds;
    galois::GAccumulator<float> nedges;
    bool unmatched = true;
    galois::substrate::PerThreadStorage<std::mt19937* > generator;

    Init_perthread init_perthread { generator };

    Init_prio init_prio { graph, generator };
    
    Execute execute { graph, unmatched };
    VerifyChange verify_change { graph };
    
    
    galois::on_each(init_perthread, galois::loopname("init-perthread"));

    galois::do_all(galois::iterate(graph), init_prio, galois::loopname("init-prio"));

    while (unmatched) {
      unmatched = false;

      galois::do_all(galois::iterate(graph), execute, galois::loopname("execute"));

      rounds += 1;
    }

    galois::runtime::reportStat_Single("IndependentSet-prioAlgo", "rounds", rounds.reduce());
  }

};

struct PrioAlgo2 {
  typedef galois::graphs::LC_CSR_Graph<prioNode2,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  struct Init_perthread {
    galois::substrate::PerThreadStorage<std::mt19937* >& generator;

    void operator()(auto tid, auto numThreads) {
      *(generator.getLocal()) = new std::mt19937(clock() + tid);
    }
  };

  struct Init_prio{
    Graph& graph;
    float avg_degree;
    float scale_avg;
    galois::substrate::PerThreadStorage<std::mt19937* >& generator;

    unsigned int hash(unsigned int val) const {
      val = ((val >> 16) ^ val) * 0x45d9f3b;
      val = ((val >> 16) ^ val) * 0x45d9f3b;
      return (val >> 16) ^ val;
    }

    void operator()(GNode src) const {
      prioNode2& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      float degree = (float) std::distance(graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
      float x = degree - hash(src) * 0.00000000023283064365386962890625f;
      int res = round(scale_avg / (avg_degree + x));
      unsigned char val = (res + res) | 1;

      nodedata.flag = val;

    }
  };

  struct Cal_degree{
    Graph& graph;
    galois::GAccumulator<float >& nedges;

    void operator()(GNode src) const {
      prioNode2& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      nedges += std::distance(graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
    }
  };


  struct Execute{
    Graph& graph;
    bool& unmatched;
    void operator()(GNode src) const{
      prioNode2& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      if(!(nodedata.flag & (unsigned char)1))
        return;

      for (auto edge : graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
        GNode dst = graph.getEdgeDst(edge);

        prioNode2& other = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        if(other.flag == (unsigned char)0xfe) //matched, highest prio
        {
          nodedata.flag = (unsigned char)0x00;
          unmatched = true;
          return;
        }

        if(nodedata.flag > other.flag)
            continue;
        else if(nodedata.flag == other.flag){
            if(src > dst)
                continue;
            else if(src == dst)
            {
                nodedata.flag = (unsigned char)0x00; //other_matched
                return;
            }
            else {
                unmatched = true;
                return;
            }
        }
        else{
            unmatched = true;
            return;
        }

      }
      /*for (auto edge : graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
         GNode dst = graph.getEdgeDst(edge);

         prioNode2& other = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
         other.flag = (unsigned char)0x00;//OTHER_MATCHED;
      }*/
      nodedata.flag = (unsigned char)0xfe; //matched, highest prio
    }
  };

  struct VerifyChange{
    Graph& graph;

    void operator()(GNode src) const{
      prioNode2& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      if(nodedata.flag == (unsigned char)0xfe){
        nodedata.flag = MATCHED;
      }
      else if(nodedata.flag == (unsigned char)0x00)
      {
        nodedata.flag = OTHER_MATCHED;
      }
      else 
        std::cout<<"error in verify_change!"<<std::endl;

    }
  };


  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> rounds;
    galois::GAccumulator<float> nedges;
    bool unmatched = true;
    galois::substrate::PerThreadStorage<std::mt19937* > generator;

    Cal_degree cal_degree { graph, nedges };
    galois::do_all(galois::iterate(graph), cal_degree, galois::loopname("cal_degree"), galois::steal());

    Init_perthread init_perthread { generator };
    float nedges_tmp = nedges.reduce();
    float avg_degree = nedges_tmp / (float)graph.size();
    unsigned char in = ~1;
    float scale_avg = ((in / 2 ) - 1) * avg_degree;
    Init_prio init_prio { graph, avg_degree, scale_avg, generator };
    
    Execute execute { graph, unmatched };
    VerifyChange verify_change { graph };
    
    galois::on_each(init_perthread, galois::loopname("init-perthread"));

    galois::do_all(galois::iterate(graph), init_prio, galois::loopname("init-prio"));

    while (unmatched) {
      unmatched = false;

      galois::do_all(galois::iterate(graph), execute, galois::loopname("execute"));

      rounds += 1;
    }
    galois::do_all(galois::iterate(graph), verify_change, galois::loopname("verify_change"));

    galois::runtime::reportStat_Single("IndependentSet-prioAlgo", "rounds", rounds.reduce());
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
  return galois::ParallelSTL::find_if(
      graph.begin(), graph.end(), is_bad<Graph>(graph))
    == graph.end();
}

template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  Algo algo;
  Graph graph;
  galois::graphs::readGraph(graph, filename);

  // galois::preAlloc(numThreads + (graph.size() * sizeof(Node) * numThreads / 8) / galois::runtime::MM::hugePageSize);
  // Tighter upper bound
  if (std::is_same<Algo, DefaultAlgo<nondet> >::value) {
    galois::preAlloc(numThreads + 16*graph.size()/galois::runtime::pagePoolSize());
  } else {
    galois::preAlloc(numThreads + 64*(sizeof(GNode) + sizeof(Node))*graph.size()/galois::runtime::pagePoolSize());
  }
  
  galois::reportPageAlloc("MeminfoPre");
  galois::StatTimer T;

  T.start();
  galois::runtime::profileVtune(
    [&] () {
    algo(graph);
    }, "algo()"
  ); 
  T.stop();
  galois::reportPageAlloc("MeminfoPost");

  std::cout << "Cardinality of maximal independent set: " 
    << galois::ParallelSTL::count_if(graph.begin(), graph.end(), is_matched<Graph>(graph)) 
    << "\n";

  if (!skipVerify && !verify(graph)) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  
  switch (algo) {
    case serial: run<SerialAlgo>(); break;
    case nondet: run<DefaultAlgo<nondet> >(); break;
    case detBase: run<DefaultAlgo<detBase> >(); break;
    case detPrefix: run<DefaultAlgo<detPrefix> >(); break;
    case detDisjoint: run<DefaultAlgo<detDisjoint> >(); break;
    case orderedBase: run<DefaultAlgo<orderedBase> >(); break;
    case pull: run<PullAlgo>(); break;
    case prio: run<PrioAlgo>(); break;
    case prio2: run<PrioAlgo2>(); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }
  return 0;
}
