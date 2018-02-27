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
  prio,
  prio2,
  edgetiledprio2
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
      clEnumVal(prio, "prio algo with priority based on degree"),
      clEnumVal(prio2, "prio algo based on Martin's GPU ECL-MIS algorithm"),
      clEnumVal(edgetiledprio2, "edge-tiled prio algo based on Martin's GPU ECL-MIS algorithm"),
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
  unsigned char flag; // 1 bit matched,6 bits prio, 1 bit undecided
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


template<Algo algo>
struct DefaultAlgo {

  typedef typename galois::graphs::LC_CSR_Graph<Node,void>
    ::template with_numa_alloc<true>::type Graph;

  typedef typename Graph::GraphNode GNode;

  struct LocalState {
    bool mod;
    explicit LocalState(): mod(false) { }
  };

  template<galois::MethodFlag Flag>
  bool build(Graph& graph, GNode src) {
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

  void modify(Graph& graph, GNode src) {
    Node& me = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      data.flag = OTHER_MATCHED;
    }

    me.flag = MATCHED;
  }

  template <int Version, typename C>
  void processNode(Graph& graph, const GNode& src, C& ctx) {
    bool mod;
    if (Version != detDisjoint || ctx.isFirstPass()) {
      mod = build<galois::MethodFlag::WRITE>(graph, src);

      if (Version == detPrefix) { 
        return;
      } else if (Version == detDisjoint) {
        LocalState* localState = ctx.template createLocalState<LocalState>();
        localState->mod = mod;
        return;
      } else {
        graph.getData(src, galois::MethodFlag::WRITE);
        ctx.cautiousPoint(); // Failsafe point
      }
    } else { // Version == detDisjoint && !ctx.isFirstPass
      LocalState* localState = ctx.template getLocalState<LocalState>();
      mod = localState->mod;
    }

    if (mod) {
      modify(graph, src);
    }
  }

  template <int Version, typename WL, typename... Args>
  void run(Graph& graph, Args&&... args) {

    auto detID = [] (const GNode& x) { 
      return x;
    };

    galois::for_each(galois::iterate(graph), 
        [&, this] (const GNode& src, auto& ctx) {
          this->processNode<Version>(graph, src, ctx);
        },
        galois::no_pushes(),
        galois::wl<WL>(),
        galois::loopname("DefaultAlgo"),
        galois::det_id< decltype(detID) >(detID),
        galois::local_state<LocalState>(),
        std::forward<Args>(args)...);

  }



  void operator()(Graph& graph) {
    typedef galois::worklists::Deterministic<> DWL;

    typedef galois::worklists::BulkSynchronous<typename galois::worklists::dChunkedFIFO<64> > BSWL;
        //typedef galois::worklists::dChunkedFIFO<256> WL;

    switch (algo) {
      case nondet: 
        run<nondet, BSWL>(graph);
        break;
      case detBase:
        run<detBase, DWL>(graph);
        break;
      case detPrefix:
        {
          auto nv = [&, this] (const GNode& src, auto& ctx) {
            this->processNode<detPrefix>(graph, src, ctx);
          };
          run<detBase, DWL>(graph, galois::neighborhood_visitor< decltype(nv) >(nv));
        }
        break;
      case detDisjoint:
        run<detDisjoint, DWL>(graph);
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
  typedef galois::InsertBag<GNode> Bag;

  using Counter = galois::GAccumulator<size_t>;

  template <typename R>
  void pull(const R& range, Graph& graph, Bag& matched, Bag& otherMatched, Bag& next, Counter& numProcessed) {

    galois::do_all(range, 
        [&] (const GNode& src) {
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
          
        },
        galois::loopname("pull"));


  }

  template <MatchFlag F>
  void take(Bag& bag, Graph& graph, Counter& numTaken) {

    galois::do_all(galois::iterate(bag), 
        [&] (const GNode& src) {
          Node& n = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          numTaken += 1;
          n.flag = F;
        },
        galois::loopname("take"));

  }


  void operator()(Graph& graph) {
    size_t rounds = 0;
    Counter numProcessed;
    Counter numTaken;

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
      numProcessed.reset();

      if (!cur->empty()) {
        pull(galois::iterate(*cur), graph, matched, otherMatched, *next, numProcessed);
      }

      size_t numCur = numProcessed.reduce();
      std::advance(ei, std::min(size, delta) - numCur);

      if (ii != ei) {
        pull(galois::iterate(ii, ei), graph, matched, otherMatched, *next, numProcessed);
      }

      ii = ei;

      numTaken.reset();

      take<MATCHED>(matched, graph, numTaken);
      take<OTHER_MATCHED>(otherMatched, graph, numTaken);

      cur->clear();
      matched.clear();
      otherMatched.clear();
      std::swap(cur, next);
      rounds += 1;
      assert(size >= numTaken.reduce());
      size -= numTaken.reduce();
      //std::cout<<size<<std::endl;
      //break;
    }

    galois::runtime::reportStat_Single("IndependentSet-PullAlgo", "rounds", rounds);
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

      nedges += std::distance(graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
    }
  };


  struct Execute{
    Graph& graph;
    galois::GReduceLogicalOR& unmatched;
    //bool& unmatched;
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
          unmatched.update(true);
          //unmatched = true;
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
                unmatched.update(true);
                //unmatched = true;
                return;
            }
        }
        else{
            unmatched.update(true);
            //unmatched = true;
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
    galois::GReduceLogicalOR unmatched;
    //bool unmatched = true;
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

    do {
      unmatched.reset();
      //unmatched = false;
      galois::do_all(galois::iterate(graph), execute, galois::loopname("execute"));

      rounds += 1;
    }while (unmatched.reduce());

    galois::do_all(galois::iterate(graph), verify_change, galois::loopname("verify_change"));

    galois::runtime::reportStat_Single("IndependentSet-prioAlgo", "rounds", rounds.reduce());
  }

};

struct edgetiledPrioAlgo2 {
  typedef galois::graphs::LC_CSR_Graph<prioNode2,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  struct EdgeTile{
      //prioNode2* nodedata;
      GNode src;
      Graph::edge_iterator beg;
      Graph::edge_iterator end;
      bool flag;
  };
  //const int EDGE_TILE_SIZE=512;

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
    galois::InsertBag<EdgeTile>& works;

    const int EDGE_TILE_SIZE=512;
    unsigned int hash(unsigned int val) const {
      val = ((val >> 16) ^ val) * 0x45d9f3b;
      val = ((val >> 16) ^ val) * 0x45d9f3b;
      return (val >> 16) ^ val;
    }

    void operator()(GNode src) const {
      prioNode2& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      auto beg = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
      const auto end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);

      float degree = (float) std::distance(beg, end);
      float x = degree - hash(src) * 0.00000000023283064365386962890625f;
      int res = round(scale_avg / (avg_degree + x));
      unsigned char val = (res + res) | 0x03;

      nodedata.flag = val;

      assert(beg <= end);
      if ((end - beg) > EDGE_TILE_SIZE) {
          for (; beg + EDGE_TILE_SIZE < end;) {
              auto ne = beg + EDGE_TILE_SIZE;
              assert(ne < end);
              works.push_back( EdgeTile{src, beg, ne} );
              beg = ne;
          }
      }
      
      if ((end - beg) > 0) {                                                                                               
          works.push_back( EdgeTile{src, beg, end} );  
      } 

    }
  };

  struct Cal_degree{
    Graph& graph;
    galois::GAccumulator<float >& nedges;

    void operator()(GNode src) const {
      nedges += std::distance(graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
    }
  };


  struct Execute{
    Graph& graph;
    galois::GAccumulator<size_t>& rounds;
    galois::GReduceLogicalOR& unmatched;
    galois::InsertBag<EdgeTile>& works;
    //bool& unmatched;
    void operator()(EdgeTile& tile) const{
      GNode src = tile.src; 
      
      prioNode2& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      if((nodedata.flag & (unsigned char)1)){ //is undecided 

        for (auto edge = tile.beg; edge != tile.end; ++edge) {
          GNode dst = graph.getEdgeDst(edge);

          prioNode2& other = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

          if(other.flag == (unsigned char)0xfe) //permanent matched, highest prio
          {
            nodedata.flag = (unsigned char)0x00;
            //unmatched.update(true);
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
                  tile.flag = false;
                  return;
              }
              else {
                  tile.flag = false;
                  unmatched.update(true);
                  //std::cout<<"here0"<<std::endl;
                  return;
              }
          }
          else{
              tile.flag = false;
              unmatched.update(true);
              if(rounds.reduce() > 5)
                std::cout<<"here1 this flag"<<std::hex<< (unsigned)nodedata.flag<< "other flag " <<std::hex<<(unsigned)other.flag <<std::endl;
              return;
          }
        }
        tile.flag = true; //temporary-matched
      }
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

  struct MatchReduce{
    Graph& graph;

    void operator()(EdgeTile& tile) const{
      auto src = tile.src;
      prioNode2& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      
      if((nodedata.flag & (unsigned char)1) && tile.flag == false){//undecided and temporary no
        nodedata.flag &= (unsigned char)0xfd; // 0x1111 1101, not temporary yes
      }
    }
  };

  struct MatchUpdate{
    Graph& graph;

    void operator()(GNode src) const{
      prioNode2& nodedata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      if((nodedata.flag & (unsigned char)0x01)){ // undecided 
        if(nodedata.flag & (unsigned char)0x02){ //temporary yes
            nodedata.flag = (unsigned char)0xfe; // 0x1111 1110, permanent yes
            for (auto edge : graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
                GNode dst = graph.getEdgeDst(edge);

                prioNode2& other = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
                other.flag = (unsigned char)0x00;//OTHER_MATCHED, permanent no
            }
        }
        else
            nodedata.flag |= (unsigned char) 0x03; //0x0000 0011, temp yes, undecided
      }
    }
  };
  
  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> rounds;
    galois::GAccumulator<float> nedges;
    galois::GReduceLogicalOR unmatched;
    //bool unmatched = true;
    galois::substrate::PerThreadStorage<std::mt19937* > generator;
    galois::InsertBag<EdgeTile> works;

    Cal_degree cal_degree { graph, nedges };
    galois::do_all(galois::iterate(graph), cal_degree, galois::loopname("cal_degree"), galois::steal());

    Init_perthread init_perthread { generator };
    float nedges_tmp = nedges.reduce();
    float avg_degree = nedges_tmp / (float)graph.size();
    unsigned char in = ~1;
    float scale_avg = ((in / 2 ) - 1) * avg_degree;
    Init_prio init_prio { graph, avg_degree, scale_avg, generator, works };
    
    Execute execute { graph, rounds, unmatched, works };
    VerifyChange verify_change { graph };
    MatchReduce match_reduce { graph };
    MatchUpdate match_update { graph };
    
    galois::on_each(init_perthread, galois::loopname("init-perthread"));

    galois::do_all(galois::iterate(graph), init_prio, galois::loopname("init-prio"));

    do {
      unmatched.reset();
      galois::do_all(galois::iterate(works), execute, galois::loopname("execute"));
      
      galois::do_all(galois::iterate(works), match_reduce, galois::loopname("match_reduce"));
      
      galois::do_all(galois::iterate(graph), match_update, galois::loopname("match_update"));
      std::cout << "round:"<< rounds.reduce()<< std::endl;
      rounds += 1;
    }while (unmatched.reduce());

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
    case pull: run<PullAlgo>(); break;
    case prio: run<PrioAlgo>(); break;
    case prio2: run<PrioAlgo2>(); break;
    case edgetiledprio2: run<edgetiledPrioAlgo2>(); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }
  return 0;
}
