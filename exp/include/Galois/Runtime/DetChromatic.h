#ifndef GALOIS_RUNTIME_DET_CHROMATIC_H
#define GALOIS_RUNTIME_DET_CHROMATIC_H

#include "Galois/Accumulator.h"
#include "Galois/AltBag.h"
#include "Galois/DoAllWrap.h"
#include "Galois/Galois.h"
//#include "Galois/GaloisUnsafe.h"

#include "Galois/Graph/Graph.h"

#include "Galois/WorkList/WorkListWrapper.h"
#include "Galois/WorkList/ExternalReference.h"

#include <atomic>
#include <vector>

namespace Galois {
namespace Runtime {

enum PriorityFunc {
  FIRST_FIT,
  BY_ID,
  RANDOM,
  MIN_DEGREE,
  MAX_DEGREE,
};

namespace cll = llvm::cl;

static cll::opt<PriorityFunc> priorityFunc (
    "priority",
    cll::desc ("choose ordering heuristic"),
    cll::values (
      clEnumValN (FIRST_FIT, "FIRST_FIT", "first fit, no priority"),
      clEnumValN (BY_ID, "BY_ID", "order by ID modulo some constant"),
      clEnumValN (RANDOM, "RANDOM", "uniform random within some small range"),
      clEnumValN (MIN_DEGREE, "MIN_DEGREE", "order by min degree first"),
      clEnumValN (MAX_DEGREE, "MAX_DEGREE", "order by max degree first"),
      clEnumValEnd),
    cll::init (BY_ID));

struct DAGdata {
  unsigned color;
  unsigned id;
  unsigned priority;
  std::atomic<bool> onWL;
  std::atomic<unsigned> indegree;


  explicit DAGdata (unsigned _id=0)
    : color (0), id (_id), priority (0), onWL (false), indegree (0) 
  {}
};

template <typename GNode>
struct DAGdataDirected: public DAGdata {

  typedef Galois::gdeque<GNode, 64> AdjList;

  AdjList preds;

  DAGdataDirected (unsigned id): DAGdata (id) {}

  void addPred (GNode n) {
    assert (std::find (preds.begin (), preds.end (), n) == preds.end ());
    preds.push_back (n);
  }
  
};

template <typename ND>
struct DAGdataComparator {

  static bool compare (const ND& left, const ND& right) {
    if (left.priority != right.priority) {
      return left.priority < right.priority;
    } else {
      return left.id < right.id;
    }
  }

  bool operator () (const ND& left, const ND& right) const {
    return compare (left, right);
  }
};

template <typename G, typename P, typename S>
struct DAGmanager {

protected:

  static const unsigned DEFAULT_CHUNK_SIZE = 4;

  typedef typename G::GraphNode GNode;
  typedef typename G::node_data_type ND;

  typedef Galois::Runtime::PerThreadVector<unsigned> PerThrdColorVec;

  G& graph;
  P visitPreds;
  S visitSuccs;
  PerThrdColorVec perThrdColorVec;
  Galois::GReduceMax<unsigned> maxColors;

  DAGmanager (G& graph, const P& visitPreds, const S& visitSuccs)
    : graph (graph), visitPreds (visitPreds), visitSuccs (visitSuccs) 
  {
    // mark 0-th color as taken
    for (unsigned i = 0; i < perThrdColorVec.numRows (); ++i) {
      auto& forbiddenColors = perThrdColorVec.get(i);
      forbiddenColors.resize (1, 0);
      forbiddenColors[0] = 0;
    }
  }

public:

  template <typename F>
  void applyToAdj (GNode src, F& f) {
    visitSuccs (src, f);
    visitPreds (src, f);
  }

  template <typename W>
  void initDAG (W& sources) {
    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [this, &sources] (GNode src) {
          
          auto& sd = graph.getData (src, Galois::NONE);

          assert (sd.indegree == 0);
          unsigned addAmt = 0;

          auto closure = [this, &sd, &addAmt] (GNode dst) {
            auto& dd = graph.getData (dst, Galois::NONE);

            if (DAGdataComparator<ND>::compare (dd, sd)) { // dd < sd
              ++addAmt;
            }
          };

          applyToAdj (src, closure);

          sd.indegree += addAmt;

          if (addAmt == 0) {
            assert (sd.indegree == 0);
            sources.push (src);
          }

        },
        "init-DAG",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());
  }

  template <typename R, typename W>
  void initActiveDAG (const R& range, W& sources) {
    Galois::do_all_choice (
        range,
        [this, &sources] (GNode src) {
          
          auto& sd = graph.getData (src, Galois::NONE);
          assert (bool (sd.onWL));

          // if (bool (sd.onWL)) {
            sd.indegree = 0; // reset

            unsigned addAmt = 0;

            auto closure = [this, &sd, &addAmt] (GNode dst) {
              auto& dd = graph.getData (dst, Galois::NONE);

                if (bool (dd.onWL) && DAGdataComparator<ND>::compare (dd, sd)) { // dd < sd
              ++addAmt;
              }
            };


            applyToAdj (src, closure);

            sd.indegree += addAmt;

            if (addAmt == 0) {
              assert (sd.indegree == 0);
              assert (bool(sd.onWL));
              sources.push (src);
            }
          // } // end if

        },
        "init-Active-DAG",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());
  }

  void assignIDs (void) {

    const size_t numNodes = graph.size ();
    Galois::on_each (
        [&] (const unsigned tid, const unsigned numT) {

          size_t num_per = (numNodes + numT - 1) / numT;
          size_t beg = tid * num_per;
          size_t end = std::min (numNodes, (tid + 1) * num_per);

          auto it_beg = graph.begin ();
          std::advance (it_beg, beg);

          auto it_end = it_beg; 
          std::advance (it_end, (end - beg));

          for (; it_beg != it_end; ++it_beg) {
            auto& nd = graph.getData (*it_beg, Galois::NONE);
            nd.id = beg++;
          }
        },
        Galois::loopname ("assign-ids"));
  }


  template <typename F>
  void assignPriorityHelper (const F& nodeFunc) {
    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [&] (GNode node) {
          nodeFunc (node);
        },
        "assign-priority",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());
  }

  static const unsigned MAX_LEVELS = 100;
  static const unsigned SEED = 10;

  struct RNG {
    std::uniform_int_distribution<unsigned> dist;
    std::mt19937 eng;
    
    RNG (void): dist (0, MAX_LEVELS), eng () {
      this->eng.seed (SEED);
    }

    unsigned operator () (void) {
      return dist(eng);
    }
  };

  void assignPriority (void) {
    assignIDs ();
    auto byId = [&] (GNode node) {
      auto& nd = graph.getData (node, Galois::NONE);
      nd.priority = nd.id % MAX_LEVELS;
    };

    Galois::Runtime::PerThreadStorage<RNG>  perThrdRNG;

    // TODO: non-deterministic at the moment
    // can be fixed by making thread K call the generator
    // N times, where N is sum of calls of all threads < K
    auto randPri = [&] (GNode node) {
      auto& rng = *(perThrdRNG.getLocal ());
      auto& nd = graph.getData (node, Galois::NONE);
      nd.priority = rng ();
    };


    auto minDegree = [&] (GNode node) {
      auto& nd = graph.getData (node, Galois::NONE);
      nd.priority = std::distance (
                      graph.edge_begin (node, Galois::NONE),
                      graph.edge_end (node, Galois::NONE));
    };

    const size_t numNodes = graph.size ();
    auto maxDegree = [&] (GNode node) {
      auto& nd = graph.getData (node, Galois::NONE);
      nd.priority = numNodes - std::distance (
                                  graph.edge_begin (node, Galois::NONE),
                                  graph.edge_end (node, Galois::NONE));
    };
    
    Galois::StatTimer t_priority ("priority assignment time: ");

    t_priority.start ();

    switch (priorityFunc) {
      case FIRST_FIT:
        // do nothing
        break;

      case BY_ID:
        assignPriorityHelper (byId);
        break;

      case RANDOM:
        assignPriorityHelper (randPri);
        break;

      case MIN_DEGREE:
        assignPriorityHelper (minDegree);
        break;

      case MAX_DEGREE:
        assignPriorityHelper (maxDegree);
        break;

      default:
        std::abort ();
    }

    t_priority.stop ();
  }


  template <typename C>
  void colorNode (GNode src, C& ctx) {
    auto& sd = graph.getData (src, Galois::NONE);
    assert (sd.indegree == 0);
    assert (sd.color == 0); // uncolored

    auto& forbiddenColors = perThrdColorVec.get ();
    assert (!forbiddenColors.empty ()); // must have at least 1 element
    std::fill (forbiddenColors.begin (), forbiddenColors.end (), unsigned (-1));

    unsigned addAmt = 0;

    auto closure = [this, &forbiddenColors, &ctx, &sd, &addAmt] (GNode dst) {
      auto& dd = graph.getData (dst, Galois::NONE);

      if (forbiddenColors.size () <= dd.color) {
        forbiddenColors.resize (dd.color + 1, unsigned (-1));
      }
      // std::printf ("Neighbor %d has color %d\n", dd.id, dd.color);

      forbiddenColors[dd.color] = sd.id;

      ++addAmt;
      unsigned x = --(dd.indegree);
      if (x == 0 && dd.color == 0) { // color == 0 is uncolored
        ctx.push (dst);
      }
    };

    applyToAdj (src, closure);

    for (size_t i = 1; i < forbiddenColors.size (); ++i) {
      if (forbiddenColors[i] != sd.id) { 
        sd.color = i;
        break;
      }
    }

    if (sd.color == 0) {
      sd.color = forbiddenColors.size ();
    }
    maxColors.update (sd.color);

    sd.indegree += addAmt;


    // std::printf ("Node %d assigned color %d\n", sd.id, sd.color);

  }

  struct ColorNodeDAG {
    typedef int tt_does_not_need_aborts;
    DAGmanager& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {

      outer.colorNode (src, ctx);
    }
  };


  void colorDAG (void) {

    Galois::StatTimer t_color ("coloring time (total): ");

    t_color.start ();

    assignPriority ();

    Galois::InsertBag<GNode> sources;

    Galois::StatTimer t_dag_init ("dag initialization time: ");

    t_dag_init.start ();
    initDAG (sources);
    t_dag_init.stop ();

    typedef Galois::WorkList::AltChunkedFIFO<DEFAULT_CHUNK_SIZE> WL_ty;

    std::printf ("Number of initial sources: %zd\n", 
        std::distance (sources.begin (), sources.end ()));

    Galois::StatTimer t_dag_color ("just the coloring time: ");

    t_dag_color.start ();
    Galois::for_each_local (sources, ColorNodeDAG {*this}, 
        Galois::loopname ("color-DAG"), Galois::wl<WL_ty> ());
    t_dag_color.stop ();

    std::printf ("DAG colored with %d colors\n", maxColors.reduceRO ());

    t_color.stop ();
  }

  unsigned getMaxColors (void) const {
    return maxColors.reduceRO ();
  }

};

template <typename G>
struct DAGmanagerInOut {
  typedef typename G::GraphNode GNode;

  struct VisitPredecessors {
    G& graph;

    template <typename F>
    void operator () (GNode src, F& func) {
      for (auto i = graph.in_edge_begin (src, Galois::NONE)
          , end_i = graph.in_edge_end (src, Galois::NONE); i != end_i; ++i) {
        GNode dst = graph.getInEdgeDst (i);
        func (dst);
      }
    }

  };


  struct VisitSuccessors {
    G& graph;

    template <typename F>
    void operator () (GNode src, F& func) {
      for (auto i = graph.edge_begin (src, Galois::NONE)
          , end_i = graph.edge_end (src, Galois::NONE); i != end_i; ++i) {
        GNode dst = graph.getEdgeDst (i);
        func (dst);
      }
    }
  };

  typedef DAGmanager<G, VisitPredecessors, VisitSuccessors> Base_ty;

  struct Manager: public Base_ty {

    Manager (G& graph): Base_ty {graph, VisitPredecessors {graph}, VisitSuccessors {graph}} 
    {}
  };

};

template <typename G>
struct DAGvisitorUndirected {

  typedef typename G::GraphNode GNode;

  struct VisitPredecessors {
    G& graph;

    template <typename F>
    void operator () (GNode src, F& func) {};
  };

  struct VisitSuccessors {
    G& graph;

    template <typename F>
    void operator () (GNode src, F& func) {

      for (auto i = graph.edge_begin (src, Galois::NONE)
           , end_i = graph.edge_end (src, Galois::NONE); i != end_i; ++i) {

        GNode dst = graph.getEdgeDst (i);
        func (dst);
      }
    }
  };

};


// TODO: complete implementation
//
// template <typename G>
// struct DAGvisitorDirected {
// 
  // void addPredecessors (void) {
// 
    // Galois::do_all_choice (
        // Galois::Runtime::makeLocalRange (graph),
        // [this] (GNode src) {
        // 
          // auto& sd = graph.getData (src, Galois::NONE);
          // 
          // unsigned addAmt = 0;
          // for (Graph::edge_iterator e = graph.edge_begin (src, Galois::NONE),
              // e_end = graph.edge_end (src, Galois::NONE); e != e_end; ++e) {
            // GNode dst = graph.getEdgeDst (e);
            // auto& dd = graph.getData (dst, Galois::NONE);
// 
            // if (src != dst) {
              // dd.addPred (src);
            // }
          // }
// 
        // },
        // "initDAG",
        // Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());
// 
  // }
// 
// };



template <typename G, typename F>
struct ChromaticExecutor {

  typedef typename G::GraphNode GNode;

  static const unsigned CHUNK_SIZE = F::CHUNK_SIZE;
  typedef Galois::WorkList::AltChunkedFIFO<CHUNK_SIZE, GNode> Inner_WL_ty;
  typedef Galois::WorkList::WLsizeWrapper<Inner_WL_ty> WL_ty;
  typedef PerThreadStorage<UserContextAccess<GNode> > PerThreadUserCtx;

  G& graph;
  F func;
  const char* loopname;
  unsigned nextIndex;

  std::vector<WL_ty*> colorWorkLists;
  PerThreadUserCtx userContexts;

  ChromaticExecutor (G& graph, const F& func, unsigned maxColors, const char* loopname)
    : graph (graph), func (func), loopname (loopname), nextIndex (0) {
    
      assert (maxColors > 0);
      colorWorkLists.resize (maxColors, nullptr);
      
      for (unsigned i = 0; i < maxColors; ++i) {
        colorWorkLists[i] = new WL_ty ();
      }
  }

  ~ChromaticExecutor (void) {
    for (unsigned i = 0; i < colorWorkLists.size (); ++i) {
      delete colorWorkLists[i];
      colorWorkLists[i] = nullptr;
    }
  }

  void push (GNode n) {
    auto& data = graph.getData (n);

    unsigned i = data.color - 1;
    assert (i < colorWorkLists.size ());

    bool expected = false;
    if (data.onWL.compare_exchange_strong (expected, true)) {
      colorWorkLists[i]->push (n);
    }
  }

  WL_ty* chooseLargest (void) {
    WL_ty* nextWL = nullptr;

    unsigned maxSize = 0;
    for (unsigned i = 0; i < colorWorkLists.size (); ++i) {

      size_t s = colorWorkLists[i]->size ();
      if (s > 0 && s > maxSize) {
        maxSize = s;
        nextWL = colorWorkLists[i];
      }
    }

    return nextWL;
  }

  WL_ty* chooseFirst (void) {
    WL_ty* nextWL = nullptr;

    for (unsigned i = 0; i < colorWorkLists.size (); ++i) {
      if (colorWorkLists[i]->size () > 0) {
        nextWL = colorWorkLists[i];
        break;
      }
    }

    return nextWL;
  }

  WL_ty* chooseNext (void) {
    WL_ty* nextWL = nullptr;

    for (unsigned i = 0; i < colorWorkLists.size (); ++i) {

      unsigned j = (nextIndex + i) % colorWorkLists.size ();
      size_t s = colorWorkLists[j]->size ();
      if (s > 0) {
        nextWL = colorWorkLists[j];
        nextIndex = (j + 1) % colorWorkLists.size ();
        break;
      }
    }

    return nextWL;
  }

  struct ApplyOperator {
    typedef int tt_does_not_need_aborts;
    ChromaticExecutor& outer;

    template <typename C>
    void operator () (GNode n, C& ctx) {

      // auto& userCtx = *(outer.userContexts.getLocal ());
// 
      // userCtx.reset ();
      // outer.func (n, userCtx);
      outer.func (n, outer);

      auto& nd = outer.graph.getData (n, Galois::NONE);
      nd.onWL = false;

      // for (auto i = userCtx.getPushBuffer ().begin (), 
          // end_i = userCtx.getPushBuffer ().end (); i != end_i; ++i) {
// 
        // outer.push (*i);
      // }
    }
  };


  template <typename R>
  void execute (const R& range) {

    // fill initial
    do_all_impl (range,
        [this] (GNode n) {
          push (n);
        },
        "fill_initial",
        false );

    unsigned rounds = 0;
    // process until done
    while (true) {
      ++rounds;

      // find non-empty WL
      WL_ty* nextWL = chooseNext ();

      if (nextWL == nullptr) {
        break;
        // double check that all WL are empty
      }

      // run for_each
      typedef Galois::WorkList::ExternalReference<WL_ty> WL;
      GNode* it = 0;

      for_each(it, it,
          ApplyOperator {*this},
          Galois::loopname("ApplyOperator"),
          Galois::wl<WL>(nextWL));

      nextWL->reset_all ();
    }

    std::printf ("ChromaticExecutor: performed %d rounds\n", rounds);

  }

};

template <typename R, typename F, typename G, typename M>
void for_each_det_chromatic (const R& range, const F& func, G& graph, M& dagManager, const char* loopname) {

  Galois::Runtime::getSystemThreadPool ().burnPower (Galois::getActiveThreads ());

  dagManager.colorDAG ();

  ChromaticExecutor<G,F> executor {graph, func, dagManager.getMaxColors (), loopname};

  executor.execute (range);

  Galois::Runtime::getSystemThreadPool ().beKind ();

}

// TODO: logic to choose correct DAG type based on graph type or some graph tag
template <typename R, typename G, typename F>
void for_each_det_chromatic (const R& range, const F& func, G& graph, const char* loopname) {

  typedef typename DAGmanagerInOut<G>::Manager  M;
  M dagManager {graph};

  for_each_det_chromatic (range, func, graph, 
      dagManager, loopname);

}

template <typename G, typename F, typename M>
struct InputGraphDAGexecutor {

  typedef typename G::GraphNode GNode;

  typedef Galois::PerThreadBag<GNode> Bag_ty;

  static const unsigned CHUNK_SIZE = F::CHUNK_SIZE;
  typedef Galois::WorkList::AltChunkedFIFO<CHUNK_SIZE, GNode> Inner_WL_ty;
  typedef Galois::WorkList::WLsizeWrapper<Inner_WL_ty> WL_ty;
  typedef PerThreadStorage<UserContextAccess<GNode> > PerThreadUserCtx;


  G& graph;
  F func;
  M& dagManager;
  const char* loopname;


  PerThreadUserCtx userContexts;

  Bag_ty nextWork;

public:

  InputGraphDAGexecutor (G& graph, const F& func, M& dagManager, const char* loopname)
    :
      graph (graph),
      func (func),
      dagManager (dagManager),
      loopname (loopname)
  {}

  void push (GNode node) {
    auto& nd = graph.getData (node, Galois::NONE);

    bool expected = false;
    if (nd.onWL.compare_exchange_strong (expected, true)) {
      nextWork.push (node);
    }
    //
    // nd.onWL = true;
  }

  struct ApplyOperator {
    typedef int tt_does_not_need_aborts;

    InputGraphDAGexecutor& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {
      // auto& userCtx = *(outer.userContexts.getLocal ());
// 
      // userCtx.reset ();
      // outer.func (src, userCtx);
      outer.func (src, outer);

      G& graph = outer.graph;

      auto& sd = graph.getData (src, Galois::NONE);
      sd.onWL = false;

      // for (auto i = userCtx.getPushBuffer ().begin (), 
          // end_i = userCtx.getPushBuffer ().end (); i != end_i; ++i) {
        // outer.push (*i);
      // }

      auto closure = [&graph, &ctx] (GNode dst) {

        auto& dd = graph.getData (dst, Galois::NONE);

        if (bool (dd.onWL)) {
          // assert (int (dd.indegree) > 0);

          
          unsigned x = --dd.indegree; 
          if (x == 0) {
            ctx.push (dst);
          }
        }

      };


      outer.dagManager.applyToAdj (src, closure);
    }
  };

  template <typename R>
  void execute (const R& range) {
    WL_ty sources;

    dagManager.assignPriority ();

    Galois::do_all_choice (
        range,
        [this] (GNode node) {
          push (node);
        }, 
        "push_initial",
        Galois::doall_chunk_size<CHUNK_SIZE> ());

    Galois::TimeAccumulator t_dag_init;
    Galois::TimeAccumulator t_dag_exec;

    unsigned rounds = 0;
    while (true) {
      ++rounds;

      assert (sources.size () == 0);

      t_dag_init.start ();
      dagManager.initActiveDAG (Galois::Runtime::makeLocalRange (nextWork), sources);
      t_dag_init.stop ();

      nextWork.clear_all_parallel ();

      if (sources.size () == 0) {
        break;
      }

      t_dag_exec.start ();
      typedef Galois::WorkList::ExternalReference<WL_ty> WL;
      typename WL::value_type* it = 0;
      Galois::for_each(it, it,
          ApplyOperator {*this},
          Galois::loopname("ApplyOperator"),
          Galois::wl<WL>(&sources));
      t_dag_exec.stop ();

      sources.reset_all ();

    }

    std::printf ("InputGraphDAGexecutor: performed %d rounds\n", rounds);
    std::printf ("InputGraphDAGexecutor: time taken by dag initialization: %lu\n", t_dag_init.get ());
    std::printf ("InputGraphDAGexecutor: time taken by dag execution: %lu\n", t_dag_exec.get ());
  }


  /*
  struct ApplyOperatorAsync {

    typedef int tt_does_not_need_aborts;

    InputGraphDAGexecutor& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {

      auto& sd = graph.getData (src, Galois::NONE);

      if (bool (sd.onWL)) {
        outer.func (src, dummyCtx); 
      }

      G& graph = outer.graph;

      auto closure = [&graph, &ctx] (GNode dst) {

        auto& dd = graph.getData (dst, Galois::NONE);

        unsigned x = --dd.indegree;
        if (x == 0) {
          ctx.push (dst);
        }
      };

      outer.dagManager.applyToAdj (src, closure);

    }
  };
  */



};



template <typename R, typename F, typename G, typename M>
void for_each_det_edge_flip_ar (const R& range, const F& func, G& graph, M& dagManager, const char* loopname) {

  Galois::Runtime::getSystemThreadPool ().burnPower (Galois::getActiveThreads ());

  InputGraphDAGexecutor<G,F, M> executor {graph, func, dagManager, loopname};

  executor.execute (range);

  Galois::Runtime::getSystemThreadPool ().beKind ();

}

template <typename R, typename F, typename G>
void for_each_det_edge_flip_ar (const R& range, const F& func, G& graph, const char* loopname) {

  typedef typename DAGmanagerInOut<G>::Manager M;
  M dagManager {graph};

  for_each_det_edge_flip_ar (range, func, graph, dagManager, loopname);
}


} // end namespace Runtime
} // end namespace Galois

#endif // GALOIS_RUNTIME_DET_CHROMATIC_H
