#ifndef GALOIS_RUNTIME_DET_CHROMATIC_H
#define GALOIS_RUNTIME_DET_CHROMATIC_H

#include "galois/Reduction.h"
#include "galois/AltBag.h"
#include "galois/DoAllWrap.h"
#include "galois/Galois.h"
#include "galois/Atomic.h"
//#include "galois/GaloisUnsafe.h"

#include "galois/graphs/Graph.h"

#include "galois/worklists/WorkListWrapper.h"
#include "galois/worklists/ExternalReference.h"

#include <atomic>
#include <vector>
#include <type_traits>
#include <random>

namespace galois {
namespace runtime {

enum class InputDAG_ExecTy {
  CHROMATIC,
  EDGE_FLIP,
  TOPO,
  PART,
  HYBRID,
};

namespace cll = llvm::cl;

static cll::opt<int> cutOffColorOpt("cutoff", cll::desc("cut off color for hybrid executor"), cll::init(9999));
static cll::opt<bool> reinitAfterCutOff("reinit", cll::desc("In hybrid execuotr: reinit DAG successors of nodes after cutoff"), cll::init(false));

static cll::opt<int> threadMultFactor("tmult", cll::desc("thread multiplication factor for low parallelism threshold"), cll::init(0));

static cll::opt<InputDAG_ExecTy> inputDAG_ExecTy (
    "executor",
    cll::desc ("Deterministic Executor Type"),
    cll::values (
      clEnumValN (InputDAG_ExecTy::CHROMATIC, "InputDAG_ExecTy::CHROMATIC", "Chromatic Executor"),
      clEnumValN (InputDAG_ExecTy::EDGE_FLIP, "InputDAG_ExecTy::EDGE_FLIP", "Edge Flipping DAG overlayed on input graph"),
      clEnumValN (InputDAG_ExecTy::TOPO, "InputDAG_ExecTy::TOPO", "Edge Flipping DAG overlayed on input graph"),
      clEnumValN (InputDAG_ExecTy::PART, "InputDAG_ExecTy::PART", "Partitioned coarsened DAG overlayed on input graph"),
      clEnumValN (InputDAG_ExecTy::HYBRID, "InputDAG_ExecTy::HYBRID", "Hybrid policies on input DAG"),
      clEnumValEnd),
    cll::init (InputDAG_ExecTy::CHROMATIC));


enum class Priority {
  FIRST_FIT,
  BY_ID,
  RANDOM,
  MIN_DEGREE,
  MAX_DEGREE,
};


static cll::opt<Priority> priorityFunc (
    "priority",
    cll::desc ("choose ordering heuristic"),
    cll::values (
      clEnumValN (Priority::FIRST_FIT, "Priority::FIRST_FIT", "first fit, no priority"),
      clEnumValN (Priority::BY_ID, "Priority::BY_ID", "order by ID modulo some constant"),
      clEnumValN (Priority::RANDOM, "Priority::RANDOM", "uniform random within some small range"),
      clEnumValN (Priority::MIN_DEGREE, "Priority::MIN_DEGREE", "order by min degree first"),
      clEnumValN (Priority::MAX_DEGREE, "Priority::MAX_DEGREE", "order by max degree first"),
      clEnumValEnd),
    cll::init (Priority::BY_ID));

struct BaseDAGdata {
  // std::atomic<unsigned> onWL;
  GAtomic<int> onWL;
  // std::atomic<unsigned> indegree;
  GAtomic<int> indegree;
  int indeg_backup;


  unsigned id;
  unsigned priority;
  int color;

  explicit BaseDAGdata (unsigned id) :
    onWL (0),
    indegree (0),
    indeg_backup (0),
    id (id),
    priority (0),
    color (-1)
  {}
};


template <typename ND>
struct DAGdataComparator {

  static int compare3val (const ND& left, const ND& right) {
    int r = left.priority - right.priority;
    if (r != 0) {
      return r;
    } else {
      return (r = left.id - right.id);
    }
  }

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

struct InputDAGdata: public BaseDAGdata {


  unsigned numSucc;
  unsigned* dagSucc;



  explicit InputDAGdata (unsigned id=0):
    BaseDAGdata (id),
    numSucc (0),
    dagSucc (nullptr)
  {}

  struct VisitDAGsuccessors {

    template <typename GNode, typename ND, typename F>
    void operator () (GNode src, ND& sd, F& f) {

      for (unsigned i = 0; i < sd.numSucc; ++i) {
        GNode dst = sd.dagSucc[i];
        f (dst);
      }
    }
  };

};

struct InputDAGdataInOut: public BaseDAGdata {


  // offset where dag successors end and predecessors begin
  ptrdiff_t dagSuccEndIn;
  ptrdiff_t dagSuccEndOut;


  explicit InputDAGdataInOut (unsigned id=0):
    BaseDAGdata (id),
    dagSuccEndIn (0),
    dagSuccEndOut (0)
  {}

};

struct InputDAGdataDirected: public InputDAGdata {

  // TODO: change to vector with pow 2 alloc
  typedef galois::gdeque<unsigned, 64> AdjList;

  AdjList incoming;

  InputDAGdataDirected (unsigned id): InputDAGdata (id) {}

  void addIncoming (unsigned n) {
    assert (std::find (incoming.begin (), incoming.end (), n) == incoming.end ());
    incoming.push_back (n);
  }

};


struct TaskDAGdata: public BaseDAGdata {

  SimpleRuntimeContext* taskCtxt;

  explicit TaskDAGdata (unsigned id=0):
    BaseDAGdata (id),
    taskCtxt (nullptr)
  {}

};


template <typename G, typename A, typename D>
struct DAGmanagerBase {

protected:
  static const bool DEBUG = false;

  static const unsigned DEFAULT_CHUNK_SIZE = 4;

  typedef typename G::GraphNode GNode;
  typedef typename G::node_data_type ND;

  typedef galois::PerThreadVector<bool> PerThrdColorVec;

  G& graph;
  A visitAdj;
  D visitDAGsucc;
  PerThrdColorVec perThrdColorVec;
  galois::GReduceMax<int> maxColors;
  bool initialized = false;

  DAGmanagerBase (G& graph, const A& visitAdj, const D& visitDAGsucc=D())
    : graph (graph), visitAdj (visitAdj), visitDAGsucc (visitDAGsucc)
  {
    // // mark 0-th color as taken
    // for (unsigned i = 0; i < perThrdColorVec.numRows (); ++i) {
      // auto& forbiddenColors = perThrdColorVec.get(i);
      // forbiddenColors.resize (1, 0);
      // forbiddenColors[0] = 0;
    // }

    if (DEBUG) {
      fprintf (stderr, "WARNING: DAGmanagerBase DEBUG mode on, timing may be off\n");
    }
  }


public:

  template <typename F>
  void applyToAdj (GNode src, F& f, const galois::MethodFlag& flag=galois::MethodFlag::UNPROTECTED) {
    visitAdj (src, f, flag);
  }

  size_t countAdj (GNode src) {
    return visitAdj.count (src);
  }


  template <typename F>
  void applyToDAGsucc (GNode src, ND& srcData, F& f) {
    visitDAGsucc (src, srcData, f);
  }

  size_t countDAGsucc (GNode src) {
    return visitDAGsucc.count (src);
  }

  template <typename P>
  void initDAG (P postInit) {

    galois::StatTimer t("initDAG");

    t.start ();

    assignPriority ();

    galois::runtime::do_all_gen (
        galois::runtime::makeLocalRange (graph),
        [this, &postInit] (GNode src) {

          auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);

          assert (sd.indegree == 0);
          int indeg = 0;

          auto countDegClosure = [this, &sd, &indeg] (GNode dst) {
            auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);

            int c = DAGdataComparator<ND>::compare3val (dd, sd);
            if (c < 0) { // dd < sd
              ++indeg;
            }
          };

          applyToAdj (src, countDegClosure);

          sd.indegree = indeg;
          sd.indeg_backup = sd.indegree;


          postInit (graph, src, sd);

        },
        "init-DAG",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    initialized = true;

    t.stop ();
  }


  // made non-static to fix clang linking error
  const int IS_ACTIVE = 2;

  template <typename R, typename W>
  void reinitActiveDAG (const R& range, W& sources) {

    galois::StatTimer t ("reinitActiveDAG");

    t.start ();

    GALOIS_ASSERT (initialized);

    galois::runtime::do_all_gen (
        range,
        [this] (GNode src) {
          auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);

          // if (sd.onWL == 0) {
            // std::printf ("assertion going to fail sd.onWL==0, for id %d\n", sd.id);
          // }

          assert (sd.onWL > 0);
          sd.indegree = 0;
          sd.onWL = IS_ACTIVE;
        },
        "reinitActiveDAG-0",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    galois::runtime::do_all_gen (
        range,
        [this] (GNode src) {

          auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
          assert (sd.onWL > 0);

          auto closure = [this] (GNode dst) {
            auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
            if (int(dd.onWL) == IS_ACTIVE) {
              ++(dd.indegree);
            }
          };

          applyToDAGsucc (src, sd, closure);
        },
        "reinitActiveDAG-1",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    galois::runtime::do_all_gen (
        range,
        [this, &sources] (GNode src) {
          auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
          assert (sd.onWL > 0);
          // std::printf ("active node %d, with indegree %d\n", sd.id, int(sd.indegree));
          if (int(sd.onWL) == IS_ACTIVE && sd.indegree == 0) {
            sources.push (src);

            // if (DEBUG) {
              // auto closure = [this] (GNode dst) {
                // auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
                // if (dd.onWL > 0) {
                  // assert (int (dd.indegree) > 0);
                // }
              // };
//
              // applyToDAGsucc (src, sd, closure);
            // }
          }
        },
        "reinitActiveDAG-2",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    t.stop ();
  }

  template <typename F, typename U>
  struct ActiveDAGoperator {

    typedef int tt_does_not_need_aborts;

    F& func;
    U& userCtx;
    DAGmanagerBase& dagManager;
    galois::GAccumulator<size_t>& edgesVisited;
    galois::GAccumulator<size_t>& edgesFlipped;


    template <typename C>
    void operator () (GNode src, C& ctx) {

      auto& sd = dagManager.graph.getData (src, galois::MethodFlag::UNPROTECTED);
      assert (sd.onWL == dagManager.IS_ACTIVE);
      sd.onWL = 0;

      // std::printf ("edge-flip executing %d\n", sd.id);

      func (src, userCtx);

      auto closure = [this, &ctx] (GNode dst) {

        auto& dd = dagManager.graph.getData (dst, galois::MethodFlag::UNPROTECTED);
        edgesVisited += 1;

        if (int (dd.onWL) == dagManager.IS_ACTIVE) { // is a succ in active dag
          // assert (dd.onWL > 0);
          edgesFlipped += 1;

          int x = --dd.indegree;
          assert (x >= 0);

          if (x == 0) {
            ctx.push (dst);
          }
        }
      };

      dagManager.applyToDAGsucc (src, sd, closure);
    }

  };

  template <typename R, typename F, typename U, typename CS = galois::chunk_size<DEFAULT_CHUNK_SIZE>>
  void runActiveDAGcomp (const R& range, F func, U& userCtx, const char* loopname, const CS& cs = CS ()) {

    using WL_ty =  galois::worklists::AltChunkedFIFO<CS::value>;

    GALOIS_ASSERT (initialized);


    char str[256];
    sprintf (str, "%s-runActiveDAGcomp", loopname);

    galois::StatTimer t(str);
    t.start ();


    WL_ty sources;
    reinitActiveDAG (range, sources);

    galois::GAccumulator<size_t> edgesVisited;
    galois::GAccumulator<size_t> edgesFlipped;


    typedef galois::worklists::ExternalReference<WL_ty> WL;
    typename WL::value_type* it = 0;
    galois::for_each(galois::iterate(it, it),
        ActiveDAGoperator<F, U> {func, userCtx, *this, edgesVisited, edgesFlipped},
        galois::loopname(loopname),
                     galois::wl<WL>(std::ref(sources)));

    // std::printf ("edgesVisited: %zd, edgesFlipped: %zd\n", edgesVisited.reduceRO (), edgesFlipped.reduceRO ());

    reportStat_Single (loopname, "heavy-edges-visited", edgesVisited.reduce());
    reportStat_Single (loopname, "heavy-edges-flipped", edgesFlipped.reduce());

    t.stop ();

  }


  template <typename W>
  void resetDAG (W& sources) {

    GALOIS_ASSERT (initialized);

    galois::runtime::do_all_gen ( galois::runtime::makeLocalRange (graph),
        [this, &sources] (GNode src) {
          auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
          sd.indegree = sd.indeg_backup;

          if (sd.indegree == 0) {
            sources.push (src);
          }
        },
        "resetDAG",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

  }

  struct FakeBag {
    void push (const GNode&) const {}
  };

  void resetDAG (void) {
    FakeBag b;
    resetDAG (b);
  }

  template <typename W>
  void collectSources (W& sources) {

    GALOIS_ASSERT (initialized);

    galois::runtime::do_all_gen ( galois::runtime::makeLocalRange (graph),
        [this, &sources] (GNode src) {
          auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
          // assert (sd.indegree == sd.indeg_backup);
          if (sd.indegree == 0) {
            sources.push (src);
          }
        },
        "collect-sources",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());
  }

  template <typename F>
  struct RunDAGcomp {
    typedef int tt_does_not_need_aborts;
    DAGmanagerBase& outer;
    F& func;

    template <typename C>
    void operator () (GNode src, C& ctx) {

      auto& sd = outer.graph.getData (src, galois::MethodFlag::UNPROTECTED);
      assert (sd.indegree == 0);

      func (src);

      auto dagClosure = [this, &ctx] (GNode dst) {
        auto& dd = outer.graph.getData (dst, galois::MethodFlag::UNPROTECTED);

        int x = --(dd.indegree);
        // assert (x >= 0); // FIXME
        if (x == 0) {
          ctx.push (dst);
        }
      };

      outer.applyToDAGsucc (src, sd, dagClosure);

    }
  };

  template <typename F, typename W>
  void runDAGcomputationImpl (F func, W& initWL, const char* loopname) {

    GALOIS_ASSERT (initialized);

    using WL_ty = galois::worklists::ExternalReference<W>;
    typename WL_ty::value_type* it = nullptr;

    galois::for_each (galois::iterate(it, it),
        RunDAGcomp<F> {*this, func},
        galois::loopname (loopname),
                      galois::wl<WL_ty> (std::ref(initWL)));

  }

  template <typename F, typename B, typename CS = galois::chunk_size<DEFAULT_CHUNK_SIZE> >
  void runDAGcomputation (F func, B& sources, const char* loopname, const CS& cs = CS ()) {

    galois::StatTimer t (loopname);

    t.start ();

    using WL_ty =  galois::worklists::AltChunkedFIFO<CS::value>;

    WL_ty initWL;

    galois::runtime::do_all_gen (
        makeLocalRange (sources),
        [&initWL] (typename WL_ty::value_type src) {
          initWL.push (src);
        },
        "copy-sources",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    runDAGcomputationImpl (func, initWL, loopname);

    t.stop ();
  }

  template <typename F, typename CS = galois::chunk_size<DEFAULT_CHUNK_SIZE> >
  void runDAGcomputation (F func, const char* loopname, const CS& cs = CS ()) {

    galois::StatTimer t (loopname);

    t.start ();
    using WL_ty =  galois::worklists::AltChunkedFIFO<CS::value>;

    WL_ty sources;

    collectSources (sources);
    runDAGcomputationImpl (func, sources, loopname);

    t.stop ();
  }

  void assignIDs (void) {

    const size_t numNodes = graph.size ();
    galois::on_each (
        [&] (const unsigned tid, const unsigned numT) {

          size_t num_per = (numNodes + numT - 1) / numT;
          size_t beg = tid * num_per;
          size_t end = std::min (numNodes, (tid + 1) * num_per);

          auto it_beg = graph.begin ();
          std::advance (it_beg, beg);

          auto it_end = it_beg;
          std::advance (it_end, (end - beg));

          for (; it_beg != it_end; ++it_beg) {
            auto& nd = graph.getData (*it_beg, galois::MethodFlag::UNPROTECTED);
            nd.id = beg++;
          }
        },
        galois::loopname ("assign-ids"));
  }


  template <typename F>
  void assignPriorityHelper (const F& nodeFunc) {
    galois::runtime::do_all_gen (
        galois::runtime::makeLocalRange (graph),
        [&] (GNode node) {
          nodeFunc (node);
        },
        "assign-priority",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());
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
    auto byId = [this] (GNode node) {
      auto& nd = graph.getData (node, galois::MethodFlag::UNPROTECTED);
      nd.priority = nd.id % MAX_LEVELS;
    };

    galois::substrate::PerThreadStorage<RNG>  perThrdRNG;

    // TODO: non-deterministic at the moment
    // can be fixed by making thread K call the generator
    // N times, where N is sum of calls of all threads < K
    auto randPri = [this, &perThrdRNG] (GNode node) {
      auto& rng = *(perThrdRNG.getLocal ());
      auto& nd = graph.getData (node, galois::MethodFlag::UNPROTECTED);
      nd.priority = rng ();
    };


    auto minDegree = [this] (GNode node) {
      auto& nd = graph.getData (node, galois::MethodFlag::UNPROTECTED);
      nd.priority = visitAdj.count (node);
      assert (nd.priority >= 0);
    };

    const size_t numNodes = graph.size ();
    auto maxDegree = [this, &numNodes] (GNode node) {
      auto& nd = graph.getData (node, galois::MethodFlag::UNPROTECTED);
      nd.priority = std::max (size_t (0), numNodes - visitAdj.count (node));
    };

    galois::StatTimer t_priority ("priority assignment time: ");

    t_priority.start ();

    switch (priorityFunc) {
      case Priority::FIRST_FIT:
        // do nothing
        break;

      case Priority::BY_ID:
        assignPriorityHelper (byId);
        break;

      case Priority::RANDOM:
        assignPriorityHelper (randPri);
        break;

      case Priority::MIN_DEGREE:
        assignPriorityHelper (minDegree);
        break;

      case Priority::MAX_DEGREE:
        assignPriorityHelper (maxDegree);
        break;

      default:
        std::abort ();
    }

    t_priority.stop ();
  }


  void colorNode (GNode src) {
    auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
    assert (sd.indegree == 0);
    assert (sd.color == -1); // uncolored

    auto& forbiddenColors = perThrdColorVec.get ();
    std::fill (forbiddenColors.begin (), forbiddenColors.end (), false);


    auto colorClosure = [this, &forbiddenColors, &sd] (GNode dst) {
      auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);

      if (int(forbiddenColors.size ()) <= dd.color) {
        forbiddenColors.resize (dd.color + 1, false);
      }
      // std::printf ("Neighbor %d has color %d\n", dd.id, dd.color);

      if (dd.color != -1) {
        forbiddenColors[dd.color] = true;
      }

    };

    applyToAdj (src, colorClosure);

    for (size_t i = 0; i < forbiddenColors.size (); ++i) {
      if (!forbiddenColors[i]) {
        sd.color = i;
        break;
      }
    }

    if (sd.color == -1) {
      sd.color = forbiddenColors.size ();
    }
    maxColors.update (sd.color);


    // std::printf ("Node %d assigned color %d\n", sd.id, sd.color);

  }

  void verifyColoring (void) {
    galois::StatTimer t_verify ("Coloring verification time: ");

    t_verify.start ();
    std::printf ("WARNING: verifying Coloring, timing will be off\n");

    galois::GReduceLogicalOR foundError;

    galois::runtime::do_all_gen (
        galois::runtime::makeLocalRange (graph),
        [&] (GNode src) {

          auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
          if (sd.color == -1) {
            std::fprintf (stderr, "ERROR: src %d found uncolored\n", sd.id);
            foundError.update (true);
          }

          auto visitAdj = [&] (GNode dst) {
            auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
            if (sd.id != dd.id && sd.color == dd.color) {
              foundError.update (true);
              std::fprintf (stderr, "ERROR: nodes %d and %d have the same color\n",
                sd.id, dd.id);

            } else {
              if (sd.id == dd.id) { assert (src == dst); };
            }
          };

          applyToAdj (src, visitAdj);

        },
        "check-coloring",
        galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    t_verify.stop ();

    if (foundError.reduce()) {
      GALOIS_DIE ("ERROR! Coloring verification failed!\n");
    } else {
      printf ("OK! Coloring verification succeeded!\n");
    }
  }

  void colorDAG (void) {

    GALOIS_ASSERT (initialized);

    auto func = [this] (GNode src) { this->colorNode (src); };
    runDAGcomputation (func, "color-DAG");
    std::printf ("DAG colored with %d colors\n", getNumColors ());
    if (false || DEBUG) {
      verifyColoring ();
    }
  }

  unsigned getNumColors (void) const {
    return maxColors.reduce() + 1;
  }

};

template <typename G>
struct DAGmanagerInOut {
  typedef typename G::GraphNode GNode;

  struct VisitAdjacent {
    G& graph;

    template <typename F>
    void operator () (GNode src, F& func, const galois::MethodFlag& flag) {

      for (auto i = graph.in_edge_begin (src, flag)
          , end_i = graph.in_edge_end (src, flag); i != end_i; ++i) {
        GNode dst = graph.getInEdgeDst (i);
        func (dst);
      }

      for (auto i = graph.edge_begin (src, flag)
          , end_i = graph.edge_end (src, flag); i != end_i; ++i) {
        GNode dst = graph.getEdgeDst (i);
        func (dst);
      }
    }

    size_t count (GNode src) const {
      ptrdiff_t in = std::distance (
          graph.in_edge_begin (src, MethodFlag::UNPROTECTED),
          graph.in_edge_end (src, MethodFlag::UNPROTECTED));
      assert (in >= 0);

      ptrdiff_t out = std::distance (
          graph.edge_begin (src, MethodFlag::UNPROTECTED),
          graph.edge_end (src, MethodFlag::UNPROTECTED));
      assert (out >= 0);

      return size_t (in + out);
    }
  };

  struct VisitDAGsuccessors {
    G& graph;

    template <typename ND, typename F>
    void operator () (GNode src, ND& sd, F& func) {

      for (auto i = graph.in_edge_begin (src, galois::MethodFlag::UNPROTECTED)
          , end_i = i + sd.dagSuccEndIn; i != end_i; ++i) {

        assert (i <= end_i);
        GNode dst = graph.getInEdgeDst (i);
        func (dst);
      }

      for (auto i = graph.edge_begin (src, galois::MethodFlag::UNPROTECTED)
          , end_i = i + sd.dagSuccEndOut; i != end_i; ++i) {

        assert (i <= end_i);
        GNode dst = graph.getEdgeDst (i);
        func (dst);
      }
    }

    size_t count (GNode src) const {

      auto& sd = graph.getData (src, MethodFlag::UNPROTECTED);
      return (sd.dagSuccEndIn + sd.dagSuccEndOut);
    }

  };

  struct VisitDAGpredecessors {
    G& graph;

    template <typename ND, typename F>
    void operator () (GNode src, ND& sd, F& func) {

      for (auto i = graph.in_edge_begin (src, MethodFlag::UNPROTECTED) + sd.dagSuccEndIn
          , end_i = graph.in_edge_end (src, MethodFlag::UNPROTECTED); i != end_i; ++i) {

        assert (i <= end_i);
        GNode dst = graph.getInEdgeDst (i);
        func (dst);
      }

      for (auto i = graph.edge_begin (src, MethodFlag::UNPROTECTED) + sd.dagSuccEndOut
          , end_i = graph.edge_end (src, MethodFlag::UNPROTECTED); i != end_i; ++i) {

        assert (i <= end_i);
        GNode dst = graph.getEdgeDst (i);
        func (dst);
      }
    }
  };

  template <typename ND>
  struct Predicate {
    G& graph;
    const ND& srcData;

    bool operator () (GNode dst) const {
      const auto& dstData = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
      return DAGdataComparator<ND>::compare3val (srcData, dstData) < 0;
    }
  };

  struct InitDAGoffsets {


    template <typename ND>
    void operator () (G& graph, GNode src, ND& sd) {

      Predicate<ND> pred {graph, sd};

      ptrdiff_t out_off = graph.partition_neighbors (src, pred);
      ptrdiff_t in_off = graph.partition_in_neighbors (src, pred);

      sd.dagSuccEndOut = out_off;
      sd.dagSuccEndIn = in_off;

      static const bool VERIFY = false;

      if (VERIFY) {

        auto funcSucc = [&pred] (GNode dst) {
          assert (pred (dst));
        };

        VisitDAGsuccessors visitDAGsucc {graph};

        visitDAGsucc (src, sd, funcSucc);

        auto funcPred = [&pred] (GNode dst) {
          assert (!pred (dst));
        };

        VisitDAGpredecessors visitDAGpred{graph};
        visitDAGpred (src, sd, funcPred);

      }
    }
  };

  typedef DAGmanagerBase<G, VisitAdjacent, VisitDAGsuccessors> Base_ty;

  struct Manager: public Base_ty {

    Manager (G& graph): Base_ty {graph, VisitAdjacent {graph}, VisitDAGsuccessors {graph}}
    {}

    void initDAG (void) {
      Base_ty::initDAG (InitDAGoffsets ());
    }

    template <typename R, typename P>
    void filterDAGsucc (const R& range, const P& pred) {

      G& graph = Base_ty::graph;

      galois::runtime::do_all_gen (
          range,
          [&graph, &pred] (GNode src) {
            auto& sd = graph.getData (src, MethodFlag::UNPROTECTED);

            auto combinedPred = [&graph, &sd, &pred] (GNode dst) {

              auto& dd = graph.getData (dst, MethodFlag::UNPROTECTED);

              using ND = typename std::remove_reference<decltype (dd)>::type;

              return pred (dd) && (DAGdataComparator<ND>::compare3val (sd, dd) < 0);
            };


            ptrdiff_t old_out = sd.dagSuccEndOut;
            ptrdiff_t old_in = sd.dagSuccEndIn;

            ptrdiff_t out_off = graph.partition_neighbors (src, combinedPred);
            ptrdiff_t in_off = graph.partition_in_neighbors (src, combinedPred);

            GALOIS_ASSERT (out_off <= old_out);
            GALOIS_ASSERT (in_off <= old_in);

            sd.dagSuccEndOut = out_off;
            sd.dagSuccEndIn = in_off;

          },
          "reinitDAG-cutoff",
          galois::chunk_size<Base_ty::DEFAULT_CHUNK_SIZE> ());

    }

  };

};

template <typename G, typename A>
struct DAGmanagerDefault: public DAGmanagerBase<G, A, InputDAGdata::VisitDAGsuccessors> {

  using Base = DAGmanagerBase<G, A, InputDAGdata::VisitDAGsuccessors>;
  using GNode = typename G::GraphNode;
  using ND = typename G::node_data_type;

  galois::runtime::Pow_2_BlockAllocator<unsigned> dagSuccAlloc;

  DAGmanagerDefault (G& graph, const A& visitAdj)
    : Base (graph, visitAdj, InputDAGdata::VisitDAGsuccessors ())
  {}

  void initDAG (void) {

    auto postInit = [this] (G& graph, GNode src, ND& sd) {

      unsigned outdeg = 0;

      auto countDegClosure = [&graph, &sd, &outdeg] (GNode dst) {
        auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);

        int c = DAGdataComparator<ND>::compare3val (dd, sd);
        if (c > 0) { // sd < dd
          ++outdeg;
        }
      };

      Base::applyToAdj (src, countDegClosure);

      sd.numSucc = outdeg;
      sd.dagSucc = dagSuccAlloc.allocate (sd.numSucc);
      assert (sd.dagSucc != nullptr);

      unsigned i = 0;
      auto fillDAGsucc = [&graph, &sd, &i] (GNode dst) {
        auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);

        int c = DAGdataComparator<ND>::compare3val (dd, sd);
        if (c > 0) { // dd > sd
          sd.dagSucc[i++] = dst;
        }
      };

      Base::applyToAdj (src, fillDAGsucc);
      assert (i == sd.numSucc);
    };

    Base::initDAG (postInit);
  }

  void freeDAGdata (void) {
    galois::runtime::do_all_gen ( galois::runtime::makeLocalRange (Base::graph),
        [this] (GNode src) {
          auto& sd = Base::graph.getData (src, galois::MethodFlag::UNPROTECTED);
          dagSuccAlloc.deallocate (sd.dagSucc, sd.numSucc);
          sd.numSucc = 0;
          sd.dagSucc = nullptr;
        },
        "freeDAGdata",
        galois::chunk_size<Base::DEFAULT_CHUNK_SIZE> ());

  }

  ~DAGmanagerDefault (void) {
    freeDAGdata ();
  }

};






template <typename G>
struct DAGvisitorUndirected {

  typedef typename G::GraphNode GNode;

  struct VisitAdjacent {
    G& graph;

    template <typename F>
    void operator () (GNode src, F& func, const galois::MethodFlag& flag) {

      for (auto i = graph.edge_begin (src, flag)
           , end_i = graph.edge_end (src, flag); i != end_i; ++i) {

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
    // galois::runtime::do_all_gen (
        // galois::runtime::makeLocalRange (graph),
        // [this] (GNode src) {
        //
          // auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
          //
          // unsigned addAmt = 0;
          // for (Graph::edge_iterator e = graph.edge_begin (src, galois::MethodFlag::UNPROTECTED),
              // e_end = graph.edge_end (src, galois::MethodFlag::UNPROTECTED); e != e_end; ++e) {
            // GNode dst = graph.getEdgeDst (e);
            // auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
//
            // if (src != dst) {
              // dd.addPred (src);
            // }
          // }
//
        // },
        // "initDAG",
        // galois::chunk_size<DEFAULT_CHUNK_SIZE> ());
//
  // }
//
// };



template <typename G, typename M, typename F>
struct ChromaticExecutor {

  typedef typename G::GraphNode GNode;

  static const unsigned CHUNK_SIZE = F::CHUNK_SIZE;
  typedef galois::worklists::AltChunkedFIFO<CHUNK_SIZE, GNode> Inner_WL_ty;
  typedef galois::worklists::WLsizeWrapper<Inner_WL_ty> WL_ty;
  typedef substrate::PerThreadStorage<UserContextAccess<GNode> > PerThreadUserCtx;

  G& graph;
  M& dagManager;
  F func;
  const char* loopname;
  unsigned nextIndex;

  std::vector<WL_ty*> colorWorkLists;
  PerThreadUserCtx userContexts;

  ChromaticExecutor (G& graph, M& dagManager, const F& func, const char* loopname)
    : graph (graph), dagManager (dagManager), func (func), loopname (loopname), nextIndex (0) {

      dagManager.initDAG ();
      dagManager.colorDAG ();
      unsigned numColors = dagManager.getNumColors ();

      assert (numColors > 0);
      colorWorkLists.resize (numColors, nullptr);

      for (unsigned i = 0; i < numColors; ++i) {
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

    unsigned i = data.color;
    assert (i < colorWorkLists.size ());

    int expected = 0;
    if (data.onWL.cas (expected, 1)) {
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
    typedef int tt_does_not_need_push;
    ChromaticExecutor& outer;

    template <typename C>
    void operator () (GNode n, C& ctx) {

      // auto& userCtx = *(outer.userContexts.getLocal ());
//
      // userCtx.reset ();
      // outer.func (n, userCtx);
      auto& nd = outer.graph.getData (n, galois::MethodFlag::UNPROTECTED);
      nd.onWL = 0;
      outer.func (n, outer);


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
    runtime::do_all_gen (range,
        [this] (GNode n) {
          push (n);
        },
        std::make_tuple(
        galois::loopname("fill_initial")));

    unsigned rounds = 0;
    // process until done
    while (true) {

      // find non-empty WL
      // TODO: cmd line option to choose
      // the order in which worklists are picked
      WL_ty* nextWL = chooseNext ();

      if (nextWL == nullptr) {
        break;
        // double check that all WL are empty
      }

      ++rounds;

      // run for_each
      typedef galois::worklists::ExternalReference<WL_ty> WL;
      GNode* it = 0;

      for_each(galois::iterate(it, it),
          ApplyOperator {*this},
          galois::loopname(loopname),
               galois::wl<WL>(std::ref(*nextWL)));

      nextWL->reset_all ();
    }

    std::printf ("ChromaticExecutor: performed %d rounds\n", rounds);

  }

};

template <typename R, typename F, typename G, typename M>
void for_each_det_chromatic (const R& range, const F& func, G& graph, M& dagManager, const char* loopname) {

  galois::substrate::getThreadPool ().burnPower (galois::getActiveThreads ());


  ChromaticExecutor<G, M, F> executor {graph, dagManager, func, loopname};

  executor.execute (range);

  galois::substrate::getThreadPool ().beKind ();

}

// TODO: logic to choose correct DAG type based on graph type or some graph tag
template <typename R, typename G, typename F>
void for_each_det_chromatic (const R& range, const F& func, G& graph, const char* loopname) {

  typedef typename DAGmanagerInOut<G>::Manager  M;
  M dagManager {graph};

  for_each_det_chromatic (range, func, graph,
      dagManager, loopname);

}


template <typename G, typename M, typename F>
struct ChromaticReuseExecutor {

  typedef typename G::GraphNode GNode;

  static const unsigned CHUNK_SIZE = F::CHUNK_SIZE;
  typedef galois::worklists::AltChunkedFIFO<CHUNK_SIZE, GNode> WL_ty;
  typedef galois::PerThreadBag<GNode> Bag_ty;

  G& graph;
  M& dagManager;
  F func;

  std::string loopname;

  std::vector<Bag_ty*> colorBags;

  ChromaticReuseExecutor (G& graph, M& dagManager, const F& func, const char* loopname) :
    graph {graph},
    dagManager {dagManager},
    func (func),
    loopname {loopname}
  {}


  void push_initial (GNode n) {
    auto& data = graph.getData (n, galois::MethodFlag::UNPROTECTED);

    unsigned i = data.color;
    assert (i < colorBags.size ());

    int expected = 0;
    if (data.onWL.cas (expected, 1)) {
      colorBags[i]->push (n);
    }
  }

  template <typename R>
  void initialize (const R& range) {
    StatTimer t_init ("ChromaticReuseExecutor: coloring and bucket initialization time:");

    t_init.start ();

    dagManager.initDAG ();
    dagManager.colorDAG ();

    unsigned numColors = dagManager.getNumColors ();

    assert (colorBags.size () == 0);
    colorBags.resize (numColors, nullptr);

    for (unsigned i = 0; i < numColors; ++i) {
      assert (colorBags[i] == nullptr);
      colorBags[i] = new Bag_ty ();
    }


    galois::do_all_choice (
        range,
        [this] (GNode node) {
          push_initial (node);
        },
        "push_initial",
        galois::chunk_size<CHUNK_SIZE> ());

    t_init.stop ();
  }

  void push (GNode n) {
    GALOIS_DIE ("push not supported");
  }

  struct ApplyOperator {

    ChromaticReuseExecutor& outer;

    void operator () (GNode n) {
      outer.func (n, outer);
    }

  };


  void execute (void) {


    StatTimer t_exec ("ChromaticReuseExecutor: execution time:");

    t_exec.start ();

    for (auto i = colorBags.begin (), end_i = colorBags.end ();
        i != end_i; ++i) {

      assert (*i != nullptr);

      galois::do_all_choice (makeLocalRange (**i),
          ApplyOperator {*this},
          loopname,
          galois::chunk_size<CHUNK_SIZE> ());

    }

    t_exec.stop ();


  }

  void resetDAG (void) const {}

};


template <typename G, typename M, typename F>
struct InputGraphDAGreuseExecutor {

  typedef typename G::GraphNode GNode;

  typedef galois::PerThreadBag<GNode> Bag_ty;

  static const unsigned CHUNK_SIZE = F::CHUNK_SIZE;
  typedef galois::worklists::AltChunkedFIFO<CHUNK_SIZE, GNode> WL_ty;


  G& graph;
  M& dagManager;
  F func;
  std::string loopname;

  Bag_ty initialSources;

  InputGraphDAGreuseExecutor (G& graph, M& dagManager, const F& func, const char* loopname)
    :
      graph (graph),
      dagManager (dagManager),
      func (func),
      loopname (loopname)
  {}

  template <typename R>
  void push_initial (const R& range) {

    do_all_choice (
        range,
        [this] (GNode node) {
          auto& sd = graph.getData (node, galois::MethodFlag::UNPROTECTED);
          sd.onWL = 1;
        },
        "push_initial",
        galois::chunk_size<CHUNK_SIZE> ());

  }

  // assumes all nodes are active
  void initialize (void) {
    StatTimer t_init ("InputGraphDAGreuseExecutor: initialization time:");

    t_init.start ();

    push_initial (makeLocalRange (graph));

    dagManager.initDAG ();
    dagManager.collectSources (initialSources);

    t_init.stop ();
  }

  template <typename R>
  void initialize (const R& range) {
    StatTimer t_init ("InputGraphDAGreuseExecutor: initialization time:");

    t_init.start ();

    push_initial (range);

    dagManager.initDAG ();

    dagManager.reinitActiveDAG (range, initialSources);

    t_init.stop ();
  }

  void push (GNode n) {
    GALOIS_DIE ("push not supported");
  }


  // TODO: revisit the logic here. Should be using runActiveDAGcomp?
  void execute (void) {

    auto f = [this] (GNode src) { this->func (src, *this); };
    dagManager.runDAGcomputation (f, initialSources, loopname, galois::chunk_size<CHUNK_SIZE> ());
  }

  void resetDAG (void) {
    dagManager.resetDAG ();
  }

  template <typename R>
  void reinitActiveDAG (const R& range) {

    initialSources.clear_all_parallel ();
    dagManager.reinitActiveDAG (range, initialSources);
  }

};

template <typename G, typename F, typename M>
struct InputGraphDAGexecutor {

  typedef typename G::GraphNode GNode;

  typedef galois::PerThreadBag<GNode> Bag_ty;

  static const unsigned CHUNK_SIZE = F::CHUNK_SIZE;
  typedef galois::worklists::AltChunkedFIFO<CHUNK_SIZE, GNode> Inner_WL_ty;
  typedef galois::worklists::WLsizeWrapper<Inner_WL_ty> WL_ty;
  typedef substrate::PerThreadStorage<UserContextAccess<GNode> > PerThreadUserCtx;



  G& graph;
  F func;
  M& dagManager;
  const char* loopname;


  PerThreadUserCtx userContexts;

  Bag_ty* currWL;
  Bag_ty* nextWL;

public:

  InputGraphDAGexecutor (G& graph, const F& func, M& dagManager, const char* loopname)
    :
      graph (graph),
      func (func),
      dagManager (dagManager),
      loopname (loopname)
  {}

  void push (GNode node) {
    assert (nextWL != nullptr);

    auto& nd = graph.getData (node, galois::MethodFlag::UNPROTECTED);

    int expected = 0;
    if (nd.onWL.cas (expected, 1)) {
      nextWL->push (node);
    }
  }

  // struct ApplyOperator {
    // typedef int tt_does_not_need_aborts;
//
    // InputGraphDAGexecutor& outer;
//
    // template <typename C>
    // void operator () (GNode src, C& ctx) {
//
      // G& graph = outer.graph;
//
      // auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
      // assert (sd.onWL > 0); // is active
      // sd.onWL = 0;
//
      // outer.func (src, outer);
//
      // auto closure = [&graph, &ctx] (GNode dst) {
//
        // auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
//
        // if (int (dd.indegree) > 0) { // is a succ in active dag
          // assert (dd.onWL > 0);
//
          // int x = --dd.indegree;
          // assert (x >= 0);
//
          // if (x == 0) {
            // ctx.push (dst);
          // }
        // }
      // };
//
      // outer.dagManager.applyToDAGsucc (src, sd, closure);
    // }
  // };

  template <typename R>
  void execute (const R& range) {

    currWL = new Bag_ty ();
    nextWL = new Bag_ty ();


    galois::do_all_choice (
        range,
        [this] (GNode node) {
          push (node);
        },
        "push_initial",
        galois::chunk_size<CHUNK_SIZE> ());

    dagManager.initDAG ();

    unsigned rounds = 0;
    while (!nextWL->empty_all ()) {

      ++rounds;
      std::swap (currWL, nextWL);

      nextWL->clear_all_parallel ();

      dagManager.runActiveDAGcomp (
          makeLocalRange (*currWL),
          func,
          *this,
          loopname,
          galois::chunk_size<CHUNK_SIZE> ());
    }

    delete currWL; currWL = nullptr;
    delete nextWL; nextWL = nullptr;

    std::printf ("InputGraphDAGexecutor: performed %d rounds\n", rounds);

    // galois::TimeAccumulator t_dag_init;
    // galois::TimeAccumulator t_dag_exec;
//
    // t_dag_init.start ();
    // dagManager.initDAG ();
    // t_dag_init.stop ();
//
    // galois::do_all_choice (
        // range,
        // [this] (GNode node) {
          // push (node);
        // },
        // "push_initial",
        // galois::chunk_size<CHUNK_SIZE> ());
//
    // WL_ty sources;
    // unsigned rounds = 0;
//
    // while (true) {
//
      // assert (sources.size () == 0);
//
      // t_dag_init.start ();
      // dagManager.reinitActiveDAG (galois::runtime::makeLocalRange (nextWork), sources);
      // nextWork.clear_all_parallel ();
      // t_dag_init.stop ();
//
//
      // if (sources.size () == 0) {
        // break;
      // }
//
      // ++rounds;
//
      // t_dag_exec.start ();
      // typedef galois::worklists::ExternalReference<WL_ty> WL;
      // typename WL::value_type* it = 0;
      // galois::for_each(it, it,
          // ApplyOperator {*this},
          // galois::loopname(loopname),
          // galois::wl<WL>(&sources));
      // t_dag_exec.stop ();
//
      // sources.reset_all ();
//
    // }
//
    // std::printf ("InputGraphDAGexecutor: performed %d rounds\n", rounds);
    // std::printf ("InputGraphDAGexecutor: time taken by dag initialization: %lu\n", t_dag_init.get ());
    // std::printf ("InputGraphDAGexecutor: time taken by dag execution: %lu\n", t_dag_exec.get ());
  }


  /*
  struct ApplyOperatorAsync {

    typedef int tt_does_not_need_aborts;

    InputGraphDAGexecutor& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {

      auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);

      if (sd.onWL > 0) {
        outer.func (src, dummyCtx);
      }

      G& graph = outer.graph;

      auto closure = [&graph, &ctx] (GNode dst) {

        auto& dd = graph.getData (dst, galois::MethodFlag::UNPROTECTED);

        int x = --dd.indegree;
        assert (x >= 0);
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

  galois::substrate::getThreadPool ().burnPower (galois::getActiveThreads ());

  InputGraphDAGexecutor<G,F, M> executor {graph, func, dagManager, loopname};

  executor.execute (range);

  galois::substrate::getThreadPool ().beKind ();

}

template <typename R, typename F, typename G>
void for_each_det_edge_flip_ar (const R& range, const F& func, G& graph, const char* loopname) {

  typedef typename DAGmanagerInOut<G>::Manager M;
  M dagManager {graph};

  for_each_det_edge_flip_ar (range, func, graph, dagManager, loopname);
}

  // three strategies for termination
  // 1. func keeps returns true when computation converges. Terminate when all
  // nodes return true.
  // 2. ctx.push just counts the number of pushes. Terminate when 0 pushes
  // performed
  // 3. ctx.push marks the node active. Apply the func to active nodes only.
  // Terminate when no active nodes. "Activeness" can also be implemented as a
  // counter, which is incremented every time a node is marked active and
  // decremented upon processing
  //
  // Other features:
  // 1. reinit the DAG on each round by a given priority function.


template <typename G, typename F, typename M>
struct InputGraphDAGtopologyDriven {


  typedef typename G::GraphNode GNode;

  typedef galois::PerThreadBag<GNode> Bag_ty;

  static const unsigned CHUNK_SIZE = F::CHUNK_SIZE;
  typedef galois::worklists::AltChunkedFIFO<CHUNK_SIZE, GNode> Inner_WL_ty;
  typedef galois::worklists::WLsizeWrapper<Inner_WL_ty> WL_ty;
  typedef substrate::PerThreadStorage<UserContextAccess<GNode> > PerThreadUserCtx;


  G& graph;
  F func;
  M& dagManager;
  const char* loopname;

  GAccumulator<size_t> numActiveFound;
  GAccumulator<size_t> numPushes;

public:

  InputGraphDAGtopologyDriven (G& graph, const F& func, M& dagManager, const char* loopname)
    :
      graph (graph),
      func (func),
      dagManager (dagManager),
      loopname (loopname)
  {}

  void push (GNode node) {
    numPushes += 1;
    auto& nd = graph.getData (node, galois::MethodFlag::UNPROTECTED);
    // ++(nd.onWL);
    nd.onWL.cas (0, 1);
  }

  template <typename R>
  void execute (const R& range) {
    Bag_ty sources;



    galois::TimeAccumulator t_dag_init;
    t_dag_init.start ();
    dagManager.initDAG ();
    dagManager.collectSources (sources);
    t_dag_init.stop ();

    galois::do_all_choice (
        range,
        [this] (GNode node) {
          push (node);
        },
        "push_initial",
        galois::chunk_size<CHUNK_SIZE> ());

    galois::TimeAccumulator t_dag_exec;

    unsigned rounds = 0;
    while (true) {

      ++rounds;

      assert (sources.size_all () != 0);

      auto f = [this] (GNode src) {
        auto& sd = graph.getData (src, galois::MethodFlag::UNPROTECTED);
        if (sd.onWL > 0) { // is active
          sd.onWL = 0;
          // --(sd.onWL);

          this->func (src, *this);
          this->numActiveFound += 1;
        }
      };

      t_dag_exec.start ();
      dagManager.runDAGcomputation (f, sources, loopname, galois::chunk_size<CHUNK_SIZE> ());
      t_dag_exec.stop ();

      bool term = (numPushes.reduce() == 0);


      if (term) { break; }

      t_dag_init.start ();

      // reset
      dagManager.resetDAG ();
      numActiveFound.reset ();
      numPushes.reset ();

      t_dag_init.stop ();

    }

    std::printf ("InputGraphDAGtopologyDriven: performed %d rounds\n", rounds);
    std::printf ("InputGraphDAGtopologyDriven: time taken by dag initialization: %lu\n", t_dag_init.get ());
    std::printf ("InputGraphDAGtopologyDriven: time taken by dag execution: %lu\n", t_dag_exec.get ());
  }


};



template <typename R, typename F, typename G, typename M>
void for_each_det_edge_flip_topo (const R& range, const F& func, G& graph, M& dagManager, const char* loopname) {

  galois::substrate::getThreadPool ().burnPower (galois::getActiveThreads ());

  InputGraphDAGtopologyDriven<G,F, M> executor {graph, func, dagManager, loopname};

  executor.execute (range);

  galois::substrate::getThreadPool ().beKind ();

}

template <typename R, typename F, typename G>
void for_each_det_edge_flip_topo (const R& range, const F& func, G& graph, const char* loopname) {

  typedef typename DAGmanagerInOut<G>::Manager M;
  M dagManager {graph};

  for_each_det_edge_flip_topo (range, func, graph, dagManager, loopname);
}


template <typename G, typename F, typename M>
struct HybridInputDAGexecutor {

  typedef typename G::GraphNode GNode;
  static const unsigned CHUNK_SIZE = F::CHUNK_SIZE;

  // typedef galois::worklists::AltChunkedFIFO<CHUNK_SIZE, GNode> Inner_WL_ty;
  // typedef galois::worklists::WLsizeWrapper<Inner_WL_ty> WL_ty;

  typedef galois::PerThreadBag<GNode> Bag_ty;

  using ParCounter = galois::GAccumulator<size_t>;

  G& graph;
  F func;
  M& dagManager;
  const char* loopname;

  unsigned cutOffColor = 10000; // arbitrary value

  std::vector<Bag_ty>* currColorBags;
  std::vector<Bag_ty>* nextColorBags;

  Bag_ty* currHeavyWork;
  Bag_ty* nextHeavyWork;
  ParCounter numPushes;


public:

  HybridInputDAGexecutor (G& graph, const F& func, M& dagManager, const char* loopname)
    :
      graph (graph),
      func (func),
      dagManager (dagManager),
      loopname (loopname)
  {}


  void colorStats (void) {

    unsigned numColors = dagManager.getNumColors ();
    assert (numColors > 0);

    std::vector<Bag_ty> colorBags (numColors);

    std::vector<ParCounter> sumDegree (numColors);
    std::vector<ParCounter> sumSucc (numColors);

    std::vector<galois::GReduceMin<size_t> > minDegree (numColors);
    std::vector<galois::GReduceMax<size_t> > maxDegree (numColors);

    std::vector<galois::GReduceMin<size_t> > minSucc (numColors);
    std::vector<galois::GReduceMax<size_t> > maxSucc (numColors);

    galois::do_all_choice (
        makeLocalRange (graph),
        [&] (GNode src) {
          auto& sd = graph.getData (src, MethodFlag::UNPROTECTED);
          assert (sd.color < numColors);
          colorBags[sd.color].push (src);

          size_t deg = dagManager.countAdj (src);
          sumDegree[sd.color] += deg;

          minDegree[sd.color].update (deg);
          maxDegree[sd.color].update (deg);

          size_t succ = dagManager.countDAGsucc (src);
          sumSucc[sd.color] += succ;

          minSucc[sd.color].update (succ);
          maxSucc[sd.color].update (succ);

        },
        "color_bags",
        galois::chunk_size<CHUNK_SIZE> ());


    size_t sumAllDegree = 0;
    size_t sumAllDAGsucc = 0;

    for (unsigned i = 0; i < numColors; ++i) {

      size_t sz = colorBags[i].size_all ();

      // std::printf ("Bucket %d, size=%zd, maxDegree = %zd, sumDegree = %zd, avgDegree=%zd, maxSucc = %zd, sumSucc = %zd, avgSucc=%zd\n",
          // i,
          // sz,
          // maxDegree[i].reduceRO (),
          // sumDegree[i].reduceRO (),
          // sumDegree[i].reduceRO ()/sz,
          // maxSucc[i].reduceRO (),
          // sumSucc[i].reduceRO (),
          // sumSucc[i].reduceRO ()/sz);
//
      sumAllDegree += sumDegree[i].reduce();
      sumAllDAGsucc += sumSucc[i].reduce();
    }

    // for (unsigned i = 0; i < numColors; ++i) {
//
      // std::printf ("Bucket %d, size=%zd, minSucc = %zd, maxSucc = %zd, sumSucc = %zd\n",
          // i, colorBags[i].size_all (), minSucc[i].reduceRO (), maxSucc[i].reduceRO (), sumSucc[i].reduceRO ());
//
    // }


    printf ("BEG_TABLE\n");
    printf ("BUCKET, SIZE, SUM_SIZE_PCT, SUM_DEG_PCT, SUM_SUCC_PCT, SUM_PRED_PCT\n");

    const size_t sumAllDAGpred = sumAllDegree - sumAllDAGsucc;
    assert (sumAllDAGpred > 0);

    size_t integralSize = 0;
    size_t integralDeg = 0;
    size_t integralSucc = 0;
    size_t integralPred = 0;
    const size_t numNodes = graph.size ();

    // for (unsigned i = numColors; i > 0;) {
      // --i;

    for (unsigned i = 0; i < numColors; ++i) {
      integralSize += colorBags[i].size_all ();
      integralDeg += sumDegree[i].reduce();
      integralSucc += sumSucc[i].reduce();
      integralPred += (sumDegree[i].reduce() - sumSucc[i].reduce());

      printf ("%d, %zd, %3lf, %3lf, %3lf, %3lf\n",
          i,
          colorBags[i].size_all (),
          double (integralSize) / double (numNodes),
          double (integralDeg) / double (sumAllDegree),
          double (integralSucc) / double (sumAllDAGsucc),
          double (integralPred) / double (sumAllDAGpred));



    }
    printf ("END_TABLE\n");
  }

  void defineCutOffColor (void) {
    static constexpr double WORK_CUTOFF_LIM = 0.98;

    unsigned numColors = dagManager.getNumColors ();

    std::vector<ParCounter> bagSizes (numColors);

    galois::do_all_choice (
        makeLocalRange (graph),
        [&] (GNode src) {
          auto& sd = graph.getData (src, MethodFlag::UNPROTECTED);
          assert (sd.color < numColors);
          bagSizes[sd.color] += 1;
        },
        "bag-sizes",
        galois::chunk_size<CHUNK_SIZE> ());

    const size_t numNodes = graph.size ();

    size_t runningSum = 0;

    for (unsigned i = 0; i < numColors; ++i) {

      cutOffColor = i + 1;
      runningSum += bagSizes[i].reduce();

      if (double (runningSum) / double (numNodes) > WORK_CUTOFF_LIM) {
        break;
      }
    }

    assert (cutOffColor <= numColors);

    std::printf ("Total colors = %d, cutOffColor = %d\n", numColors, cutOffColor);



  }

  void push (GNode node) {
    auto& nd = graph.getData (node, galois::MethodFlag::UNPROTECTED);

    int expected = 0;
    if (nd.onWL.cas (expected, 1)) {

      numPushes += 1;

      if (nd.color < cutOffColor) {
        assert (nextColorBags != nullptr);
        (*nextColorBags)[nd.color].push (node);

      } else {
        nextHeavyWork->push (node);
      }
    }
  }

  void printRoundStats (unsigned round) {
    printf ("============ Round %d ==========\n", round);
    for (unsigned i = 0; i < currColorBags->size (); ++i) {
      printf ("Bucket %d has size %zd\n", i, (*currColorBags)[i].size_all ());
    }
    printf ("Heavy Bucket has size %zd\n", currHeavyWork->size_all ());
  }

  template <typename CS>
  void runBag (Bag_ty& bag, bool runEdgeFlip, const char* loopname, const CS& cs) {

    if (runEdgeFlip) {

      dagManager.runActiveDAGcomp (
          makeLocalRange (bag),
          func, *this, loopname,
          cs);

    } else {

      galois::do_all_choice(
          makeLocalRange (bag),
          [this] (GNode src) {
            auto& sd = graph.getData (src, MethodFlag::UNPROTECTED);
            assert (sd.onWL > 0);
            sd.onWL = 0;
            // printf ("Chromatic executing %d\n", sd.id);
            func (src, *this);
          },
          loopname,
          cs);

    }

  }


  template <typename R>
  void execute (const R& range) {

    galois::TimeAccumulator t_chromatic;
    galois::TimeAccumulator t_edge_flip;

    dagManager.initDAG ();
    dagManager.colorDAG ();

    // defineCutOffColor ();
    cutOffColor = cutOffColorOpt;
    cutOffColor = std::min (unsigned (cutOffColor), dagManager.getNumColors ());

    if (reinitAfterCutOff) {

      galois::StatTimer t_reinit ("reinit after cutoff");

      t_reinit.start ();
      using ND = typename G::node_data_type;
      auto filterAfter = [this] (const ND& dd) {
        return dd.color >= cutOffColor;
      };

      Bag_ty nodesAfterCutoff;
      // Bag_ty nodesBeforeCutoff;

      galois::do_all_choice (
          makeLocalRange (graph),
          // [this, &filterAfter, &nodesAfterCutoff, &nodesBeforeCutoff] (GNode src) {
          [this, &filterAfter, &nodesAfterCutoff] (GNode src) {
            auto& sd = graph.getData (src, MethodFlag::UNPROTECTED);
            if (filterAfter (sd)) {
              nodesAfterCutoff.push (src);
            } else {
              // nodesBeforeCutoff.push (src);
            }
          },
          "filter-after-cutoff",
          galois::chunk_size<CHUNK_SIZE> ());

      dagManager.filterDAGsucc (makeLocalRange (nodesAfterCutoff), filterAfter);

      // dagManager.filterDAGsucc (
          // makeLocalRange (nodesBeforeCutoff),
          // [&filterAfter] (const ND& dd) { return !filterAfter (dd); }
          // );
      // nodesBeforeCutoff.clear_all_parallel ();

      nodesAfterCutoff.clear_all_parallel ();

      t_reinit.stop ();
    }

    // colorStats (); return;

    currColorBags = new std::vector<Bag_ty> (cutOffColor);
    nextColorBags = new std::vector<Bag_ty> (cutOffColor);

    currHeavyWork = new Bag_ty ();
    nextHeavyWork = new Bag_ty ();

    galois::do_all_choice (
        range,
        [this] (GNode node) {
          push (node);
        },
        "push_initial",
        galois::chunk_size<CHUNK_SIZE> ());

    unsigned rounds = 0;

    galois::StatTimer t_heavy("operator-heavy-serial");
    while (true) {

      ++rounds;
      std::swap (currColorBags, nextColorBags);
      std::swap (currHeavyWork, nextHeavyWork);
      numPushes.reset ();

      for (unsigned i = 0; i < nextColorBags->size (); ++i) {
        (*nextColorBags)[i].clear_all_parallel ();
      }

      nextHeavyWork->clear_all_parallel ();


      if (false) {
        printRoundStats (rounds);
      }

      static const unsigned HEAVY_CHUNK_SIZE = 1;

      Bag_ty* eflipBag = nullptr;
      bool runEdgeFlip = false;

      for (unsigned i = 0; i < currColorBags->size (); ++i) {
        if (!(*currColorBags)[i].empty_all ()) {

          size_t sz = (*currColorBags)[i].size_all ();

          if (sz < (threadMultFactor * galois::getActiveThreads ())) {

            if (eflipBag == nullptr) {
              eflipBag = &((*currColorBags)[i]);
            } else {
              eflipBag->splice_all ((*currColorBags)[i]);
              runEdgeFlip = true;
            }

            continue;

          } else {

            // if (eflipBag != nullptr) {
              // // eflipBag has some stuff, first run that
              // runBag (*eflipBag, runEdgeFlip,
                  // (runEdgeFlip ? "operator-edge-flip" : "operator-chromatic"),
                  // galois::chunk_size<HEAVY_CHUNK_SIZE> ());
//
              // eflipBag = nullptr;
              // runEdgeFlip = false;
            // }

            // now run the current bucket using chromatic
            runBag ((*currColorBags) [i], false, "operator-chromatic",
                galois::chunk_size<HEAVY_CHUNK_SIZE> ());


          } // end if sz
        } // end if not empty
      } // end for


      if (eflipBag != nullptr) {
        // std::printf ("running edge-flip before cutoff, size=%zd, distance=%ld\n", eflipBag->size_all (),
            // std::distance (eflipBag->begin (), eflipBag->end ()));

        // eflipBag has some stuff, first run that
        runBag (*eflipBag, runEdgeFlip,
            (runEdgeFlip ? "operator-edge-flip" : "operator-chromatic"),
            galois::chunk_size<HEAVY_CHUNK_SIZE> ());

        eflipBag = nullptr;
        runEdgeFlip = false;
      }



      // std::printf ("running edge-flip after cutoff, for heavy bucket, size=%zd\n", currHeavyWork->size_all ());
      // run edge flip
      runBag (*currHeavyWork, true, "operator-edge-flip-remaining",
          galois::chunk_size<HEAVY_CHUNK_SIZE> ());


      if (numPushes.reduce() == 0) {
        break;
      }
      t_heavy.stop ();



    } //end while

    std::printf ("HybridInputDAGexecutor performed %d rounds\n", rounds);

    delete currColorBags; currColorBags = nullptr;
    delete nextColorBags; nextColorBags = nullptr;
    delete currHeavyWork; currHeavyWork = nullptr;
    delete nextHeavyWork; nextHeavyWork = nullptr;


  }

};

template <typename R, typename F, typename G, typename M>
void for_each_det_input_hybrid (const R& range, const F& func, G& graph, M& dagManager, const char* loopname) {

  galois::substrate::getThreadPool ().burnPower (galois::getActiveThreads ());

  HybridInputDAGexecutor<G,F, M> executor {graph, func, dagManager, loopname};

  executor.execute (range);

  galois::substrate::getThreadPool ().beKind ();

}

template <typename R, typename F, typename G>
void for_each_det_input_hybrid (const R& range, const F& func, G& graph, const char* loopname) {

  typedef typename DAGmanagerInOut<G>::Manager M;
  M dagManager {graph};

  for_each_det_input_hybrid (range, func, graph, dagManager, loopname);
}




template <InputDAG_ExecTy EXEC>
struct ForEachDet_InputDAG {
};


template <>
struct ForEachDet_InputDAG<InputDAG_ExecTy::CHROMATIC> {

  template <typename R, typename F, typename G>
  static void run (const R& range, const F& func, G& graph, const char* loopname) {
    for_each_det_chromatic (range, func, graph, loopname);
  }
};

template <>
struct ForEachDet_InputDAG<InputDAG_ExecTy::EDGE_FLIP> {

  template <typename R, typename F, typename G>
  static void run (const R& range, const F& func, G& graph, const char* loopname) {
    for_each_det_edge_flip_ar (range, func, graph, loopname);
  }
};

template <>
struct ForEachDet_InputDAG<InputDAG_ExecTy::TOPO> {

  template <typename R, typename F, typename G>
  static void run (const R& range, const F& func, G& graph, const char* loopname) {
    for_each_det_edge_flip_topo (range, func, graph, loopname);
  }
};

template <>
struct ForEachDet_InputDAG<InputDAG_ExecTy::HYBRID> {

  template <typename R, typename F, typename G>
  static void run (const R& range, const F& func, G& graph, const char* loopname) {
    for_each_det_input_hybrid (range, func, graph, loopname);
  }
};

} // end namespace runtime
} // end namespace galois

#endif // GALOIS_RUNTIME_DET_CHROMATIC_H
