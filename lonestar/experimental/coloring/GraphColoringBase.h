#ifndef GRAPH_COLORING_BASE_H
#define GRAPH_COLORING_BASE_H

#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Timer.h"
#include "Galois/Galois.h"
#include "Galois/DoAllWrap.h"
#include "Galois/Graphs/Util.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/PerThreadContainer.h"
// #include "Galois/Graph/FileGraph.h"

#include "Galois/Runtime/Sampling.h"

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

static cll::opt<std::string> filename (cll::Positional, cll::desc ("<input file>"), cll::Required);

static cll::opt<HeuristicType> heuristic (
    cll::desc ("choose heuristic"),
    cll::values (
      clEnumVal (FIRST_FIT, "first fit, no priority"),
      clEnumVal (BY_ID, "order by ID modulo some constant"),
      clEnumVal (RANDOM, "uniform random within some small range"),
      clEnumVal (MIN_DEGREE, "order by min degree first"),
      clEnumVal (MAX_DEGREE, "order by max degree first"),
      clEnumValEnd),
    cll::init (FIRST_FIT));


static cll::opt<bool> useParaMeter ("parameter", cll::desc ("use parameter executor"), cll::init (false));

static const char* const name = "Graph Coloring";
static const char* const desc = "Greedy coloring of graphs with minimal number of colors";
static const char* const url = "graph-coloring";

template <typename G>
class GraphColoringBase: private boost::noncopyable {

protected:

  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  typedef Galois::PerThreadVector<unsigned> PerThrdColorVec;
  typedef typename G::GraphNode GN;
  typedef typename G::node_data_type NodeData;

  G graph;
  PerThrdColorVec perThrdColorVec;

  void readGraph (void) {
    Galois::Graph::readGraph (graph, filename);

    const size_t numNodes = graph.size ();
    Galois::GAccumulator<size_t> numEdges;

    Galois::StatTimer t_init ("initialization time: ");
    
    t_init.start ();
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
            // graph.getData (*it_beg, Galois::MethodFlag::UNPROTECTED) = NodeData (beg++);
            auto* ndptr = &(graph.getData (*it_beg, Galois::MethodFlag::UNPROTECTED));
            ndptr->~NodeData();
            new (ndptr) NodeData (beg++);
            
            

            size_t deg = std::distance (
              graph.edge_begin (*it_beg, Galois::MethodFlag::UNPROTECTED),
              graph.edge_end (*it_beg, Galois::MethodFlag::UNPROTECTED));

            numEdges.update (deg);
          }


        },
        Galois::loopname ("initialize"));

    // color 0 is reserved as uncolored value
    // therefore, we put in at least 1 entry to handle the 
    // corner case when a node with no neighbors is being colored
    for (unsigned i = 0; i < perThrdColorVec.numRows (); ++i) {
      perThrdColorVec.get(i).resize (1, 0);
    }

    t_init.stop ();

    std::printf ("Graph read with %zd nodes and %zd edges\n", numNodes, numEdges.reduceRO ());

  }

  void colorNode (GN src) {

    auto& sd = graph.getData (src, Galois::MethodFlag::UNPROTECTED);

    auto& forbiddenColors = perThrdColorVec.get ();
    std::fill (forbiddenColors.begin (), forbiddenColors.end (), unsigned (-1));

    for (typename G::edge_iterator e = graph.edge_begin (src, Galois::MethodFlag::UNPROTECTED),
        e_end = graph.edge_end (src, Galois::MethodFlag::UNPROTECTED); e != e_end; ++e) {

      GN dst = graph.getEdgeDst (e);
      auto& dd = graph.getData (dst, Galois::MethodFlag::UNPROTECTED);

      if (forbiddenColors.size () <= dd.color) {
        forbiddenColors.resize (dd.color + 1, unsigned (-1));
      }
      // std::printf ("Neighbor %d has color %d\n", dd.id, dd.color);

      forbiddenColors[dd.color] = sd.id;
    }


    bool colored = false;
    for (size_t i = 1; i < forbiddenColors.size (); ++i) {
      if (forbiddenColors[i] != sd.id) { 
        sd.color = i;
        colored = true;
        break;
      }
    }

    if (!colored) {
      sd.color = forbiddenColors.size ();
    }

    // std::printf ("Node %d assigned color %d\n", sd.id, sd.color);

  }

  template <typename F>
  void assignPriorityHelper (const F& nodeFunc) {
    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [&] (GN node) {
          nodeFunc (node);
        },
        "assign-priority",
        Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());
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

    auto byId = [&] (GN node) {
      auto& nd = graph.getData (node, Galois::MethodFlag::UNPROTECTED);
      nd.priority = nd.id % MAX_LEVELS;
    };


    Galois::Substrate::PerThreadStorage<RNG>  perThrdRNG;

    auto randPri = [&] (GN node) {
      auto& rng = *(perThrdRNG.getLocal ());
      auto& nd = graph.getData (node, Galois::MethodFlag::UNPROTECTED);
      nd.priority = rng ();
    };


    auto minDegree = [&] (GN node) {
      auto& nd = graph.getData (node, Galois::MethodFlag::UNPROTECTED);
      nd.priority = std::distance (
                      graph.edge_begin (node, Galois::MethodFlag::UNPROTECTED),
                      graph.edge_end (node, Galois::MethodFlag::UNPROTECTED));
    };

    const size_t numNodes = graph.size ();
    auto maxDegree = [&] (GN node) {
      auto& nd = graph.getData (node, Galois::MethodFlag::UNPROTECTED);
      nd.priority = numNodes - std::distance (
                                  graph.edge_begin (node, Galois::MethodFlag::UNPROTECTED),
                                  graph.edge_end (node, Galois::MethodFlag::UNPROTECTED));
    };
    
    Galois::StatTimer t_priority ("priority assignment time: ");

    t_priority.start ();

    switch (heuristic) {
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

  void verify (void) {
    if (skipVerify) { return; }

    Galois::StatTimer t_verify ("verification time: ");

    t_verify.start ();

    Galois::GReduceLogicalOR foundError;
    Galois::GReduceMax<unsigned> maxColor;

    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [&] (GN src) {
          auto& sd = graph.getData (src, Galois::MethodFlag::UNPROTECTED);
          if (sd.color == 0) {
            std::fprintf (stderr, "ERROR: src %d found uncolored\n", sd.id);
            foundError.update (true);
          }
          for (typename G::edge_iterator e = graph.edge_begin (src, Galois::MethodFlag::UNPROTECTED),
              e_end = graph.edge_end (src, Galois::MethodFlag::UNPROTECTED); e != e_end; ++e) {

            GN dst = graph.getEdgeDst (e);
            auto& dd = graph.getData (dst, Galois::MethodFlag::UNPROTECTED);
            if (sd.color == dd.color) {
              foundError.update (true);
              std::fprintf (stderr, "ERROR: nodes %d and %d have the same color\n",
                sd.id, dd.id);

            }
          }

          maxColor.update (sd.color);

        }, 
        "check-coloring",
        Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    std::printf ("Graph colored with %d colors\n", maxColor.reduce ());

    t_verify.stop ();

    if (foundError.reduceRO ()) {
      GALOIS_DIE ("ERROR! verification failed!\n");
    } else {
      printf ("OK! verification succeeded!\n");
    }
  }

  virtual void colorGraph (void) = 0;

public:

  void run (int argc, char* argv[]) {
    LonestarStart (argc, argv, name, desc, url);
    Galois::StatManager sm;

    readGraph ();

    Galois::preAlloc (Galois::getActiveThreads () + 2*sizeof(NodeData)*graph.size ()/Galois::Runtime::pagePoolSize());
    Galois::reportPageAlloc("MeminfoPre");

    Galois::StatTimer t;

    t.start ();
    colorGraph ();
    t.stop ();

    Galois::reportPageAlloc("MeminfoPost");

    verify ();
  }
};
#endif // GRAPH_COLORING_BASE_H
