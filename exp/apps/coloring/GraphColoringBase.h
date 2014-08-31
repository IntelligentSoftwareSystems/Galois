#ifndef GRAPH_COLORING_BASE_H
#define GRAPH_COLORING_BASE_H

#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/DoAllWrap.h"
#include "Galois/Graph/Util.h"
#include "Galois/Graph/Graph.h"
// #include "Galois/Graph/FileGraph.h"

#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/PerThreadContainer.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <cstdio>

enum HeuristicType {
  FIRST_FIT,
  MIN_DEGREE,
  MAX_DEGREE,
};

namespace cll = llvm::cl;

static cll::opt<std::string> filename (cll::Positional, cll::desc ("<input file>"), cll::Required);

static cll::opt<HeuristicType> heuristic (
    cll::desc ("choose heuristic"),
    cll::values (
      clEnumVal (FIRST_FIT, "first fit"),
      clEnumVal (MIN_DEGREE, "order by min degree first"),
      clEnumVal (MAX_DEGREE, "order by max degree first"),
      clEnumValEnd),
    cll::init (FIRST_FIT));


static const char* const name = "Graph Coloring";
static const char* const desc = "Greedy coloring of graphs with minimal number of colors";
static const char* const url = "graph-coloring";

template <typename G>
class GraphColoringBase: private boost::noncopyable {

protected:

  static const unsigned DEFAULT_CHUNK_SIZE = 16;

  typedef Galois::Runtime::PerThreadVector<unsigned> PerThrdColorVec;
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
            // graph.getData (*it_beg, Galois::NONE) = NodeData (beg++);
            auto* ndptr = &(graph.getData (*it_beg, Galois::NONE));
            ndptr->~NodeData();
            new (ndptr) NodeData (beg++);
            
            

            size_t deg = std::distance (
              graph.edge_begin (*it_beg, Galois::NONE),
              graph.edge_end (*it_beg, Galois::NONE));

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

    auto& sd = graph.getData (src, Galois::NONE);

    auto& forbiddenColors = perThrdColorVec.get ();
    std::fill (forbiddenColors.begin (), forbiddenColors.end (), unsigned (-1));

    for (typename G::edge_iterator e = graph.edge_begin (src, Galois::NONE),
        e_end = graph.edge_end (src, Galois::NONE); e != e_end; ++e) {

      GN dst = graph.getEdgeDst (e);
      auto& dd = graph.getData (dst, Galois::NONE);

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

  };


  void verify (void) {
    if (skipVerify) { return; }

    Galois::StatTimer t_verify ("verification time: ");

    t_verify.start ();

    Galois::GReduceLogicalOR foundError;
    Galois::GReduceMax<unsigned> maxColor;

    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [&] (GN src) {
          auto& sd = graph.getData (src, Galois::NONE);
          if (sd.color == 0) {
            std::fprintf (stderr, "ERROR: src %d found uncolored\n", sd.id);
            foundError.update (true);
          }
          for (typename G::edge_iterator e = graph.edge_begin (src, Galois::NONE),
              e_end = graph.edge_end (src, Galois::NONE); e != e_end; ++e) {

            GN dst = graph.getEdgeDst (e);
            auto& dd = graph.getData (dst, Galois::NONE);
            if (sd.color == dd.color) {
              foundError.update (true);
              std::fprintf (stderr, "ERROR: nodes %d and %d have the same color\n",
                sd.id, dd.id);

            }
          }

          maxColor.update (sd.color);

        }, 
        "check-coloring",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());

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

  virtual void run (int argc, char* argv[]) {
    LonestarStart (argc, argv, name, desc, url);
    Galois::StatManager sm;

    readGraph ();

    Galois::StatTimer t;

    t.start ();
    colorGraph ();
    t.stop ();

    verify ();
  }
};
#endif // GRAPH_COLORING_BASE_H
