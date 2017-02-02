#ifndef PAGE_RANK_ALAMERE_H
#define PAGE_RANK_ALAMERE_H

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <set>

#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/DoAllWrap.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/FileGraph.h"
#include "llvm/Support/CommandLine.h"

#include "Galois/Runtime/Sampling.h"

#include "Lonestar/BoilerPlate.h"

#include "PageRankOld.h"

namespace pagerank {

typedef Galois::GAccumulator<unsigned> ParCounter;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile (cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> transposeFile ("graphTranspose", cll::desc ("<transpose file>"), cll::Required);


static const bool DOALL_STEAL = true;


static const char* const name = "Page Rank";
static const char* const desc = "Computes Page Rank over a directed graph";
static const char* const url = "pagerank";

static const double INIT_PR_VAL = 1.0;

static const double RANDOM_JUMP = 0.3;

static const double TERM_THRESH = 1.0e-6;

static const bool DEBUG = false;

static const unsigned DEFAULT_CHUNK_SIZE = 4;

class PageRankAlamere {


  struct PNode {
    double deg_inv;
    double value[2];

    explicit PNode (double init=INIT_PR_VAL, unsigned out_deg=0) {
      // Assumption: 0 is to be read first
      // can't init both to same because the computation 
      // may converge immediately due to all 1s
      value[0] = init;
      value[1] = 0.0;

      deg_inv = 1 / double (out_deg);
    }

    // even vs odd
    double getValue (unsigned iteration) const {
      return value[iteration % 2];
    }

    double getScaledValue (unsigned iteration) const {
      return getValue (iteration) * deg_inv;
    }

    void setValue (unsigned iteration, double val) {
      value[(iteration + 1) % 2] = val;
    }

  };

  typedef typename Galois::Graph::LC_CSR_Graph<PNode,void>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type
    InnerGraph;

  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;
  typedef std::vector<GNode> VecGNode;

  void initGraph (const std::string& inputFile, const std::string& transposeFile, Graph& graph) {

    Galois::Graph::readGraph (graph, inputFile, transposeFile);

    // size_t numEdges = 0; 
    // size_t selfEdges = 0;
    ParCounter numEdges;
    ParCounter selfEdges;

    Galois::StatTimer t_init ("Time for initializing PageRank data: ");

    t_init.start ();

    Galois::do_all_choice (Galois::Runtime::makeLocalRange (graph),
        [&] (GNode n) {
              unsigned out_deg = 0;
              for (auto j = graph.edge_begin (n, Galois::MethodFlag::UNPROTECTED)
                , endj = graph.edge_end (n, Galois::MethodFlag::UNPROTECTED); j != endj; ++j) {
                GNode neigh = graph.getEdgeDst (*j);
                if (n != neigh) {
                  out_deg += 1;
                } else{
                  selfEdges += 1;
                }
              }


              if (DEBUG) {
                int in_deg = std::distance (graph.in_edge_begin (n, Galois::MethodFlag::UNPROTECTED)
                , graph.in_edge_end (n, Galois::MethodFlag::UNPROTECTED));
                std::cout << "Node: " << graph.idFromNode (n) << " has out degree: " << out_deg 
                  << ", in degree: " << in_deg << std::endl;
              }

              graph.getData (n, Galois::MethodFlag::UNPROTECTED) = PNode (INIT_PR_VAL, out_deg);

              numEdges += out_deg;

            },
        std::make_tuple(
            Galois::loopname("init_loop"), 
            Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

    t_init.stop ();

    std::cout << "Graph read with: " << graph.size () << " nodes, and: " << numEdges.reduce ()
      << " non-self edges" << std::endl;
    std::cout << "Number of selfEdges: " << selfEdges.reduce () << std::endl;
    
  }

  struct PageRankOp {

    Graph& graph;
    const unsigned round;
    Galois::GReduceLogicalAND& allConverged; 
    
    PageRankOp (
        Graph& graph,
        const unsigned round,
        Galois::GReduceLogicalAND& allConverged)
      : 
        graph (graph),
        round (round),
        allConverged (allConverged)
    {}

    template <typename C>
    void operator () (GNode src, C&) {
      (*this) (src);
    }

    void operator () (GNode src) {


      double sumRanks = 0.0;

      if (DEBUG) {
        std::cout << "Processing Node: " << graph.idFromNode (src) << std::endl;
      }

      for (auto ni = graph.in_edge_begin (src, Galois::MethodFlag::UNPROTECTED)
          , endni = graph.in_edge_end (src, Galois::MethodFlag::UNPROTECTED); ni != endni; ++ni) {

        GNode pred = graph.getInEdgeDst (ni);

        if (pred != src) { // non-self edge
          const PNode& predData = graph.getData (pred, Galois::MethodFlag::UNPROTECTED);
          sumRanks += predData.getScaledValue (round);

          if (DEBUG) {
            std::cout << "Value from Neighbor: " << graph.idFromNode (pred) 
              << " is: " << predData.getScaledValue (round) << std::endl;
          }

        }
      }

      PNode& srcData = graph.getData (src, Galois::MethodFlag::UNPROTECTED);

      double updatedValue = RANDOM_JUMP + (1 - RANDOM_JUMP) * sumRanks;

      double currValue = srcData.getValue (round);

      srcData.setValue (round, updatedValue);

      if (std::fabs (currValue - updatedValue) > TERM_THRESH) {
        allConverged.update (false);
      }

      // if (round > 100) {
        // std::cout << "Difference: " << (currValue - updatedValue) << std::endl;
      // }
      
    }
  };

  unsigned runPageRank (Graph& graph) {


    unsigned round = 0;

    while (true) {

      Galois::GReduceLogicalAND allConverged;

      Galois::do_all_choice (Galois::Runtime::makeLocalRange (graph),
          PageRankOp (graph, round, allConverged), 
          std::make_tuple (
            Galois::loopname("page_rank_inner"), 
            Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));



      if (DEBUG) {
        std::cout << "Finished round: " << round << std::endl;

        for (auto i = graph.begin (), endi = graph.end ();
            i != endi; ++i) {

          PNode& data = graph.getData (*i, Galois::MethodFlag::UNPROTECTED);

          std::cout << "Node: " << graph.idFromNode (*i) << ", page ranke values: " << 
            data.getValue (round) << ", " << data.getValue (round + 1) << std::endl;
        }
      }


      if (allConverged.reduceRO ()) {
        break;
      }

      ++round;


    }

    std::cout << "number of round completed: " << round << std::endl;

    return round;

  }

  bool checkConvergence (Graph& graph, const unsigned round) {
    Galois::GReduceLogicalAND allConverged;

    Galois::do_all_choice (Galois::Runtime::makeLocalRange (graph),
        [&] (GNode src) {
          double sum = 0;

          for (auto jj = graph.in_edge_begin(src, Galois::MethodFlag::UNPROTECTED), ej = graph.in_edge_end(src, Galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
            GNode dst = graph.getInEdgeDst(jj);
            if (dst != src) {
              auto& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
              sum += ddata.getScaledValue (round);
            }
          }

          float value = RANDOM_JUMP + (1 - RANDOM_JUMP) * sum;
          auto& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
          float diff = std::fabs(value - sdata.getValue (round));

          
          
          if (diff > TERM_THRESH) {
            allConverged.update (false);
            // std::fprintf (stderr, "ERROR: convergence failed on node %d, error=%f, tolerance=%f\n", src, diff, TERM_THRESH);
          }
        }, 
        std::make_tuple(
            Galois::loopname("check-convergence"), 
            Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

    return allConverged.reduceRO ();
  }

  void verify (Graph& graph, const unsigned round) {
    if (skipVerify) {
      std::printf ("WARNING, Verification skipped\n");
      return;
    }

    if (!checkConvergence (graph, round)) {
      std::fprintf (stderr, "ERROR: Convergence check FAILED\n");
    } else {
      std::printf ("OK: Convergence check PASSED\n");
    }

  }

public:

  void run (int argc, char* argv[]) {
    LonestarStart (argc, argv, name, desc, url);
    Galois::StatManager sm;

    Graph graph;

    Galois::StatTimer t_input ("time to initialize input:");

    t_input.start ();
    initGraph (inputFile, transposeFile, graph);
    t_input.stop ();

    Galois::reportPageAlloc("MeminfoPre");
    Galois::StatTimer t_run;
    
    t_run.start ();
    const unsigned round = runPageRank (graph);
    t_run.stop ();

    Galois::reportPageAlloc("MeminfoPost");


    Galois::StatTimer t_verify ("Time to verify:");
    t_verify.start ();
    verify (graph, round);
    t_verify.stop ();


  }

};


} // end namespace pagerank

#endif // PAGE_RANK_ALAMERE_H
