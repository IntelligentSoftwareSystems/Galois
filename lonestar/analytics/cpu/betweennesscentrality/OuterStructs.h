#ifndef GALOIS_BC_OUTER
#define GALOIS_BC_OUTER

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "Lonestar/BoilerPlate.h"
#include <boost/iterator/filter_iterator.hpp>

#include <iomanip>
#include <fstream>

using OuterGraph = galois::graphs::LC_CSR_Graph<void, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type;
using OuterGNode = OuterGraph::GraphNode;

////////////////////////////////////////////////////////////////////////////////

class BCOuter {
  OuterGraph* G;
  int NumNodes;

  galois::substrate::PerThreadStorage<double*> CB; // betweeness measure
  galois::substrate::PerThreadStorage<double*> perThreadSigma;
  galois::substrate::PerThreadStorage<int*> perThreadD;
  galois::substrate::PerThreadStorage<double*> perThreadDelta;
  galois::substrate::PerThreadStorage<galois::gdeque<OuterGNode>*>
      perThreadSucc;

public:
  /**
   * Constructor initializes thread local storage.
   */
  BCOuter(OuterGraph& g) : G(&g), NumNodes(g.size()) { InitializeLocal(); }

  /**
   * Constructor destroys thread local storage.
   */
  ~BCOuter(void) { DeleteLocal(); }

  //! Function that does BC for a single souce; called by a thread
  void doBC(const OuterGNode curSource) {
    galois::gdeque<OuterGNode> SQ;

    double* sigma                    = *perThreadSigma.getLocal();
    int* d                           = *perThreadD.getLocal();
    double* delta                    = *perThreadDelta.getLocal();
    galois::gdeque<OuterGNode>* succ = *perThreadSucc.getLocal();

    sigma[curSource] = 1;
    d[curSource]     = 1;

    SQ.push_back(curSource);

    // Do bfs while computing number of shortest paths (saved into sigma)
    // and successors of nodes;
    // Note this bfs makes it so source has distance of 1 instead of 0
    for (auto qq = SQ.begin(), eq = SQ.end(); qq != eq; ++qq) {
      int src = *qq;

      for (auto edge : G->edges(src, galois::MethodFlag::UNPROTECTED)) {
        int dest = G->getEdgeDst(edge);

        if (!d[dest]) {
          SQ.push_back(dest);
          d[dest] = d[src] + 1;
        }

        if (d[dest] == d[src] + 1) {
          sigma[dest] = sigma[dest] + sigma[src];
          succ[src].push_back(dest);
        }
      }
    }

    // Back-propogate the dependency values (delta) along the BFS DAG
    // ignore the source (hence SQ.size > 1 and not SQ.empty)
    while (SQ.size() > 1) {
      int leaf = SQ.back();
      SQ.pop_back();

      double sigma_leaf = sigma[leaf]; // has finalized short path value
      double delta_leaf = delta[leaf];
      auto& succ_list   = succ[leaf];

      for (auto succ = succ_list.begin(), succ_end = succ_list.end();
           succ != succ_end; ++succ) {
        delta_leaf += (sigma_leaf / sigma[*succ]) * (1.0 + delta[*succ]);
      }
      delta[leaf] = delta_leaf;
    }

    // save result of this source's BC, reset all local values for next
    // source
    double* Vec = *CB.getLocal();
    for (int i = 0; i < NumNodes; ++i) {
      Vec[i] += delta[i];
      delta[i] = 0;
      sigma[i] = 0;
      d[i]     = 0;
      succ[i].clear();
    }
  }

  /**
   * Runs betweeness-centrality proper. Instead of a vector of sources,
   * it will operate on the first numSources sources.
   *
   * @param numSources Num sources to get BC contribution for
   */
  void runAll(unsigned numSources) {
    // Each thread works on an individual source node
    galois::do_all(
        galois::iterate(0u, numSources),
        [&](const OuterGNode& curSource) { doBC(curSource); }, galois::steal(),
        galois::loopname("Main"));
  }

  /**
   * Runs betweeness-centrality proper.
   *
   * @tparam Cont type of the data structure that holds the nodes to treat
   * as a source during betweeness-centrality.
   *
   * @param v Data structure that holds nodes to treat as a source during
   * betweeness-centrality
   */
  template <typename Cont>
  void run(const Cont& v) {
    // Each thread works on an individual source node
    galois::do_all(
        galois::iterate(v),
        [&](const OuterGNode& curSource) { doBC(curSource); }, galois::steal(),
        galois::loopname("Main"));
  }

  /**
   * Verification for reference torus graph inputs.
   * All nodes should have the same betweenness value up to
   * some tolerance.
   */
  void verify() {
    double sampleBC = 0.0;
    bool firstTime  = true;
    for (int i = 0; i < NumNodes; ++i) {
      double bc = (*CB.getRemote(0))[i];

      for (unsigned j = 1; j < galois::getActiveThreads(); ++j)
        bc += (*CB.getRemote(j))[i];

      if (firstTime) {
        sampleBC = bc;
        galois::gInfo("BC: ", sampleBC);
        firstTime = false;
      } else {
        // check if over some tolerance value
        if ((bc - sampleBC) > 0.0001) {
          galois::gInfo("If torus graph, verification failed ",
                        (bc - sampleBC));
          return;
        }
      }
    }
  }

  /**
   * Print betweeness-centrality measures.
   *
   * @param begin first node to print BC measure of
   * @param end iterator after last node to print
   * @param out stream to output to
   * @param precision precision of the floating points outputted by the function
   */
  void printBCValues(size_t begin, size_t end, std::ostream& out,
                     int precision = 6) {
    for (; begin != end; ++begin) {
      double bc = (*CB.getRemote(0))[begin];

      for (unsigned j = 1; j < galois::getActiveThreads(); ++j)
        bc += (*CB.getRemote(j))[begin];

      out << begin << " " << std::setiosflags(std::ios::fixed)
          << std::setprecision(precision) << bc << "\n";
    }
  }

  /**
   * Print all betweeness centrality values in the graph.
   */
  void printBCcertificate() {
    std::stringstream foutname;
    foutname << "outer_certificate_" << galois::getActiveThreads();

    std::ofstream outf(foutname.str().c_str());
    galois::gInfo("Writing certificate...");

    printBCValues(0, NumNodes, outf, 9);

    outf.close();
  }

  //! sanity check of BC values
  void outerSanity(OuterGraph& graph) {
    galois::GReduceMax<float> accumMax;
    galois::GReduceMin<float> accumMin;
    galois::GAccumulator<float> accumSum;
    accumMax.reset();
    accumMin.reset();
    accumSum.reset();

    // get max, min, sum of BC values using accumulators and reducers
    galois::do_all(
        galois::iterate(graph),
        [&](LevelGNode n) {
          double bc = (*CB.getRemote(0))[n];

          for (unsigned j = 1; j < galois::getActiveThreads(); ++j)
            bc += (*CB.getRemote(j))[n];

          accumMax.update(bc);
          accumMin.update(bc);
          accumSum += bc;
        },
        galois::no_stats(), galois::loopname("OuterSanity"));

    galois::gPrint("Max BC is ", accumMax.reduce(), "\n");
    galois::gPrint("Min BC is ", accumMin.reduce(), "\n");
    galois::gPrint("BC sum is ", accumSum.reduce(), "\n");
  }

private:
  /**
   * Initialize an array at some provided address.
   *
   * @param addr Address to initialize array at
   */
  template <typename T>
  void initArray(T** addr) {
    *addr = new T[NumNodes]();
  }

  /**
   * Destroy an array at some provided address.
   *
   * @param addr Address to destroy array at
   */
  template <typename T>
  void deleteArray(T** addr) {
    delete[] * addr;
  }

  /**
   * Initialize local thread storage.
   */
  void InitializeLocal(void) {
    galois::on_each([this](unsigned, unsigned) {
      this->initArray(CB.getLocal());
      this->initArray(perThreadSigma.getLocal());
      this->initArray(perThreadD.getLocal());
      this->initArray(perThreadDelta.getLocal());
      this->initArray(perThreadSucc.getLocal());
    });
  }

  /**
   * Destroy local thread storage.
   */
  void DeleteLocal(void) {
    galois::on_each([this](unsigned, unsigned) {
      this->deleteArray(CB.getLocal());
      this->deleteArray(perThreadSigma.getLocal());
      this->deleteArray(perThreadD.getLocal());
      this->deleteArray(perThreadDelta.getLocal());
      this->deleteArray(perThreadSucc.getLocal());
    });
  }
};

/**
 * Functor that indicates if a node contains outgoing edges
 */
struct HasOut {
  OuterGraph* graph;
  HasOut(OuterGraph* g) : graph(g) {}

  bool operator()(const OuterGNode& n) const {
    return graph->edge_begin(n) != graph->edge_end(n);
  }
};

////////////////////////////////////////////////////////////////////////////////

void doOuterBC() {
  OuterGraph g;
  galois::graphs::readGraph(g, inputFile);

  BCOuter bcOuter(g);

  size_t NumNodes = g.size();

  // preallocate pages for use in algorithm
  galois::reportPageAlloc("MeminfoPre");
  galois::preAlloc(galois::getActiveThreads() * NumNodes / 1650);
  galois::reportPageAlloc("MeminfoMid");

  // vector of sources to process; initialized if doing outSources
  std::vector<OuterGNode> v;
  // preprocessing: find the nodes with out edges we will process and skip
  // over nodes with no out edges; only done if numOfSources isn't specified
  if (numOfSources == 0) {
    // find first node with out edges
    boost::filter_iterator<HasOut, OuterGraph::iterator> begin =
        boost::make_filter_iterator(HasOut(&g), g.begin(), g.end());
    boost::filter_iterator<HasOut, OuterGraph::iterator> end =
        boost::make_filter_iterator(HasOut(&g), g.end(), g.end());
    // adjustedEnd = last node we will process based on how many iterations
    // (i.e. sources) we want to do
    boost::filter_iterator<HasOut, OuterGraph::iterator> adjustedEnd =
        iterLimit ? galois::safe_advance(begin, end, (int)iterLimit) : end;

    size_t iterations = std::distance(begin, adjustedEnd);
    galois::gPrint("Num Nodes: ", NumNodes, " Start Node: ", startSource,
                   " Iterations: ", iterations, "\n");
    // vector of nodes we want to process
    v.insert(v.end(), begin, adjustedEnd);
  }

  // execute algorithm
  galois::StatTimer execTime("Timer_0");
  execTime.start();
  // either run a contiguous chunk of sources from beginning or run using
  // sources with outgoing edges only
  if (numOfSources > 0) {
    bcOuter.runAll(numOfSources);
  } else {
    bcOuter.run(v);
  }
  execTime.stop();

  bcOuter.printBCValues(0, std::min(10UL, NumNodes), std::cout, 6);
  bcOuter.outerSanity(g);
  if (output)
    bcOuter.printBCcertificate();

  if (!skipVerify)
    bcOuter.verify();

  galois::reportPageAlloc("MeminfoPost");
}
#endif
