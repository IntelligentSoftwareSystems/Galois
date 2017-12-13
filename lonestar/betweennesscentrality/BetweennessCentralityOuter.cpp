/** Betweenness centrality application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 * Betweeness-centrality. "Outer" because parallelism is in the outer loop,
 * i.e. each thread works on a single source.
 *
 * @author Dimitrios Prountzos <dprountz@cs.utexas.edu>
 */

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/filter_iterator.hpp>

#include <iomanip>
#include <fstream>

static const char* name = "Betweenness Centrality";
static const char* desc = "Computes the betweenness centrality of all nodes in "
                          "a graph";
static const char* url  = "betweenness_centrality";

static llvm::cl::opt<std::string> filename(llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::Required);
static llvm::cl::opt<int> iterLimit("limit",
                                    llvm::cl::desc("Limit number of iterations "
                                                   "to value (0 is all nodes)"),
                                    llvm::cl::init(0));
static llvm::cl::opt<unsigned int> startNode("startNode",
                                             llvm::cl::desc("Node to start "
                                                            "search from"),
                                             llvm::cl::init(0));
static llvm::cl::opt<bool> forceVerify("forceVerify",
                                       llvm::cl::desc("Abort if not verified; "
                                                      "only makes sense for "
                                                      "torus graphs"));
static llvm::cl::opt<bool> printAll("printAll",
                                    llvm::cl::desc("Print betweenness values "
                                                   "for all nodes"));

using Graph = galois::graphs::LC_CSR_Graph<void, void>
                ::with_no_lockable<true>::type
                ::with_numa_alloc<true>::type;
using GNode = Graph::GraphNode;

class BCOuter {
  Graph* G;
  int NumNodes;

  galois::substrate::PerThreadStorage<double*> CB; // betweeness measure
  galois::substrate::PerThreadStorage<double*> perThreadSigma;
  galois::substrate::PerThreadStorage<int*> perThreadD;
  galois::substrate::PerThreadStorage<double*> perThreadDelta;
  galois::substrate::PerThreadStorage<galois::gdeque<GNode>*> perThreadSucc;

public:
  /**
   * Constructor initializes thread local storage.
   */
  BCOuter (Graph& g) : G(&g), NumNodes(g.size()) {
    InitializeLocal();
  }

  /**
   * Constructor destroys thread local storage.
   */
  ~BCOuter (void) {
    DeleteLocal();
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
      [&] (const GNode& curSource) {
        galois::gdeque<GNode> SQ;

        double* sigma = *perThreadSigma.getLocal();
        int* d = *perThreadD.getLocal();
        double* delta = *perThreadDelta.getLocal();
        galois::gdeque<GNode>* succ = *perThreadSucc.getLocal();

        sigma[curSource] = 1;
        d[curSource] = 1;

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
          auto& succ_list = succ[leaf];

          for (auto succ = succ_list.begin(), succ_end = succ_list.end();
               succ != succ_end;
               ++succ) {
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
          d[i] = 0;
          succ[i].clear();
        }
      },
      galois::steal(),
      galois::loopname("Main")
    );
  }

  /**
   * Verification for reference torus graph inputs.
   * All nodes should have the same betweenness value up to
   * some tolerance.
   */
  void verify() {
    double sampleBC = 0.0;
    bool firstTime = true;
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
          if (forceVerify) abort();
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

private:
  /**
   * Initialize an array at some provided address.
   *
   * @param addr Address to initialize array at
   */
  template<typename T>
  void initArray(T** addr) {
    *addr = new T[NumNodes]();
  }

  /**
   * Destroy an array at some provided address.
   *
   * @param addr Address to destroy array at
   */
  template<typename T>
  void deleteArray(T** addr) {
    delete [] *addr;
  }

  /**
   * Initialize local thread storage.
   */
  void InitializeLocal(void) {
    galois::on_each(
      [this] (unsigned, unsigned) {
        this->initArray(CB.getLocal());
        this->initArray(perThreadSigma.getLocal());
        this->initArray(perThreadD.getLocal());
        this->initArray(perThreadDelta.getLocal());
        this->initArray(perThreadSucc.getLocal());
      }
    );
  }

  /**
   * Destroy local thread storage.
   */
  void DeleteLocal(void) {
    galois::on_each(
      [this] (unsigned, unsigned) {
        this->deleteArray(CB.getLocal());
        this->deleteArray(perThreadSigma.getLocal());
        this->deleteArray(perThreadD.getLocal());
        this->deleteArray(perThreadDelta.getLocal());
        this->deleteArray(perThreadSucc.getLocal());
      }
    );
  }
};

/**
 * Functor that indicates if a node contains outgoing edges
 */
struct HasOut: public std::unary_function<GNode, bool> {
  Graph* graph;
  HasOut(Graph* g) : graph(g) { }

  bool operator()(const GNode& n) const {
    return graph->edge_begin(n) != graph->edge_end(n);
  }
};

int main(int argc, char** argv) {
  galois::SharedMemSys Gal;
  LonestarStart(argc, argv, name, desc, url);

  Graph g;
  galois::graphs::readGraph(g, filename);

  BCOuter bcOuter(g);

  size_t NumNodes = g.size();

  // preallocate pages for use in algorithm
  galois::reportPageAlloc("MeminfoPre");
  galois::preAlloc(galois::getActiveThreads() * NumNodes / 1650);
  galois::reportPageAlloc("MeminfoMid");

  // preprocessing: find the nodes with out edges we will process and skip
  // over nodes with no out edges

  // find first node with out edges
  boost::filter_iterator<HasOut, Graph::iterator> begin =
      boost::make_filter_iterator(HasOut(&g), g.begin(), g.end());
  boost::filter_iterator<HasOut, Graph::iterator> end =
      boost::make_filter_iterator(HasOut(&g), g.end(), g.end());
  // adjustedEnd = last node we will process based on how many iterations
  // (i.e. sources) we want to do
  boost::filter_iterator<HasOut, Graph::iterator> adjustedEnd =
    iterLimit ? galois::safe_advance(begin, end, (int)iterLimit) : end;

  size_t iterations = std::distance(begin, adjustedEnd);

  // vector of nodes we want to process
  galois::gstl::Vector<GNode> v(begin, adjustedEnd);

  galois::gPrint("Num Nodes: ", NumNodes, " Start Node: ", startNode,
                 " Iterations: ",  iterations, "\n");

  // execute algorithm
  galois::StatTimer T;
  T.start();
  bcOuter.run(v);
  T.stop();

  bcOuter.printBCValues(0, std::min(10ul, NumNodes), std::cout, 6);

  if (printAll) bcOuter.printBCcertificate();
  if (forceVerify || !skipVerify) bcOuter.verify();

  galois::reportPageAlloc("MeminfoPost");

  return 0;
}
