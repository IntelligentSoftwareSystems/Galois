/** Page rank application -*- C++ -*-
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
 */

#include "Lonestar/BoilerPlate.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/LargeArray.h"
#include "galois/PerThreadContainer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>

#define DEBUG 0

namespace cll = llvm::cl;

const char* name = "Page Rank";
const char* desc =
    "Computes page ranks a la Page and Brin. This is a pull-style algorithm.";
const char* url = 0;

constexpr static const float ALPHA = (1 - 0.85);
constexpr static const float PR_INIT_VAL = 1.0;
constexpr static const float TOLERANCE = 1.0e-5;
constexpr static const unsigned MAX_ITER = 1000;
constexpr static const unsigned CHUNK_SIZE = 16;

cll::opt<std::string> filename(cll::Positional,
                               cll::desc("<tranpose of input graph>"),
                               cll::Required);
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(TOLERANCE));
cll::opt<unsigned int> maxIterations("maxIterations",
                                     cll::desc("Maximum iterations"),
                                     cll::init(MAX_ITER));


struct LNode {
  float value[2];
  uint32_t nout; // Compute the out degrees in the original graph

  float getPageRank() const { return value[1]; }
  float getPageRank(unsigned int it) const { return value[it & 1]; }
  void setPageRank(unsigned it, float v) { value[(it + 1) & 1] = v; }
  void finalize(void) { value[1] = value[0]; }
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>
  ::with_no_lockable<true>::type
  ::with_numa_alloc<true>::type
  Graph;
typedef typename Graph::GraphNode GNode;

void initNodeData(Graph& g) {
  galois::do_all(galois::iterate(g), 
      [&] (const GNode& n) {
      LNode& data   = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.value[0] = PR_INIT_VAL;
      data.value[1] = PR_INIT_VAL;
      data.nout     = 0;
    },
    galois::no_stats(), galois::loopname("Initialize"));
}

void computeOutDeg(Graph& graph) {
  galois::StatTimer t("computeOutDeg");
  t.start();

  galois::LargeArray<std::atomic<size_t> > vec;
  vec.allocateInterleaved(graph.size());

  galois::do_all(galois::iterate(graph),
      [&] (const GNode& src) {
        vec.constructAt(src, 0ul);
      },
      galois::no_stats(),
      galois::loopname("InitDegVec"));


  galois::do_all(galois::iterate(graph),
       [&] (const GNode& src) {
         for (auto nbr : graph.edges(src)) {
           GNode dst = graph.getEdgeDst(nbr);
           vec[dst].fetch_add(1ul);
         }
       },
       galois::steal(), 
       galois::chunk_size<CHUNK_SIZE>(),
       galois::no_stats(),
       galois::loopname("ComputeDeg"));

  galois::do_all(galois::iterate(graph),
      [&] (const GNode& src) {
        auto& srcData = graph.getData(src);
        srcData.nout = vec[src];
      },
      galois::no_stats(),
      galois::loopname("CopyDeg"));

  t.stop();

}

/*
void computeOutDeg(Graph& graph) {
  galois::StatTimer t("computeOutDeg");
  t.start();

  galois::PerThreadVector<uint32_t> perThrdVecs;

  galois::on_each([&] (const unsigned tid, const unsigned numT) {
      perThrdVecs.get(tid).resize(graph.size(), 0u);
      });

  galois::do_all(galois::iterate(graph),
       [&] (const GNode& src) {
         for (auto nbr : graph.edges(src)) {
           GNode dst = graph.getEdgeDst(nbr);
           perThrdVecs.get()[dst] += 1;
         }
       },
       galois::steal(), 
       galois::chunk_size<CHUNK_SIZE>(),
       galois::no_stats(),
       galois::loopname("ComputeDeg"));


  const unsigned numT = galois::getActiveThreads();

  galois::do_all(galois::iterate(graph),
      [&] (const GNode& src) {
        auto& srcData = graph.getData(src);
        for (unsigned i = 0; i < numT; ++i) {
          srcData.nout += perThrdVecs.get(i)[src];  
        }
      },
      galois::no_stats(),
      galois::loopname("ReduceDeg"));



  t.stop();


}
*/

void finalizePR(Graph& g) {
    galois::do_all(galois::iterate(g), 
        [&] (const GNode& n) {
          LNode& data   = g.getData(n, galois::MethodFlag::UNPROTECTED);
          data.finalize();
        },
        galois::no_stats(),
        galois::loopname("Finalize"));
}

void computePageRank(Graph& graph) {

    unsigned int iteration   = 0;
    galois::GReduceMax<float> max_delta;

    while (true) {
      galois::do_all(galois::iterate(graph), 
          [&] (const GNode& src) {
            LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
            float sum   = 0.0;

            for (auto jj = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
                      ej = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
                 jj != ej; ++jj) {
              GNode dst = graph.getEdgeDst(jj);

              LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              sum += ddata.getPageRank(iteration) / ddata.nout;
            }

            float value = sum * (1.0 - ALPHA) + ALPHA;
            float diff  = std::fabs(value - sdata.getPageRank(iteration));

            max_delta.update(diff);
            sdata.setPageRank(iteration, value);
          },
          galois::no_stats(), 
          galois::steal(),
          galois::chunk_size<CHUNK_SIZE>(),
          galois::loopname("PageRank"));

      float delta = max_delta.reduce();

#if DEBUG
      std::cout << "iteration: " << iteration << " max delta: " << delta
                << " small delta: " << sdelta << " ("
                << sdelta / (float)graph.size() << ")"
                << "\n";
#endif

      iteration += 1;
      if (delta <= tolerance || iteration >= maxIterations)
        break;
      max_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      finalizePR(graph);
    }
}

template <typename GNode>
struct TopPair {
  float value;
  GNode id;

  TopPair(float v, GNode i) : value(v), id(i) {}

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};

template <typename Graph>
static void printTop(Graph& graph, int topn) {
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair, GNode> Top;

  Top top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    float value           = n.getPageRank();
    Pair key(value, src);

    if ((int)top.size() < topn) {
      top.insert(std::make_pair(key, src));
      continue;
    }

    if (top.begin()->first < key) {
      top.erase(top.begin());
      top.insert(std::make_pair(key, src));
    }
  }

  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (typename Top::reverse_iterator ii = top.rbegin(), ei = top.rend();
       ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();

  Graph transposeGraph;
  std::cout << "Reading graph: " << filename << std::endl;
  galois::graphs::readGraph(transposeGraph, filename);
  std::cout << "Read " << transposeGraph.size() << " nodes, " 
    << transposeGraph.sizeEdges() << " edges" << std::endl;

  galois::preAlloc(numThreads + (2 * transposeGraph.size() *
                                 sizeof(typename Graph::node_data_type)) /
                                    galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running Edge Async push version, tolerance: " << tolerance
            << "\n";


  initNodeData(transposeGraph);
  computeOutDeg(transposeGraph);

  galois::StatTimer Tmain;
  Tmain.start();

  computePageRank(transposeGraph);

  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(transposeGraph, 10);
  }

  T.stop();

  return 0;
}
