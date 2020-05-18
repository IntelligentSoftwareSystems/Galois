/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

// modified by Joyce Whang <joyce@cs.utexas.edu>

#include "galois/config.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/runtime/PerThreadStorage.h"
#include "galois/runtime/Barrier.h"
#include "galois/Graph/LCGraph.h"
#include "galois/Graph/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/worklists/WorkListDebug.h"

#include GALOIS_CXX11_STD_HEADER(atomic)
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <numa.h>

#include "PageRank.h"
#include "GraphLabAlgo.h"
#include "LigraAlgo.h"
#include "PagerankDelta.h"
#define PAGESIZE (4096)

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin";
static const char* url  = 0;
// static const float tolerance = 0.0001; // Joyce
static const float amp  = (1 / tolerance) * (-1000); // Joyce
static const float amp2 = (1 / tolerance) * (-1);    // Joyce

enum Algo {
  graphlab,
  graphlabAsync,
  ligra,
  ligraChi,
  pull,
  pull2,
  polypush,
  polypull,
  polypull2,
  duppull,
  push,
  pagerankWorklist,
  serial,
  synch,   // Joyce
  prt_rsd, // Joyce
  prt_deg, // Joyce
};

cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"),
                               cll::Required);
static cll::opt<std::string>
    transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph",
                                     cll::desc("Input graph is symmetric"));
static cll::opt<std::string>
    outputPullFilename("outputPull",
                       cll::desc("Precompute data for Pull algorithm to file"));
cll::opt<unsigned int> maxIterations("maxIterations",
                                     cll::desc("Maximum iterations"),
                                     cll::init(100000000));
cll::opt<unsigned int>
    memoryLimit("memoryLimit",
                cll::desc("Memory limit for out-of-core algorithms (in MB)"),
                cll::init(~0U));
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::pull, "pull",
                           "Use precomputed data perform pull-based algorithm"),
                clEnumValN(Algo::pull2, "pull2", "Use pull-based algorithm"),
                clEnumValN(Algo::polypush, "polypush",
                           "polymer-like push-based algorithm"),
                clEnumValN(Algo::polypull, "polypull",
                           "polymer-like pull-based algorithm"),
                clEnumValN(Algo::polypull2, "polypull2",
                           "polymer-like more partitions pull-based algorithm"),
                clEnumValN(Algo::duppull, "duppull",
                           "numa-awared pull-based algorithm with duplicates"),
                clEnumValN(Algo::push, "push",
                           "Use precomputed data perform push-based algorithm"),
                clEnumValN(Algo::serial, "serial",
                           "Compute PageRank in serial"),
                clEnumValN(Algo::synch, "synch", "Synchronous version..."),
                clEnumValN(Algo::prt_rsd, "prt_rsd",
                           "Prioritized (max. residual) version..."),
                clEnumValN(Algo::prt_deg, "prt_deg",
                           "Prioritized (degree biased) version..."),
                clEnumValN(Algo::graphlab, "graphlab",
                           "Use GraphLab programming model"),
                clEnumValN(Algo::graphlabAsync, "graphlabAsync",
                           "Use GraphLab-Asynchronous programming model"),
                clEnumValN(Algo::ligra, "ligra", "Use Ligra programming model"),
                clEnumValN(Algo::ligraChi, "ligraChi",
                           "Use Ligra and GraphChi programming model"),
                clEnumValN(Algo::pagerankWorklist, "pagerankWorklist",
                           "Use worklist-based algorithm"),
                clEnumValEnd),
    cll::init(Algo::pull));

struct SerialAlgo {
  typedef galois::graphs::LC_CSR_Graph<PNode, void>::with_no_lockable<
      true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Serial"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      g.getData(n).value = 1.0;
      g.getData(n).accum.write(0.0);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;
    unsigned int numNodes  = graph.size();

    while (true) {
      float max_delta          = std::numeric_limits<float>::min();
      unsigned int small_delta = 0;
      double sum_delta         = 0;

      for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        GNode src    = *ii;
        PNode& sdata = graph.getData(src);
        int neighbors =
            std::distance(graph.edge_begin(src), graph.edge_end(src));
        for (auto jj = graph.edge_begin(src), ej = graph.edge_end(src);
             jj != ej; ++jj) {
          GNode dst    = graph.getEdgeDst(jj);
          PNode& ddata = graph.getData(dst);
          float delta  = sdata.value / neighbors;
          ddata.accum.write(ddata.accum.read() + delta);
        }
      }

      for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        GNode src    = *ii;
        PNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
        float value  = (1.0 - alpha) * sdata.accum.read() + alpha;
        float diff   = std::fabs(value - sdata.value);
        if (diff <= tolerance)
          ++small_delta;
        if (diff > max_delta)
          max_delta = diff;
        sum_delta += diff;
        sdata.value = value;
        sdata.accum.write(0);
      }

      iteration += 1;

      std::cout << "iteration: " << iteration << " sum delta: " << sum_delta
                << " max delta: " << max_delta
                << " small delta: " << small_delta << " ("
                << small_delta / (float)numNodes << ")"
                << "\n";

      if (max_delta <= tolerance || iteration >= maxIterations)
        break;
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }
  }
};

struct PullAlgo {
  struct LNode {
    float value[2];

    float getPageRank() { return value[1]; }
    float getPageRank(unsigned int it) { return value[it & 1]; }
    void setPageRank(unsigned it, float v) { value[(it + 1) & 1] = v; }
  };
  typedef galois::graphs::LC_InlineEdge_Graph<LNode, float>::
      with_compressed_node_ptr<true>::type ::with_no_lockable<
          true>::type ::with_numa_alloc<false>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Pull"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<size_t> small_delta;
  galois::GAccumulator<float> sum_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.value[0] = 1.0;
      data.value[1] = 1.0;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    PullAlgo* self;
    Graph& graph;
    unsigned int iteration;

    Process(PullAlgo* s, Graph& g, unsigned int i)
        : self(s), graph(g), iteration(i) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
      double sum   = 0;

      for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        float w   = graph.getEdgeData(jj);

        LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);
        sum += ddata.getPageRank(iteration) * w;
      }

      float value = sum * (1.0 - alpha) + alpha;
      float diff  = std::fabs(value - sdata.getPageRank(iteration));

      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
      self->sum_delta.update(diff);
      sdata.setPageRank(iteration, value);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;

    while (true) {
      galois::for_each(
          graph, Process(this, graph, iteration),
          galois::wl<galois::worklists::PerSocketChunkFIFO<256>>());
      iteration += 1;

      float delta   = max_delta.reduce();
      size_t sdelta = small_delta.reduce();

      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta.reduce()
                << " max delta: " << delta << " small delta: " << sdelta << " ("
                << sdelta / (float)graph.size() << ")"
                << "\n";

      if (delta <= tolerance || iteration >= maxIterations)
        break;
      max_delta.reset();
      small_delta.reset();
      sum_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }
  }
};

struct PullAlgo2 {
  struct LNode {
    float value[2];
    unsigned int nout;
    float getPageRank() { return value[1]; }
    float getPageRank(unsigned int it) { return value[it & 1]; }
    void setPageRank(unsigned it, float v) { value[(it + 1) & 1] = v; }
  };

  typedef typename galois::graphs::LC_InlineEdge_Graph<LNode, void>::
      with_numa_alloc<true>::type ::with_no_lockable<true>::type InnerGraph;

  //! [Define LC_InOut_Graph]
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  //! [Define LC_InOut_Graph]

  typedef typename Graph::GraphNode GNode;

  std::string name() const { return "Pull2"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.value[0] = 1.0;
      data.value[1] = 1.0;
      int outs      = std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                               g.edge_end(n, galois::MethodFlag::NONE));
      data.nout     = outs;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    PullAlgo2* self;
    Graph& graph;
    unsigned int iteration;

    Process(PullAlgo2* s, Graph& g, unsigned int i)
        : self(s), graph(g), iteration(i) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {

      LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);

      //! [Access in-neighbors of LC_InOut_Graph]
      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.in_edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst    = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);
        sum += ddata.getPageRank(iteration) / ddata.nout;
      }
      //! [Access in-neighbors of LC_InOut_Graph]

      float value = (1.0 - alpha) * sum + alpha;
      float diff  = std::fabs(value - sdata.getPageRank(iteration));
      sdata.setPageRank(iteration, value);
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;

    while (true) {
      galois::for_each(graph, Process(this, graph, iteration));
      iteration += 1;

      float delta   = max_delta.reduce();
      size_t sdelta = small_delta.reduce();

      std::cout << "iteration: " << iteration << " max delta: " << delta
                << " small delta: " << sdelta << " ("
                << sdelta / (float)graph.size() << ")"
                << "\n";

      if (delta <= tolerance || iteration >= maxIterations) {
        break;
      }
      max_delta.reset();
      small_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }
  }
};

//-------------michael's code start-----------------------
//-------------polymer-like pagerank(tp-driven push-based numa-aware)

template <typename Graph>
static void printTop(Graph& graph, int topn);

int idcount = 0;

inline bool LCAS(long* ptr, long oldv, long newv) {
  unsigned char ret;
  /* Note that sete sets a 'byte' not the word */
  __asm__ __volatile__("  lock\n"
                       "  cmpxchgq %2,%1\n"
                       "  sete %0\n"
                       : "=q"(ret), "=m"(*ptr)
                       : "r"(newv), "m"(*ptr), "a"(oldv)
                       : "memory");
  return ret;
}

// compare and swap on 4 byte quantity
inline bool SCAS(int* ptr, int oldv, int newv) {
  unsigned char ret;
  /* Note that sete sets a 'byte' not the word */
  __asm__ __volatile__("  lock\n"
                       "  cmpxchgl %2,%1\n"
                       "  sete %0\n"
                       : "=q"(ret), "=m"(*ptr)
                       : "r"(newv), "m"(*ptr), "a"(oldv)
                       : "memory");
  return ret;
}

template <class ET>
inline bool CAS(ET* ptr, ET oldv, ET newv) {
  if (sizeof(ET) == 8) {
    return LCAS((long*)ptr, *((long*)&oldv), *((long*)&newv));
    // return __sync_bool_compare_and_swap((long*)ptr, (long)oldv, (long)newv);
  } else if (sizeof(ET) == 4) {
    return SCAS((int*)ptr, *((int*)&oldv), *((int*)&newv));
    // return __sync_bool_compare_and_swap((int*)ptr, (int)oldv, (int)newv);
  } else {
    std::cout << "CAS bad length" << std::endl;
    abort();
  }
}

template <class ET>
inline void writeAdd(ET* a, ET b) {
  volatile ET newV, oldV;
  do {
    oldV = *a;
    newV = oldV + b;
  } while (!CAS(a, oldV, newV));
}

struct PolyPush {

  struct LocalData {

    unsigned int nnodes;
    unsigned int max_nodes_per_package;
    unsigned int max_edges_per_package;

    bool* Bit_curr;
    bool* Bit_next;
    float* PR_curr;
    float* PR_next;
    unsigned int* nodeList;
    unsigned int* edgeList;
    unsigned int* outDegree;
    unsigned int* inDegree;
    LocalData(unsigned int nnodes1, unsigned int max_nodes_per_package1,
              unsigned int max_edges_per_package1)
        : nnodes(nnodes1), max_nodes_per_package(max_nodes_per_package1),
          max_edges_per_package(max_edges_per_package1){};

    void alloc() {
      using namespace galois::runtime::MM;
      /*Bit_curr = (bool * ) largeAlloc( max_nodes_per_package * sizeof(bool));
      Bit_next = (bool * ) largeAlloc( max_nodes_per_package * sizeof(bool));
      PR_curr = (float * ) largeAlloc( max_nodes_per_package * sizeof(float));
      PR_next = (float * ) largeAlloc( max_nodes_per_package * sizeof(float));
      nodeList = (int * ) largeAlloc( (nnodes + 1) * sizeof(int));
      edgeList = (int * ) largeAlloc( max_edges_per_package * sizeof(int));*/
      /*Bit_curr = (bool * ) std::malloc( max_nodes_per_package * sizeof(bool));
      Bit_next = (bool * ) std::malloc( max_nodes_per_package * sizeof(bool));
      PR_curr = (float * ) std::malloc( max_nodes_per_package * sizeof(float));
      PR_next = (float * ) std::malloc( max_nodes_per_package * sizeof(float));
      nodeList = (int * ) std::malloc( (nnodes + 1) * sizeof(int));
      edgeList = (int * ) std::malloc( max_edges_per_package * sizeof(int));*/
      Bit_curr = (bool*)numa_alloc_local(max_nodes_per_package * sizeof(bool));
      Bit_next = (bool*)numa_alloc_local(max_nodes_per_package * sizeof(bool));
      PR_curr = (float*)numa_alloc_local(max_nodes_per_package * sizeof(float));
      PR_next = (float*)numa_alloc_local(max_nodes_per_package * sizeof(float));
      nodeList =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));
      edgeList = (unsigned int*)numa_alloc_local(max_edges_per_package *
                                                 sizeof(unsigned int));
      outDegree =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));
      inDegree =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));

      next_reset();
    }

    void mfree() {
      using namespace galois::runtime::MM;
      /*largeFree( (void *)Bit_curr, max_nodes_per_package * sizeof(bool));
      largeFree( (void *)Bit_next, max_nodes_per_package * sizeof(bool));
      largeFree( (void *)PR_curr, max_nodes_per_package * sizeof(float));
      largeFree( (void *)PR_next, max_nodes_per_package * sizeof(float));
      largeFree( (void *)nodeList, (nnodes + 1) * sizeof(int));
      largeFree( (void *)edgeList, max_edges_per_package * sizeof(int));*/
      /*std::free( (void *)Bit_curr);
      std::free( (void *)Bit_next);
      std::free( (void *)PR_curr);
      std::free( (void *)PR_next);
      std::free( (void *)nodeList);
      std::free( (void *)edgeList);*/
      numa_free((void*)Bit_curr, max_nodes_per_package * sizeof(bool));
      numa_free((void*)Bit_next, max_nodes_per_package * sizeof(bool));
      numa_free((void*)PR_curr, max_nodes_per_package * sizeof(float));
      numa_free((void*)PR_next, max_nodes_per_package * sizeof(float));
      numa_free((void*)nodeList, (nnodes + 1) * sizeof(unsigned int));
      numa_free((void*)edgeList, max_edges_per_package * sizeof(unsigned int));
      numa_free((void*)outDegree, (nnodes + 1) * sizeof(unsigned int));
      numa_free((void*)inDegree, (nnodes + 1) * sizeof(unsigned int));
    }

    void next_reset() {
      memset(PR_next, 0, max_nodes_per_package * sizeof(float));
      memset(Bit_next, 0, max_nodes_per_package * sizeof(bool));
      // printf(" %d %d %f %f\n",Bit_next[0], Bit_next[max_nodes_per_package -
      // 1], PR_next[0], PR_next[max_nodes_per_package - 1] );
    }
  };

  struct LNode {
    unsigned int outDegree;
    unsigned int inDegree;
    unsigned int id;
    float pagerank;
    float getPageRank() { return pagerank; }
    void setPageRank(float pr) { pagerank = pr; }
  };
  typedef galois::graphs::LC_InlineEdge_Graph<LNode, void>::with_no_lockable<
      true>::type ::with_numa_alloc<false>::type // interleaved allocate
      InnerGraph;

  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  /*typedef galois::graphs::LC_CSR_Graph<LNode,void>
    ::with_no_lockable<true>::type Graph;*/
  typedef Graph::GraphNode GNode;

  typedef galois::runtime::PerPackageStorage<LocalData> PerPackageData;

  typedef galois::runtime::Barrier Barrier;

  std::string name() const { return "polymer push"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<float> sum_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::NONE);
      unsigned int outs =
          std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                        g.edge_end(n, galois::MethodFlag::NONE));
      data.outDegree = outs;
      unsigned int ins =
          std::distance(g.in_edge_begin(n, galois::MethodFlag::NONE),
                        g.in_edge_end(n, galois::MethodFlag::NONE));
      data.inDegree = ins;
      GNode start   = *g.begin();
      data.id       = g.idFromNode(n); // n - start;
    }
  };

  struct Copy {
    Graph& graph;
    PerPackageData& packageData;
    unsigned int* nodesPerPackage;
    unsigned int* nodesStartPackage;
    unsigned int nPackages;
    Copy(Graph& g, PerPackageData& pData, unsigned int* nodesP,
         unsigned int nPackages)
        : graph(g), packageData(pData), nodesPerPackage(nodesP),
          nPackages(nPackages) {
      nodesStartPackage            = new unsigned int[nPackages + 1];
      nodesStartPackage[0]         = 0;
      nodesStartPackage[nPackages] = graph.size();
      for (int i = 1; i < nPackages; i++) {
        nodesStartPackage[i] = nodesStartPackage[i - 1] + nodesPerPackage[i];
      }
    }
    void operator()(unsigned tid, unsigned numT) {
      if (galois::runtime::LL::isPackageLeader(tid)) {
        unsigned int currPackage =
            galois::runtime::LL::getPackageForThread(tid);
        for (auto ii = graph.begin(), end = graph.end(); ii != end; ii++) {
          GNode src    = *ii;
          LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);

          if (sdata.id >= nodesStartPackage[currPackage] &&
              sdata.id < nodesStartPackage[currPackage + 1])
            sdata.pagerank =
                packageData.getLocal()
                    ->PR_curr[sdata.id - nodesStartPackage[currPackage]];
        }

        packageData.getLocal()->mfree();
      }
    }
  };

  struct distributeEdges {
    Graph& graph;
    PerPackageData& packageData;
    unsigned int* nodesPerPackage;
    unsigned int* edgesPerPackage;

    distributeEdges(Graph& g, PerPackageData& p, unsigned int* nodes,
                    unsigned int* edges)
        : graph(g), packageData(p), nodesPerPackage(nodes),
          edgesPerPackage(edges){};

    void operator()(unsigned tid, unsigned numT) {
      if (galois::runtime::LL::isPackageLeader(tid)) {
        // printf("tid: %d\n", tid);
        packageData.getLocal()->alloc();

        unsigned int nnodes = graph.size();
        unsigned int currPackage =
            galois::runtime::LL::getPackageForThread(tid);
        unsigned int* outDegree = packageData.getLocal()->outDegree;
        unsigned int* inDegree  = packageData.getLocal()->inDegree;
        packageData.getLocal()->nodeList[nnodes] = edgesPerPackage[currPackage];
        // printf("tid: %d, edgesPerPackage: %d", tid,
        // edgesPerPackage[currPackage]);
        unsigned int rangeLow = 0;
        for (int i = 0; i < currPackage; i++) {
          rangeLow += nodesPerPackage[i];
        }
        unsigned int rangeHi = rangeLow + nodesPerPackage[currPackage];

        unsigned int edgeCount = 0;
        // std::cerr << "tid: " << tid <<"range:"<<rangeLow<<"-"<<rangeHi<<
        // std::endl; printf("work0.3 %d %d\n", rangeLow, rangeHi);
        unsigned int indegree_temp = 0;

        // std::cerr << "tid: " << tid <<"edgeCount:"<<edgeCount<< " inDegree
        // "<< indegree_temp<<std::endl;
        edgeCount = 0;
        for (auto ii = graph.begin(), end = graph.end(); ii != end; ii++) {
          GNode src    = *ii;
          LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
          // printf("%d %d\n", sdata.id, edgeCount);
          packageData.getLocal()->nodeList[sdata.id] = edgeCount;
          outDegree[sdata.id]                        = sdata.outDegree;
          inDegree[sdata.id]                         = sdata.inDegree;
          for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                    ej = graph.edge_end(src, galois::MethodFlag::NONE);
               jj != ej; ++jj) {
            GNode dst    = graph.getEdgeDst(jj);
            LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);

            if (ddata.id < rangeHi && ddata.id >= rangeLow) {
              packageData.getLocal()->edgeList[edgeCount] = ddata.id;
              edgeCount++;
            }
          }
        }
        // std::cerr << "tid1: " << tid << std::endl;
        // printf("work0.5\n");
        //===set node bit and PR==============
        for (unsigned int i = 0; i < nodesPerPackage[currPackage]; i++) {
          packageData.getLocal()->PR_curr[i]  = 1.0;
          packageData.getLocal()->PR_next[i]  = 0.0;
          packageData.getLocal()->Bit_curr[i] = true;
          packageData.getLocal()->Bit_next[i] = false;
        }
      }
    }
  };

  struct PartitionInfo {
    unsigned int numThreads, nnodes, nPackages, coresPerPackage,
        coresLastPackage;
    unsigned int* nodesPerPackage;
    unsigned int* nodesStartPackage;
    unsigned int* edgesPerPackage;
    unsigned int* nodesPerCore;
    unsigned int* nodesStartCore;
    unsigned int* nodesPerCoreLastPkg;
    unsigned int* nodesStartCoreLastPkg;
    PartitionInfo(unsigned int nnodes, unsigned int numThreads,
                  unsigned int nPackages, unsigned int coresPerPackage,
                  unsigned int coresLastPackage)
        : nnodes(nnodes), numThreads(numThreads), nPackages(nPackages),
          coresPerPackage(coresPerPackage), coresLastPackage(coresLastPackage) {
      edgesPerPackage   = new unsigned int[nPackages];
      nodesPerPackage   = new unsigned int[nPackages];
      nodesStartPackage = new unsigned int[nPackages + 1];

      nodesPerCore   = new unsigned int[coresPerPackage];
      nodesStartCore = new unsigned int[coresPerPackage];

      nodesPerCoreLastPkg   = new unsigned int[coresLastPackage];
      nodesStartCoreLastPkg = new unsigned int[coresLastPackage];

      memset(edgesPerPackage, 0, sizeof(unsigned int) * nPackages);
      memset(nodesPerPackage, 0, sizeof(unsigned int) * nPackages);
      memset(nodesStartPackage, 0, sizeof(unsigned int) * nPackages);

      memset(nodesPerCore, 0, sizeof(unsigned int) * coresPerPackage);
      memset(nodesStartCore, 0, sizeof(unsigned int) * coresPerPackage);
      memset(nodesPerCoreLastPkg, 0, sizeof(unsigned int) * coresLastPackage);
      memset(nodesStartCoreLastPkg, 0, sizeof(unsigned int) * coresLastPackage);
    }

    void partitionByDegree(Graph& graph, unsigned int numThreads,
                           unsigned int* nodesPerThread,
                           unsigned int* edgesPerThread) {
      unsigned int n = graph.size();
      // int *degrees = new int [n];

      // parallel_for(intT i = 0; i < n; i++) degrees[i] =
      // GA.V[i].getInDegree();}

      unsigned int* accum = new unsigned int[numThreads];
      for (int i = 0; i < numThreads; i++) {
        accum[i]          = 0;
        nodesPerThread[i] = 0;
      }

      unsigned int averageDegree = graph.sizeEdges() / numThreads;
      std::cout << "averageDegree is " << averageDegree << std::endl;
      unsigned int counter = 0;
      for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        GNode src    = *ii;
        LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
        accum[counter] += sdata.inDegree;
        nodesPerThread[counter]++;
        if ((accum[counter] >= averageDegree && counter < numThreads - 1) ||
            ii == ei - 1) {
          edgesPerThread[counter] = accum[counter];
          counter++;
        }
      }
      delete[] accum;
    }

    void subPartitionByDegree(Graph& graph, unsigned int nCores,
                              unsigned int* nodesPerCore,
                              unsigned int* nodesStartCore) {
      unsigned int n = graph.size();

      unsigned int aveNodes = n / nCores;

      for (unsigned int i = 0; i < nCores - 1; i++) {
        nodesPerCore[i]   = aveNodes;
        nodesStartCore[i] = aveNodes * i;
      }
      nodesStartCore[nCores - 1] = aveNodes * (nCores - 1);
      nodesPerCore[nCores - 1]   = n - nodesStartCore[nCores - 1];
    }

    void partition(Graph& graph) {
      unsigned int* nodesPerThread = new unsigned int[numThreads];
      unsigned int* edgesPerThread = new unsigned int[numThreads];

      partitionByDegree(graph, numThreads, nodesPerThread, edgesPerThread);

      for (unsigned int i = 0; i < nPackages; i++) {
        unsigned int coresCurrPackage;
        if (i == nPackages - 1)
          coresCurrPackage = coresLastPackage;
        else
          coresCurrPackage = coresPerPackage;

        nodesPerPackage[i] = 0;
        edgesPerPackage[i] = 0;

        for (unsigned int j = 0; j < coresCurrPackage; j++) {
          nodesPerPackage[i] += nodesPerThread[coresPerPackage * i + j];
          edgesPerPackage[i] += edgesPerThread[coresPerPackage * i + j];
        }

        if (i > 0)
          nodesStartPackage[i] =
              nodesStartPackage[i - 1] + nodesPerPackage[i - 1];
      }
      nodesStartPackage[nPackages] =
          nodesStartPackage[nPackages - 1] + nodesPerPackage[nPackages - 1];

      subPartitionByDegree(graph, coresPerPackage, nodesPerCore,
                           nodesStartCore);
      subPartitionByDegree(graph, coresLastPackage, nodesPerCoreLastPkg,
                           nodesStartCoreLastPkg);

      // printf("nodesStartPackage: %d %d %d\n", nodesStartPackage[0],
      // nodesStartPackage[1], nodesStartPackage[2]);
      /*for(int i = 0; i < numThreads; i++)
      {
        printf("%d %d ", i, nodesPerThread[i]);
      }
      printf("\n");
      for(int i = 0; i < numThreads; i++)
      {
        printf("%d %d ", i, edgesPerThread[i]);
      }
      printf("\n");*/
      delete[] nodesPerThread;
      delete[] edgesPerThread;
    }

    void mfree() {
      delete[] edgesPerPackage;
      delete[] nodesPerPackage;
      delete[] nodesStartPackage;

      delete[] nodesPerCore;
      delete[] nodesStartCore;

      delete[] nodesPerCoreLastPkg;
      delete[] nodesStartCoreLastPkg;
    }

    void print() {
      printf("ePerPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, edgesPerPackage[i]);
        printf("\n");
      }
      printf("nPerPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, nodesPerPackage[i]);
        printf("\n");
      }
      printf("nstartPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, nodesStartPackage[i]);
        printf("\n");
      }
      printf("nPerCore: ");
      for (int i = 0; i < coresPerPackage; i++) {

        printf("%d %d ", i, nodesPerCore[i]);
        printf("\n");
      }
      printf("nStartCore: ");
      for (int i = 0; i < coresPerPackage; i++) {

        printf("%d %d ", i, nodesStartCore[i]);
        printf("\n");
      }
      printf("nPerCoreLastPkg: ");
      for (int i = 0; i < coresLastPackage; i++) {

        printf("%d %d ", i, nodesPerCoreLastPkg[i]);
        printf("\n");
      }
      printf("nPerCorePkg: ");
      for (int i = 0; i < coresLastPackage; i++) {

        printf("%d %d", i, nodesStartCoreLastPkg[i]);
        printf("\n");
      }
    }
  };

  struct Process {
    PolyPush* self;
    PerPackageData& packageData;
    PartitionInfo& partitionInfo;
    Barrier& gbarrier;
    Process(PerPackageData& p, PartitionInfo& partitionInfo, Barrier& gbarrier,
            PolyPush* self)
        : packageData(p), partitionInfo(partitionInfo), gbarrier(gbarrier),
          self(self) {}
    void update(float* PR_next, unsigned offset, float value) {
      writeAdd(PR_next + offset, value);
      // PR_next[offset] += value;
    }
    void operator()(unsigned tid, unsigned numT) {
      unsigned leader      = galois::runtime::LL::getLeaderForThread(tid);
      unsigned currPackage = galois::runtime::LL::getPackageForThread(tid);

      unsigned int nPackages = partitionInfo.nPackages;

      unsigned int threadStart =
          (currPackage == nPackages - 1)
              ? partitionInfo.nodesStartCoreLastPkg[tid - leader]
              : partitionInfo.nodesStartCore[tid - leader];
      unsigned int nodesCurrCore =
          (currPackage == nPackages - 1)
              ? partitionInfo.nodesPerCoreLastPkg[tid - leader]
              : partitionInfo.nodesPerCore[tid - leader];

      unsigned int threadEnd = threadStart + nodesCurrCore;

      unsigned int activePackage = 0;
      bool *Bit_curr, *Bit_next;
      unsigned int *edgelist, *nodelist, *outDegree, *inDegree;
      float *PR_curr, *PR_next;
      unsigned int startOffset, packageOffset, nout;
      unsigned int firstEdge, dst, localnout;
      float delta;

      unsigned int* nodesStartPackage = partitionInfo.nodesStartPackage;
      for (unsigned int i = 0; i < nPackages; i++) {
        if (threadStart >= nodesStartPackage[i] &&
            threadStart < nodesStartPackage[i + 1]) {
          activePackage = i;
          startOffset   = nodesStartPackage[i];
          break;
        }
      }
      nodelist  = packageData.getLocal()->nodeList;
      edgelist  = packageData.getLocal()->edgeList;
      outDegree = packageData.getLocal()->outDegree;
      inDegree  = packageData.getLocal()->inDegree;
      if (currPackage == activePackage) {
        Bit_curr = packageData.getLocal()->Bit_curr;
        PR_curr  = packageData.getLocal()->PR_curr;
      } else {
        Bit_curr = packageData.getRemoteByPkg(activePackage)->Bit_curr;
        PR_curr  = packageData.getRemoteByPkg(activePackage)->PR_curr;
      }

      float diff;
      PR_next  = packageData.getLocal()->PR_next;
      Bit_next = packageData.getLocal()->Bit_next;

      gbarrier.wait();
      // printf("tid: %d start: %d end: %d \n", tid, threadStart, threadEnd);
      // gbarrier.wait();
      for (unsigned int i = threadStart; i < threadEnd; i++) {
        packageOffset = i - startOffset;
        if (i == nodesStartPackage[activePackage + 1]) {
          activePackage++;
          startOffset   = nodesStartPackage[activePackage];
          packageOffset = 0; // i - startOffset;
          if (currPackage == activePackage) {
            Bit_curr = packageData.getLocal()->Bit_curr;
            PR_curr  = packageData.getLocal()->PR_curr;
          } else {
            Bit_curr = packageData.getRemoteByPkg(activePackage)->Bit_curr;
            PR_curr  = packageData.getRemoteByPkg(activePackage)->PR_curr;
          }
        }

        if (Bit_curr[packageOffset] && outDegree[i] != 0) {
          nout  = outDegree[i];
          delta = PR_curr[packageOffset] / (float)nout;
          for (unsigned int j = nodelist[i]; j < nodelist[i + 1]; j++) {
            dst = edgelist[j];
            if (delta >= tolerance) {
              unsigned int offset = dst - nodesStartPackage[currPackage];
              update(PR_next, offset, delta);
              Bit_next[offset] = true;

              diff = delta;
            }
          }
        }
      }
    }
  };

  struct Process2 {
    PolyPush* self;
    PerPackageData& packageData;
    PartitionInfo& partitionInfo;
    Barrier& gbarrier;
    Process2(PerPackageData& p, PartitionInfo& partitionInfo, Barrier& gbarrier,
             PolyPush* self)
        : packageData(p), partitionInfo(partitionInfo), gbarrier(gbarrier),
          self(self) {}

    void damping(float* PR_next, unsigned tid, PolyPush* self,
                 PerPackageData& packageData) {
      unsigned int localtid =
          tid - galois::runtime::LL::getLeaderForThread(tid);
      unsigned int coresCurrPackage;
      unsigned int nodesPerCore;
      unsigned int currPackage = galois::runtime::LL::getPackageForThread(tid);
      if (currPackage != partitionInfo.nPackages - 1) {
        coresCurrPackage = partitionInfo.coresPerPackage;
      } else {
        coresCurrPackage = partitionInfo.coresLastPackage;
      }

      nodesPerCore =
          partitionInfo.nodesPerPackage[currPackage] / coresCurrPackage;
      float diff;
      if (localtid != coresCurrPackage - 1) {
        for (unsigned int i = localtid * nodesPerCore;
             i < (localtid + 1) * nodesPerCore; i++) {
          PR_next[i] = alpha2 * PR_next[i] + 1 - alpha2;
          /*diff = packageData.getLocal()->PR_curr[i] - PR_next[i];
          self->max_delta.update(diff);
          self->sum_delta.update(diff);*/
        }
      } else {
        for (unsigned int i = localtid * nodesPerCore;
             i < partitionInfo.nodesPerPackage[currPackage]; i++) {
          PR_next[i] = alpha2 * PR_next[i] + 1 - alpha2;
          /*diff = packageData.getLocal()->PR_curr[i] - PR_next[i];
          self->max_delta.update(diff);
          self->sum_delta.update(diff);*/
        }
      }
    }

    void operator()(unsigned tid, unsigned numT) {
      float* PR_next = packageData.getLocal()->PR_next;
      damping(PR_next, tid, self, packageData);
      gbarrier.wait();
      if (galois::runtime::LL::isPackageLeader(tid)) {
        float* temp;
        temp                            = packageData.getLocal()->PR_curr;
        packageData.getLocal()->PR_curr = packageData.getLocal()->PR_next;
        packageData.getLocal()->PR_next = temp;

        bool* tempBit;
        tempBit                          = packageData.getLocal()->Bit_curr;
        packageData.getLocal()->Bit_curr = packageData.getLocal()->Bit_next;
        packageData.getLocal()->Bit_next = tempBit;

        packageData.getLocal()->next_reset();
      }
    }
  };

  void operator()(Graph& graph) {
    // nPackages = LL::getMaxPackages();
    unsigned int coresPerPackage = galois::runtime::LL::getMaxCores() /
                                   galois::runtime::LL::getMaxPackages();

    unsigned int nPackages        = (numThreads - 1) / coresPerPackage + 1;
    unsigned int coresLastPackage = numThreads % coresPerPackage;
    if (coresLastPackage == 0)
      coresLastPackage = coresPerPackage;
    Barrier& gbarrier = galois::runtime::getSystemBarrier();

    unsigned int nnodes = graph.size();
    std::cout << "nnodes:" << nnodes << " nedges:" << graph.sizeEdges()
              << std::endl;

    PartitionInfo partitionInfo(nnodes, numThreads, nPackages, coresPerPackage,
                                coresLastPackage);

    partitionInfo.partition(graph);

    unsigned int max_nodes_per_package = 0, max_edges_per_package = 0;
    unsigned int* nodesPerPackage = partitionInfo.nodesPerPackage;
    unsigned int* edgesPerPackage = partitionInfo.edgesPerPackage;
    for (int i = 0; i < nPackages; i++) {
      max_nodes_per_package = (max_nodes_per_package > nodesPerPackage[i])
                                  ? max_nodes_per_package
                                  : nodesPerPackage[i];
      max_edges_per_package = (max_edges_per_package > edgesPerPackage[i])
                                  ? max_edges_per_package
                                  : edgesPerPackage[i];
    }

    PerPackageData packageData(nnodes, max_nodes_per_package,
                               max_edges_per_package);

    galois::on_each(
        distributeEdges(graph, packageData, nodesPerPackage, edgesPerPackage));
    unsigned int iteration = 0;
    galois::StatTimer T("pure time");

    // partitionInfo.print();
    T.start();
    while (true) {
      galois::on_each(Process(packageData, partitionInfo, gbarrier, this));
      galois::on_each(Process2(packageData, partitionInfo, gbarrier, this));
      iteration += 1;

      float delta = max_delta.reduce();

      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta.reduce()
                << " max delta: " << delta << "\n";
      delta = 1;
      if (delta <= tolerance || iteration >= maxIterations)
        break;

      max_delta.reset();
      sum_delta.reset();
      /*galois::on_each(Copy(graph, packageData, nodesPerPackage,nPackages));
      printTop(graph, 10);*/
    }
    T.stop();
    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }
    galois::on_each(Copy(graph, packageData, nodesPerPackage, nPackages));

    partitionInfo.mfree();

    /*if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }*/
  }
};

struct PolyPull {

  struct LocalData {

    unsigned int nnodes;
    unsigned int max_nodes_per_package;
    unsigned int max_edges_per_package;

    bool* Bit_curr;
    bool* Bit_next;
    float* PR_curr;
    float* PR_next;
    unsigned int* nodeList;
    unsigned int* edgeList;
    unsigned int* outDegree;
    unsigned int* inDegree;
    LocalData(unsigned int nnodes1, unsigned int max_nodes_per_package1,
              unsigned int max_edges_per_package1)
        : nnodes(nnodes1), max_nodes_per_package(max_nodes_per_package1),
          max_edges_per_package(max_edges_per_package1){};

    void alloc() {
      using namespace galois::runtime::MM;
      /*Bit_curr = (bool * ) largeAlloc( max_nodes_per_package * sizeof(bool));
      Bit_next = (bool * ) largeAlloc( max_nodes_per_package * sizeof(bool));
      PR_curr = (float * ) largeAlloc( max_nodes_per_package * sizeof(float));
      PR_next = (float * ) largeAlloc( max_nodes_per_package * sizeof(float));
      nodeList = (int * ) largeAlloc( (nnodes + 1) * sizeof(int));
      edgeList = (int * ) largeAlloc( max_edges_per_package * sizeof(int));*/
      /*Bit_curr = (bool * ) std::malloc( max_nodes_per_package * sizeof(bool));
      Bit_next = (bool * ) std::malloc( max_nodes_per_package * sizeof(bool));
      PR_curr = (float * ) std::malloc( max_nodes_per_package * sizeof(float));
      PR_next = (float * ) std::malloc( max_nodes_per_package * sizeof(float));
      nodeList = (int * ) std::malloc( (nnodes + 1) * sizeof(int));
      edgeList = (int * ) std::malloc( max_edges_per_package * sizeof(int));*/
      Bit_curr = (bool*)numa_alloc_local(max_nodes_per_package * sizeof(bool));
      Bit_next = (bool*)numa_alloc_local(max_nodes_per_package * sizeof(bool));
      PR_curr = (float*)numa_alloc_local(max_nodes_per_package * sizeof(float));
      PR_next = (float*)numa_alloc_local(max_nodes_per_package * sizeof(float));
      nodeList =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));
      edgeList = (unsigned int*)numa_alloc_local(max_edges_per_package *
                                                 sizeof(unsigned int));
      outDegree =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));
      inDegree =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));

      next_reset();
    }

    void mfree() {
      using namespace galois::runtime::MM;
      /*largeFree( (void *)Bit_curr, max_nodes_per_package * sizeof(bool));
      largeFree( (void *)Bit_next, max_nodes_per_package * sizeof(bool));
      largeFree( (void *)PR_curr, max_nodes_per_package * sizeof(float));
      largeFree( (void *)PR_next, max_nodes_per_package * sizeof(float));
      largeFree( (void *)nodeList, (nnodes + 1) * sizeof(int));
      largeFree( (void *)edgeList, max_edges_per_package * sizeof(int));*/
      /*std::free( (void *)Bit_curr);
      std::free( (void *)Bit_next);
      std::free( (void *)PR_curr);
      std::free( (void *)PR_next);
      std::free( (void *)nodeList);
      std::free( (void *)edgeList);*/
      numa_free((void*)Bit_curr, max_nodes_per_package * sizeof(bool));
      numa_free((void*)Bit_next, max_nodes_per_package * sizeof(bool));
      numa_free((void*)PR_curr, max_nodes_per_package * sizeof(float));
      numa_free((void*)PR_next, max_nodes_per_package * sizeof(float));
      numa_free((void*)nodeList, (nnodes + 1) * sizeof(unsigned int));
      numa_free((void*)edgeList, max_edges_per_package * sizeof(unsigned int));
      numa_free((void*)outDegree, (nnodes + 1) * sizeof(unsigned int));
      numa_free((void*)inDegree, (nnodes + 1) * sizeof(unsigned int));
    }

    void next_reset() {
      memset(PR_next, 0, max_nodes_per_package * sizeof(float));
      memset(Bit_next, 0, max_nodes_per_package * sizeof(bool));
      // printf(" %d %d %f %f\n",Bit_next[0], Bit_next[max_nodes_per_package -
      // 1], PR_next[0], PR_next[max_nodes_per_package - 1] );
    }
  };

  struct LNode {
    unsigned int outDegree;
    unsigned int inDegree;
    unsigned int id;
    float pagerank;
    float getPageRank() { return pagerank; }
    void setPageRank(float pr) { pagerank = pr; }
  };
  typedef galois::graphs::LC_InlineEdge_Graph<LNode, void>::with_no_lockable<
      true>::type ::with_numa_alloc<true>::type InnerGraph;

  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  /*typedef galois::graphs::LC_CSR_Graph<LNode,void>
    ::with_no_lockable<true>::type Graph;*/
  typedef Graph::GraphNode GNode;

  typedef galois::runtime::PerPackageStorage<LocalData> PerPackageData;

  typedef galois::runtime::Barrier Barrier;

  std::string name() const { return "polymer pull"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<float> sum_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::NONE);
      unsigned int outs =
          std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                        g.edge_end(n, galois::MethodFlag::NONE));
      data.outDegree = outs;
      unsigned int ins =
          std::distance(g.in_edge_begin(n, galois::MethodFlag::NONE),
                        g.in_edge_end(n, galois::MethodFlag::NONE));
      data.inDegree = ins;
      GNode start   = *g.begin();
      data.id       = g.idFromNode(n); // n - start;
    }
  };

  struct Copy {
    Graph& graph;
    PerPackageData& packageData;
    unsigned int* nodesPerPackage;
    unsigned int* nodesStartPackage;
    unsigned int nPackages;
    Copy(Graph& g, PerPackageData& pData, unsigned int* nodesP,
         unsigned int nPackages)
        : graph(g), packageData(pData), nodesPerPackage(nodesP),
          nPackages(nPackages) {
      nodesStartPackage            = new unsigned int[nPackages + 1];
      nodesStartPackage[0]         = 0;
      nodesStartPackage[nPackages] = graph.size();
      for (int i = 1; i < nPackages; i++) {
        nodesStartPackage[i] = nodesStartPackage[i - 1] + nodesPerPackage[i];
      }
    }
    void operator()(unsigned tid, unsigned numT) {
      if (galois::runtime::LL::isPackageLeader(tid)) {
        unsigned int currPackage =
            galois::runtime::LL::getPackageForThread(tid);
        for (auto ii = graph.begin(), end = graph.end(); ii != end; ii++) {
          GNode src    = *ii;
          LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);

          if (sdata.id >= nodesStartPackage[currPackage] &&
              sdata.id < nodesStartPackage[currPackage + 1])
            sdata.pagerank =
                packageData.getLocal()
                    ->PR_curr[sdata.id - nodesStartPackage[currPackage]];
        }

        packageData.getLocal()->mfree();
      }
    }
  };

  struct distributeEdges {
    Graph& graph;
    PerPackageData& packageData;
    unsigned int* nodesPerPackage;
    unsigned int* edgesPerPackage;

    distributeEdges(Graph& g, PerPackageData& p, unsigned int* nodes,
                    unsigned int* edges)
        : graph(g), packageData(p), nodesPerPackage(nodes),
          edgesPerPackage(edges){};

    void operator()(unsigned tid, unsigned numT) {
      if (galois::runtime::LL::isPackageLeader(tid)) {
        // printf("tid: %d\n", tid);
        packageData.getLocal()->alloc();

        unsigned int nnodes = graph.size();
        unsigned int currPackage =
            galois::runtime::LL::getPackageForThread(tid);
        unsigned int* outDegree = packageData.getLocal()->outDegree;
        unsigned int* inDegree  = packageData.getLocal()->inDegree;
        packageData.getLocal()->nodeList[nnodes] = edgesPerPackage[currPackage];
        // printf("tid: %d, edgesPerPackage: %d", tid,
        // edgesPerPackage[currPackage]);
        unsigned int rangeLow = 0;
        for (int i = 0; i < currPackage; i++) {
          rangeLow += nodesPerPackage[i];
        }
        unsigned int rangeHi = rangeLow + nodesPerPackage[currPackage];

        unsigned int edgeCount = 0;
        // std::cerr << "tid: " << tid <<"range:"<<rangeLow<<"-"<<rangeHi<<
        // std::endl; printf("work0.3 %d %d\n", rangeLow, rangeHi);
        unsigned int indegree_temp = 0;

        // std::cerr << "tid: " << tid <<"edgeCount:"<<edgeCount<< " inDegree
        // "<< indegree_temp<<std::endl;
        edgeCount = 0;
        for (auto ii = graph.begin(), end = graph.end(); ii != end; ii++) {
          GNode src    = *ii;
          LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
          // printf("%d %d\n", sdata.id, edgeCount);
          packageData.getLocal()->nodeList[sdata.id] = edgeCount;
          outDegree[sdata.id]                        = sdata.outDegree;
          inDegree[sdata.id]                         = sdata.inDegree;
          for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                    ej = graph.edge_end(src, galois::MethodFlag::NONE);
               jj != ej; ++jj) {
            GNode dst    = graph.getEdgeDst(jj);
            LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);

            if (ddata.id < rangeHi && ddata.id >= rangeLow) {
              packageData.getLocal()->edgeList[edgeCount] = ddata.id;
              edgeCount++;
            }
          }
        }
        // std::cerr << "tid1: " << tid << std::endl;
        // printf("work0.5\n");
        //===set node bit and PR==============
        for (unsigned i = 0; i < nodesPerPackage[currPackage]; i++) {
          packageData.getLocal()->PR_curr[i]  = 1.0;
          packageData.getLocal()->PR_next[i]  = 0.0;
          packageData.getLocal()->Bit_curr[i] = true;
          packageData.getLocal()->Bit_next[i] = false;
        }
      }
    }
  };

  struct PartitionInfo {
    unsigned int numThreads, nnodes, nPackages, coresPerPackage,
        coresLastPackage;
    unsigned* nodesPerPackage;
    unsigned* nodesStartPackage;
    unsigned* edgesPerPackage;
    unsigned* nodesPerCore;
    unsigned* nodesStartCore;
    unsigned* nodesPerCoreLastPkg;
    unsigned* nodesStartCoreLastPkg;
    PartitionInfo(unsigned nnodes, unsigned numThreads, unsigned nPackages,
                  unsigned coresPerPackage, unsigned coresLastPackage)
        : nnodes(nnodes), numThreads(numThreads), nPackages(nPackages),
          coresPerPackage(coresPerPackage), coresLastPackage(coresLastPackage) {
      edgesPerPackage   = new unsigned[nPackages];
      nodesPerPackage   = new unsigned[nPackages];
      nodesStartPackage = new unsigned[nPackages + 1];

      nodesPerCore   = new unsigned[coresPerPackage];
      nodesStartCore = new unsigned[coresPerPackage];

      nodesPerCoreLastPkg   = new unsigned[coresLastPackage];
      nodesStartCoreLastPkg = new unsigned[coresLastPackage];

      memset(edgesPerPackage, 0, sizeof(unsigned) * nPackages);
      memset(nodesPerPackage, 0, sizeof(unsigned) * nPackages);
      memset(nodesStartPackage, 0, sizeof(unsigned) * nPackages);

      memset(nodesPerCore, 0, sizeof(unsigned) * coresPerPackage);
      memset(nodesStartCore, 0, sizeof(unsigned) * coresPerPackage);
      memset(nodesPerCoreLastPkg, 0, sizeof(unsigned) * coresLastPackage);
      memset(nodesStartCoreLastPkg, 0, sizeof(unsigned) * coresLastPackage);
    }

    void partitionByDegree(Graph& graph, unsigned numThreads,
                           unsigned* nodesPerThread, unsigned* edgesPerThread) {
      int n = graph.size();
      // int *degrees = new int [n];

      // parallel_for(intT i = 0; i < n; i++) degrees[i] =
      // GA.V[i].getInDegree();}

      unsigned int* accum = new unsigned int[numThreads];
      for (int i = 0; i < numThreads; i++) {
        accum[i]          = 0;
        nodesPerThread[i] = 0;
      }

      unsigned int averageDegree = graph.sizeEdges() / numThreads;
      std::cout << "averageDegree is " << averageDegree << std::endl;
      int counter = 0;
      for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        GNode src    = *ii;
        LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
        accum[counter] += sdata.outDegree;
        nodesPerThread[counter]++;
        if ((accum[counter] >= averageDegree && counter < numThreads - 1) ||
            ii == ei - 1) {
          edgesPerThread[counter] = accum[counter];
          counter++;
        }
      }
      delete[] accum;
    }

    void subPartitionByDegree(Graph& graph, int nCores, unsigned* nodesPerCore,
                              unsigned* nodesStartCore) {
      unsigned n = graph.size();

      unsigned aveNodes = n / nCores;

      for (int i = 0; i < nCores - 1; i++) {
        nodesPerCore[i]   = aveNodes;
        nodesStartCore[i] = aveNodes * i;
      }
      nodesStartCore[nCores - 1] = aveNodes * (nCores - 1);
      nodesPerCore[nCores - 1]   = n - nodesStartCore[nCores - 1];
    }

    void partition(Graph& graph) {
      unsigned* nodesPerThread = new unsigned[numThreads];
      unsigned* edgesPerThread = new unsigned[numThreads];

      partitionByDegree(graph, numThreads, nodesPerThread, edgesPerThread);

      for (int i = 0; i < nPackages; i++) {
        int coresCurrPackage;
        if (i == nPackages - 1)
          coresCurrPackage = coresLastPackage;
        else
          coresCurrPackage = coresPerPackage;

        nodesPerPackage[i] = 0;
        edgesPerPackage[i] = 0;

        for (int j = 0; j < coresCurrPackage; j++) {
          nodesPerPackage[i] += nodesPerThread[coresPerPackage * i + j];
          edgesPerPackage[i] += edgesPerThread[coresPerPackage * i + j];
        }

        if (i > 0)
          nodesStartPackage[i] =
              nodesStartPackage[i - 1] + nodesPerPackage[i - 1];
      }
      nodesStartPackage[nPackages] =
          nodesStartPackage[nPackages - 1] + nodesPerPackage[nPackages - 1];

      subPartitionByDegree(graph, coresPerPackage, nodesPerCore,
                           nodesStartCore);
      subPartitionByDegree(graph, coresLastPackage, nodesPerCoreLastPkg,
                           nodesStartCoreLastPkg);

      // printf("nodesStartPackage: %d %d %d\n", nodesStartPackage[0],
      // nodesStartPackage[1], nodesStartPackage[2]);
      /*for(int i = 0; i < numThreads; i++)
      {
        printf("%d %d ", i, nodesPerThread[i]);
      }
      printf("\n");
      for(int i = 0; i < numThreads; i++)
      {
        printf("%d %d ", i, edgesPerThread[i]);
      }
      printf("\n");*/
      delete[] nodesPerThread;
      delete[] edgesPerThread;
    }

    void mfree() {
      delete[] edgesPerPackage;
      delete[] nodesPerPackage;
      delete[] nodesStartPackage;

      delete[] nodesPerCore;
      delete[] nodesStartCore;

      delete[] nodesPerCoreLastPkg;
      delete[] nodesStartCoreLastPkg;
    }

    void print() {
      printf("ePerPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, edgesPerPackage[i]);
        printf("\n");
      }
      printf("nPerPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, nodesPerPackage[i]);
        printf("\n");
      }
      printf("nstartPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, nodesStartPackage[i]);
        printf("\n");
      }
      printf("nPerCore: ");
      for (int i = 0; i < coresPerPackage; i++) {

        printf("%d %d ", i, nodesPerCore[i]);
        printf("\n");
      }
      printf("nStartCore: ");
      for (int i = 0; i < coresPerPackage; i++) {

        printf("%d %d ", i, nodesStartCore[i]);
        printf("\n");
      }
      printf("nPerCoreLastPkg: ");
      for (int i = 0; i < coresLastPackage; i++) {

        printf("%d %d ", i, nodesPerCoreLastPkg[i]);
        printf("\n");
      }
      printf("nPerCorePkg: ");
      for (int i = 0; i < coresLastPackage; i++) {

        printf("%d %d", i, nodesStartCoreLastPkg[i]);
        printf("\n");
      }
    }
  };

  struct Process {
    PolyPull* self;
    PerPackageData& packageData;
    PartitionInfo& partitionInfo;
    Barrier& gbarrier;
    Process(PerPackageData& p, PartitionInfo& partitionInfo, Barrier& gbarrier,
            PolyPull* self)
        : packageData(p), partitionInfo(partitionInfo), gbarrier(gbarrier),
          self(self) {}
    void update(float* PR_next, unsigned offset, float value) {
      writeAdd(PR_next + offset, value);
      // PR_next[offset] += value;
    }
    void operator()(unsigned tid, unsigned numT) {
      unsigned leader      = galois::runtime::LL::getLeaderForThread(tid);
      unsigned currPackage = galois::runtime::LL::getPackageForThread(tid);

      unsigned nPackages = partitionInfo.nPackages;

      unsigned threadStart =
          (currPackage == nPackages - 1)
              ? partitionInfo.nodesStartCoreLastPkg[tid - leader]
              : partitionInfo.nodesStartCore[tid - leader];
      unsigned nodesCurrCore =
          (currPackage == nPackages - 1)
              ? partitionInfo.nodesPerCoreLastPkg[tid - leader]
              : partitionInfo.nodesPerCore[tid - leader];

      unsigned threadEnd = threadStart + nodesCurrCore;

      unsigned activePackage = 0;
      bool *Bit_curr, *Bit_next;
      unsigned *edgelist, *nodelist, *outDegree, *inDegree;
      float *PR_curr, *PR_next;
      unsigned startOffset, packageOffset, nout;
      unsigned firstEdge, src, localnout;
      float sum, delta;

      unsigned* nodesStartPackage = partitionInfo.nodesStartPackage;
      for (int i = 0; i < nPackages; i++) {
        if (threadStart >= nodesStartPackage[i] &&
            threadStart < nodesStartPackage[i + 1]) {
          activePackage = i;
          startOffset   = nodesStartPackage[i];
          break;
        }
      }
      nodelist  = packageData.getLocal()->nodeList;
      edgelist  = packageData.getLocal()->edgeList;
      outDegree = packageData.getLocal()->outDegree;
      inDegree  = packageData.getLocal()->inDegree;
      Bit_curr  = packageData.getLocal()->Bit_curr;
      PR_curr   = packageData.getLocal()->PR_curr;

      PR_next  = packageData.getRemoteByPkg(activePackage)->PR_next;
      Bit_next = packageData.getRemoteByPkg(activePackage)->Bit_next;

      gbarrier.wait();
      // printf("tid: %d start: %d end: %d \n", tid, threadStart, threadEnd);
      // gbarrier.wait();

      for (unsigned i = threadStart; i < threadEnd; i++) {
        packageOffset = i - startOffset;
        if (i == nodesStartPackage[activePackage + 1]) {
          activePackage++;
          startOffset   = nodesStartPackage[activePackage];
          packageOffset = 0; // i - startOffset;
          if (currPackage == activePackage) {
            Bit_next = packageData.getLocal()->Bit_next;
            PR_next  = packageData.getLocal()->PR_next;
          } else {
            Bit_next = packageData.getRemoteByPkg(activePackage)->Bit_next;
            PR_next  = packageData.getRemoteByPkg(activePackage)->PR_next;
          }
        }

        if (nodelist[i] != nodelist[i + 1]) {
          sum = 0;
          for (unsigned j = nodelist[i]; j < nodelist[i + 1]; j++) {
            src = edgelist[j];
            {
              nout            = outDegree[src];
              unsigned offset = src - nodesStartPackage[currPackage];
              delta           = PR_curr[offset] / (float)nout;

              if (delta >= tolerance && Bit_curr[offset]) {
                sum += delta;
                Bit_next[packageOffset] = true;
              }
            }
          }
          update(PR_next, packageOffset, sum);
        }
      }
    }
  };

  struct Process2 {
    PolyPull* self;
    PerPackageData& packageData;
    PartitionInfo& partitionInfo;
    Barrier& gbarrier;
    Process2(PerPackageData& p, PartitionInfo& partitionInfo, Barrier& gbarrier,
             PolyPull* self)
        : packageData(p), partitionInfo(partitionInfo), gbarrier(gbarrier),
          self(self) {}

    void damping(float* PR_next, unsigned tid, PolyPull* self,
                 PerPackageData& packageData) {
      unsigned localtid = tid - galois::runtime::LL::getLeaderForThread(tid);
      unsigned coresCurrPackage;
      unsigned nodesPerCore;
      unsigned currPackage = galois::runtime::LL::getPackageForThread(tid);
      if (currPackage != partitionInfo.nPackages - 1) {
        coresCurrPackage = partitionInfo.coresPerPackage;
      } else {
        coresCurrPackage = partitionInfo.coresLastPackage;
      }

      nodesPerCore =
          partitionInfo.nodesPerPackage[currPackage] / coresCurrPackage;
      float diff;
      if (localtid != coresCurrPackage - 1) {
        for (unsigned i = localtid * nodesPerCore;
             i < (localtid + 1) * nodesPerCore; i++) {
          PR_next[i] = alpha2 * PR_next[i] + 1 - alpha2;
          /* diff = std::fabs(packageData.getLocal()->PR_curr[i] - PR_next[i]);
           self->max_delta.update(diff);
           self->sum_delta.update(diff);*/
        }
      } else {
        for (unsigned i = localtid * nodesPerCore;
             i < partitionInfo.nodesPerPackage[currPackage]; i++) {
          PR_next[i] = alpha2 * PR_next[i] + 1 - alpha2;
          /*diff = std::fabs(packageData.getLocal()->PR_curr[i] - PR_next[i]);
          self->max_delta.update(diff);
          self->sum_delta.update(diff);*/
        }
      }
    }

    void operator()(unsigned tid, unsigned numT) {
      float* PR_next = packageData.getLocal()->PR_next;
      damping(PR_next, tid, self, packageData);
      gbarrier.wait();
      if (galois::runtime::LL::isPackageLeader(tid)) {
        float* temp;
        temp                            = packageData.getLocal()->PR_curr;
        packageData.getLocal()->PR_curr = packageData.getLocal()->PR_next;
        packageData.getLocal()->PR_next = temp;

        bool* tempBit;
        tempBit                          = packageData.getLocal()->Bit_curr;
        packageData.getLocal()->Bit_curr = packageData.getLocal()->Bit_next;
        packageData.getLocal()->Bit_next = tempBit;

        packageData.getLocal()->next_reset();
      }
    }
  };

  void operator()(Graph& graph) {
    // nPackages = LL::getMaxPackages();
    unsigned coresPerPackage = galois::runtime::LL::getMaxCores() /
                               galois::runtime::LL::getMaxPackages();

    unsigned nPackages        = (numThreads - 1) / coresPerPackage + 1;
    unsigned coresLastPackage = numThreads % coresPerPackage;
    if (coresLastPackage == 0)
      coresLastPackage = coresPerPackage;
    Barrier& gbarrier = galois::runtime::getSystemBarrier();

    unsigned nnodes = graph.size();
    std::cout << "nnodes:" << nnodes << " nedges:" << graph.sizeEdges()
              << std::endl;

    PartitionInfo partitionInfo(nnodes, numThreads, nPackages, coresPerPackage,
                                coresLastPackage);

    partitionInfo.partition(graph);

    unsigned max_nodes_per_package = 0, max_edges_per_package = 0;
    unsigned* nodesPerPackage = partitionInfo.nodesPerPackage;
    unsigned* edgesPerPackage = partitionInfo.edgesPerPackage;
    for (unsigned i = 0; i < nPackages; i++) {
      max_nodes_per_package = (max_nodes_per_package > nodesPerPackage[i])
                                  ? max_nodes_per_package
                                  : nodesPerPackage[i];
      max_edges_per_package = (max_edges_per_package > edgesPerPackage[i])
                                  ? max_edges_per_package
                                  : edgesPerPackage[i];
    }

    PerPackageData packageData(nnodes, max_nodes_per_package,
                               max_edges_per_package);

    galois::on_each(
        distributeEdges(graph, packageData, nodesPerPackage, edgesPerPackage));
    unsigned int iteration = 0;
    galois::StatTimer T("pure time");

    // partitionInfo.print();
    T.start();
    while (true) {
      galois::on_each(Process(packageData, partitionInfo, gbarrier, this));
      galois::on_each(Process2(packageData, partitionInfo, gbarrier, this));
      iteration += 1;

      float delta = max_delta.reduce();

      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta.reduce()
                << " max delta: " << delta << "\n";
      // delta = 1;
      if (iteration >= maxIterations)
        break;

      max_delta.reset();
      sum_delta.reset();
      // galois::on_each(Copy(graph, packageData, nodesPerPackage,nPackages));
    }
    T.stop();
    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }
    galois::on_each(Copy(graph, packageData, nodesPerPackage, nPackages));

    partitionInfo.mfree();

    /*if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }*/
  }
};

#if (0)
/*struct PolyPull2 {

  struct LocalData {

    unsigned int nnodes;
    unsigned int max_nodes_per_package;
    unsigned int max_edges_per_package;

    bool *Bit_curr;
    bool *Bit_next;
    float *PR_curr;
    float *PR_next;
    unsigned int *nodeList;
    unsigned int *edgeList;
    unsigned int *outDegree;
    unsigned int *inDegree;
    LocalData(unsigned int nnodes1, unsigned int max_nodes_per_package1,
  unsigned int max_edges_per_package1):
      nnodes(nnodes1),max_nodes_per_package(max_nodes_per_package1),max_edges_per_package(max_edges_per_package1)
  {};

    void alloc() {
      using namespace galois::runtime::MM;
      /*Bit_curr = (bool * ) largeAlloc( max_nodes_per_package * sizeof(bool));
      Bit_next = (bool * ) largeAlloc( max_nodes_per_package * sizeof(bool));
      PR_curr = (float * ) largeAlloc( max_nodes_per_package * sizeof(float));
      PR_next = (float * ) largeAlloc( max_nodes_per_package * sizeof(float));
      nodeList = (int * ) largeAlloc( (nnodes + 1) * sizeof(int));
      edgeList = (int * ) largeAlloc( max_edges_per_package * sizeof(int));*/
/*Bit_curr = (bool * ) std::malloc( max_nodes_per_package * sizeof(bool));
Bit_next = (bool * ) std::malloc( max_nodes_per_package * sizeof(bool));
PR_curr = (float * ) std::malloc( max_nodes_per_package * sizeof(float));
PR_next = (float * ) std::malloc( max_nodes_per_package * sizeof(float));
nodeList = (int * ) std::malloc( (nnodes + 1) * sizeof(int));
edgeList = (int * ) std::malloc( max_edges_per_package * sizeof(int));*/
/*Bit_curr = (bool * ) numa_alloc_local( max_nodes_per_package * sizeof(bool));
Bit_next = (bool * ) numa_alloc_local( max_nodes_per_package * sizeof(bool));
PR_curr = (float * ) numa_alloc_local( max_nodes_per_package * sizeof(float));
PR_next = (float * ) numa_alloc_local( max_nodes_per_package * sizeof(float));
nodeList = (unsigned int * ) numa_alloc_local( (nnodes + 1) * sizeof(unsigned
int)); edgeList = (unsigned int * ) numa_alloc_local( max_edges_per_package *
sizeof(unsigned int)); outDegree = (unsigned int * ) numa_alloc_local( (nnodes +
1) * sizeof(unsigned int)); inDegree = (unsigned int * ) numa_alloc_local(
(nnodes + 1) * sizeof(unsigned int));

next_reset();
}

void mfree() {
using namespace galois::runtime::MM;
/*largeFree( (void *)Bit_curr, max_nodes_per_package * sizeof(bool));
largeFree( (void *)Bit_next, max_nodes_per_package * sizeof(bool));
largeFree( (void *)PR_curr, max_nodes_per_package * sizeof(float));
largeFree( (void *)PR_next, max_nodes_per_package * sizeof(float));
largeFree( (void *)nodeList, (nnodes + 1) * sizeof(int));
largeFree( (void *)edgeList, max_edges_per_package * sizeof(int));*/
/*std::free( (void *)Bit_curr);
std::free( (void *)Bit_next);
std::free( (void *)PR_curr);
std::free( (void *)PR_next);
std::free( (void *)nodeList);
std::free( (void *)edgeList);*/
/*numa_free( (void *)Bit_curr, max_nodes_per_package * sizeof(bool));
numa_free( (void *)Bit_next, max_nodes_per_package * sizeof(bool));
numa_free( (void *)PR_curr, max_nodes_per_package * sizeof(float));
numa_free( (void *)PR_next, max_nodes_per_package * sizeof(float));
numa_free( (void *)nodeList, (nnodes + 1) * sizeof(unsigned int ));
numa_free( (void *)edgeList, max_edges_per_package * sizeof(unsigned int));
numa_free( (void *)outDegree, (nnodes + 1) * sizeof(unsigned int ));
numa_free( (void *)inDegree, (nnodes + 1) * sizeof(unsigned int ));
}

void next_reset(){
memset(PR_next, 0, max_nodes_per_package * sizeof(float));
memset(Bit_next, 0, max_nodes_per_package * sizeof(bool));
//printf(" %d %d %f %f\n",Bit_next[0], Bit_next[max_nodes_per_package - 1],
PR_next[0], PR_next[max_nodes_per_package - 1] );
}
};


struct LNode {
unsigned int outDegree;
unsigned int inDegree;
unsigned int id;
float pagerank;
float getPageRank() {return pagerank; }
void setPageRank(float pr) { pagerank = pr; }
};
typedef galois::graphs::LC_InlineEdge_Graph<LNode,void>
::with_no_lockable<true>::type
::with_numa_alloc<true>::type
InnerGraph;

typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
/*typedef galois::graphs::LC_CSR_Graph<LNode,void>
::with_no_lockable<true>::type Graph;*/
typedef Graph::GraphNode GNode;

typedef galois::runtime::PerPackageStorage<LocalData> PerPackageData;

typedef galois::runtime::Barrier Barrier;

std::string name() const { return "polymer pull"; }

galois::GReduceMax<float> max_delta;
galois::GAccumulator<float> sum_delta;

void readGraph(Graph& graph) {
  if (transposeGraphName.size()) {
    galois::graphs::readGraph(graph, transposeGraphName);
  } else {
    std::cerr
        << "Need to pass precomputed graph through -graphTranspose option\n";
    abort();
  }
}

struct Initialize {
  Graph& g;
  Initialize(Graph& g) : g(g) {}
  void operator()(Graph::GraphNode n) const {
    LNode& data       = g.getData(n, galois::MethodFlag::NONE);
    unsigned int outs = std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                                      g.edge_end(n, galois::MethodFlag::NONE));
    data.outDegree    = outs;
    unsigned int ins =
        std::distance(g.in_edge_begin(n, galois::MethodFlag::NONE),
                      g.in_edge_end(n, galois::MethodFlag::NONE));
    data.inDegree = ins;
    GNode start   = *g.begin();
    data.id       = g.idFromNode(n); // n - start;
  }
};

struct Copy {
  Graph& graph;
  PerPackageData& packageData;
  unsigned int* nodesPerPackage;
  unsigned int* nodesStartPackage;
  unsigned int nPackages;
  Copy(Graph& g, PerPackageData& pData, unsigned int* nodesP,
       unsigned int nPackages)
      : graph(g), packageData(pData), nodesPerPackage(nodesP),
        nPackages(nPackages) {
    nodesStartPackage            = new unsigned int[nPackages + 1];
    nodesStartPackage[0]         = 0;
    nodesStartPackage[nPackages] = graph.size();
    for (int i = 1; i < nPackages; i++) {
      nodesStartPackage[i] = nodesStartPackage[i - 1] + nodesPerPackage[i];
    }
  }
  void operator()(unsigned tid, unsigned numT) {
    if (galois::runtime::LL::isPackageLeader(tid)) {
      unsigned int currPackage = galois::runtime::LL::getPackageForThread(tid);
      for (auto ii = graph.begin(), end = graph.end(); ii != end; ii++) {
        GNode src    = *ii;
        LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);

        if (sdata.id >= nodesStartPackage[currPackage] &&
            sdata.id < nodesStartPackage[currPackage + 1])
          sdata.pagerank =
              packageData.getLocal()
                  ->PR_curr[sdata.id - nodesStartPackage[currPackage]];
      }

      packageData.getLocal()->mfree();
    }
  }
};

struct distributeEdges {
  Graph& graph;
  PerPackageData& packageData;
  unsigned int* nodesPerPackage;
  unsigned int* edgesPerPackage;
  unsigned superPartition;

  distributeEdges(Graph& g, PerPackageData& p, unsigned int* nodes,
                  unsigned int* edges, unsigned superPartition)
      : graph(g), packageData(p), nodesPerPackage(nodes),
        edgesPerPackage(edges), superPartition(superPartition){};

  void operator()(unsigned tid, unsigned numT) {
    if (galois::runtime::LL::isPackageLeader(tid)) {
      // printf("tid: %d\n", tid);
      packageData.getLocal()->alloc();

      unsigned int nnodes      = graph.size();
      unsigned int currPackage = galois::runtime::LL::getPackageForThread(tid);
      unsigned int* outDegree  = packageData.getLocal()->outDegree;
      unsigned int* inDegree   = packageData.getLocal()->inDegree;
      packageData.getLocal()->nodeList[nnodes] = edgesPerPackage[currPackage];
      // printf("tid: %d, edgesPerPackage: %d", tid,
      // edgesPerPackage[currPackage]);
      unsigned int rangeLow = 0;
      for (int i = 0; i < currPackage * superPartition; i++) {
        rangeLow += nodesPerPackage[i];
      }
      unsigned int rangeHi =
          rangeLow + nodesPerPackage[currPackage * superPartition];

      unsigned int edgeCount = 0;
      // std::cerr << "tid: " << tid <<"range:"<<rangeLow<<"-"<<rangeHi<<
      // std::endl; printf("work0.3 %d %d\n", rangeLow, rangeHi);

      // std::cerr << "tid: " << tid <<"edgeCount:"<<edgeCount<< " inDegree "<<
      // indegree_temp<<std::endl;
      for (int i = 0; i < superPartition; i++) {
        edgeCount = 0;
        if (i != 0) {
          rangeLow = rangeHi;
          rangeHi = rangeLow + nodesPerPackage[currPackage * superPartition + i]
        }
        for (auto ii = graph.begin(), end = graph.end(); ii != end; ii++) {
          GNode src    = *ii;
          LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
          // printf("%d %d\n", sdata.id, edgeCount);
          packageData.getLocal()->nodeList[sdata.id] = edgeCount;
          outDegree[sdata.id]                        = sdata.outDegree;
          inDegree[sdata.id]                         = sdata.inDegree;
          for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                    ej = graph.edge_end(src, galois::MethodFlag::NONE);
               jj != ej; ++jj) {
            GNode dst    = graph.getEdgeDst(jj);
            LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);

            if (ddata.id < rangeHi && ddata.id >= rangeLow) {
              packageData.getLocal()->edgeList[edgeCount] = ddata.id;
              edgeCount++;
            }
          }
        }
      }
      // std::cerr << "tid1: " << tid << std::endl;
      // printf("work0.5\n");
      //===set node bit and PR==============
      for (unsigned i = 0; i < nodesPerPackage[currPackage]; i++) {
        packageData.getLocal()->PR_curr[i]  = 1.0;
        packageData.getLocal()->PR_next[i]  = 0.0;
        packageData.getLocal()->Bit_curr[i] = true;
        packageData.getLocal()->Bit_next[i] = false;
      }
    }
  }
};

struct PartitionInfo {
  unsigned int numThreads, nnodes, nPackages, coresPerPackage, coresLastPackage,
      superPartition;
  unsigned* nodesPerPackage;
  unsigned* nodesStartPackage;
  unsigned* edgesPerPackage;
  unsigned* nodesPerCore;
  unsigned* nodesStartCore;
  unsigned* nodesPerCoreLastPkg;
  unsigned* nodesStartCoreLastPkg;
  PartitionInfo(unsigned nnodes, unsigned numThreads, unsigned nPackages,
                unsigned coresPerPackage, unsigned coresLastPackage,
                unsigned superPartition)
      : nnodes(nnodes), numThreads(numThreads), nPackages(nPackages),
        coresPerPackage(coresPerPackage), coresLastPackage(coresLastPackage),
        superPartition(superPartition) {
    edgesPerPackage   = new unsigned[nPackages * superPartition];
    nodesPerPackage   = new unsigned[nPackages * superPartition];
    nodesStartPackage = new unsigned[nPackages * superPartition + 1];

    nodesPerCore   = new unsigned[coresPerPackage];
    nodesStartCore = new unsigned[coresPerPackage];

    nodesPerCoreLastPkg   = new unsigned[coresLastPackage];
    nodesStartCoreLastPkg = new unsigned[coresLastPackage];

    memset(edgesPerPackage, 0, sizeof(unsigned) * nPackages * superPartition);
    memset(nodesPerPackage, 0, sizeof(unsigned) * nPackages * superPartition);
            memset(nodesStartPackage, 0, sizeof(unsigned) * (nPackages * superPartition + 1);

            memset(nodesPerCore, 0, sizeof(unsigned) * coresPerPackage);
            memset(nodesStartCore, 0, sizeof(unsigned) * coresPerPackage);
            memset(nodesPerCoreLastPkg, 0, sizeof(unsigned) * coresLastPackage);
            memset(nodesStartCoreLastPkg, 0, sizeof(unsigned) * coresLastPackage);
  }

  void partitionByDegree(Graph& graph, unsigned numThreads,
                         unsigned* nodesPerThread, unsigned* edgesPerThread) {
    int n = graph.size();
    // int *degrees = new int [n];

    // parallel_for(intT i = 0; i < n; i++) degrees[i] = GA.V[i].getInDegree();}

    unsigned int* accum = new unsigned int[numThreads];
    for (int i = 0; i < numThreads; i++) {
      accum[i]          = 0;
      nodesPerThread[i] = 0;
    }

    unsigned int averageDegree = graph.sizeEdges() / numThreads;
    std::cout << "averageDegree is " << averageDegree << std::endl;
    int counter = 0;
    for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      GNode src    = *ii;
      LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
      accum[counter] += sdata.outDegree;
      nodesPerThread[counter]++;
      if ((accum[counter] >= averageDegree && counter < numThreads - 1) ||
          ii == ei - 1) {
        edgesPerThread[counter] = accum[counter];
        counter++;
      }
    }
    delete[] accum;
  }

  void subPartitionByDegree(Graph& graph, int nCores, unsigned* nodesPerCore,
                            unsigned* nodesStartCore) {
    unsigned n = graph.size();

    unsigned aveNodes = n / nCores;

    for (int i = 0; i < nCores - 1; i++) {
      nodesPerCore[i]   = aveNodes;
      nodesStartCore[i] = aveNodes * i;
    }
    nodesStartCore[nCores - 1] = aveNodes * (nCores - 1);
    nodesPerCore[nCores - 1]   = n - nodesStartCore[nCores - 1];
  }

  void partition(Graph& graph) {
    unsigned* nodesPerThread = new unsigned[numThreads];
    unsigned* edgesPerThread = new unsigned[numThreads];

    partitionByDegree(graph, numThreads, nodesPerThread, edgesPerThread);

    for (int i = 0; i < nPackages * superPartition; i += superPartition) {
      int coresCurrPackage;
      if (i == nPackages - 1)
        coresCurrPackage = coresLastPackage;
      else
        coresCurrPackage = coresPerPackage;

      nodesPerPackage[i] = 0;
      edgesPerPackage[i] = 0;

      for (int j = 0; j < coresCurrPackage; j++) {
        nodesPerPackage[i] += nodesPerThread[coresPerPackage * i + j];
        edgesPerPackage[i] += edgesPerThread[coresPerPackage * i + j];
      }

      if (i > 0)
        nodesStartPackage[i] =
            nodesStartPackage[i - 1] + nodesPerPackage[i - 1];

      for (int x = 1; x < superPartition; x++) {
        nodesPerPackage[i + x] = nodesPerPackage[i] / superPartition;
        edgesPerPackage[i + x] = edgesPerPackage[i] / superPartition;
        nodesStartPackage[i + x] =
            nodesStartPackage[i] + x * nodesPerPackage[i] / superPartition;
      }
      nodesPerPackage[i] = nodesPerPackage[i] / superPartition;
      edgesPerPackage[i] = edgesPerPackage[i] / superPartition;
    }
    nodesStartPackage[nPackages * superPartition] = nnodes;

    subPartitionByDegree(graph, coresPerPackage, nodesPerCore, nodesStartCore);
    subPartitionByDegree(graph, coresLastPackage, nodesPerCoreLastPkg,
                         nodesStartCoreLastPkg);

    delete[] nodesPerThread;
    delete[] edgesPerThread;
  }

  void mfree() {
    delete[] edgesPerPackage;
    delete[] nodesPerPackage;
    delete[] nodesStartPackage;

    delete[] nodesPerCore;
    delete[] nodesStartCore;

    delete[] nodesPerCoreLastPkg;
    delete[] nodesStartCoreLastPkg;
  }

  void print() {
    printf("ePerPkg: ");
    for (int i = 0; i < nPackages; i++) {

      printf("%d %d ", i, edgesPerPackage[i]);
      printf("\n");
    }
    printf("nPerPkg: ");
    for (int i = 0; i < nPackages; i++) {

      printf("%d %d ", i, nodesPerPackage[i]);
      printf("\n");
    }
    printf("nstartPkg: ");
    for (int i = 0; i < nPackages; i++) {

      printf("%d %d ", i, nodesStartPackage[i]);
      printf("\n");
    }
    printf("nPerCore: ");
    for (int i = 0; i < coresPerPackage; i++) {

      printf("%d %d ", i, nodesPerCore[i]);
      printf("\n");
    }
    printf("nStartCore: ");
    for (int i = 0; i < coresPerPackage; i++) {

      printf("%d %d ", i, nodesStartCore[i]);
      printf("\n");
    }
    printf("nPerCoreLastPkg: ");
    for (int i = 0; i < coresLastPackage; i++) {

      printf("%d %d ", i, nodesPerCoreLastPkg[i]);
      printf("\n");
    }
    printf("nPerCorePkg: ");
    for (int i = 0; i < coresLastPackage; i++) {

      printf("%d %d", i, nodesStartCoreLastPkg[i]);
      printf("\n");
    }
  }
};

struct Process {
  PolyPull2* self;
  PerPackageData& packageData;
  PartitionInfo& partitionInfo;
  Barrier& gbarrier;
  Process(PerPackageData& p, PartitionInfo& partitionInfo, Barrier& gbarrier,
          PolyPull2* self)
      : packageData(p), partitionInfo(partitionInfo), gbarrier(gbarrier),
        self(self) {}
  void update(float* PR_next, unsigned offset, float value) {
    writeAdd(PR_next + offset, value);
    // PR_next[offset] += value;
  }
  void operator()(unsigned tid, unsigned numT) {
    unsigned leader      = galois::runtime::LL::getLeaderForThread(tid);
    unsigned currPackage = galois::runtime::LL::getPackageForThread(tid);

    unsigned nPackages = partitionInfo.nPackages;

    unsigned threadStart =
        (currPackage == nPackages - 1)
            ? partitionInfo.nodesStartCoreLastPkg[tid - leader]
            : partitionInfo.nodesStartCore[tid - leader];
    unsigned nodesCurrCore =
        (currPackage == nPackages - 1)
            ? partitionInfo.nodesPerCoreLastPkg[tid - leader]
            : partitionInfo.nodesPerCore[tid - leader];

    unsigned threadEnd = threadStart + nodesCurrCore;

    unsigned activePackage = 0;
    bool *Bit_curr, *Bit_next;
    unsigned *edgelist, *nodelist, *outDegree, *inDegree;
    float *PR_curr, *PR_next;
    unsigned startOffset, packageOffset, nout;
    unsigned firstEdge, src, localnout;
    float sum, delta;

    unsigned* nodesStartPackage = partitionInfo.nodesStartPackage;
    for (int i = 0; i < nPackages; i++) {
      if (threadStart >= nodesStartPackage[i] &&
          threadStart < nodesStartPackage[i + 1]) {
        activePackage = i;
        startOffset   = nodesStartPackage[i];
        break;
      }
    }

    nodelist  = packageData.getLocal()->nodeList;
    edgelist  = packageData.getLocal()->edgeList;
    outDegree = packageData.getLocal()->outDegree;
    inDegree  = packageData.getLocal()->inDegree;
    Bit_curr  = packageData.getLocal()->Bit_curr;
    PR_curr   = packageData.getLocal()->PR_curr;

    PR_next  = packageData.getRemoteByPkg(activePackage)->PR_next;
    Bit_next = packageData.getRemoteByPkg(activePackage)->Bit_next;

    gbarrier.wait();
    // printf("tid: %d start: %d end: %d \n", tid, threadStart, threadEnd);
    // gbarrier.wait();

    for (unsigned i = threadStart; i < threadEnd; i++) {
      packageOffset = i - startOffset;
      if (i >= nodesStartPackage[activePackage + 1]) {
        activePackage++;
        startOffset   = nodesStartPackage[activePackage];
        packageOffset = 0; // i - startOffset;
        if (currPackage == activePackage) {
          Bit_next = packageData.getLocal()->Bit_next;
          PR_next  = packageData.getLocal()->PR_next;
        } else {
          Bit_next = packageData.getRemoteByPkg(activePackage)->Bit_next;
          PR_next  = packageData.getRemoteByPkg(activePackage)->PR_next;
        }
      }

      if (nodelist[i] != nodelist[i + 1]) {
        sum = 0;
        for (unsigned j = nodelist[i]; j < nodelist[i + 1]; j++) {
          src = edgelist[j];
          {
            nout            = outDegree[src];
            unsigned offset = src - nodesStartPackage[currPackage];
            delta           = PR_curr[offset] / (float)nout;

            if (delta >= tolerance && Bit_curr[offset]) {
              sum += delta;
              Bit_next[packageOffset] = true;
            }
          }
        }
        update(PR_next, packageOffset, sum);
      }
    }
  }
};

struct Process2 {
  PolyPull2* self;
  PerPackageData& packageData;
  PartitionInfo& partitionInfo;
  Barrier& gbarrier;
  Process2(PerPackageData& p, PartitionInfo& partitionInfo, Barrier& gbarrier,
           PolyPull2* self)
      : packageData(p), partitionInfo(partitionInfo), gbarrier(gbarrier),
        self(self) {}

  void damping(float* PR_next, unsigned tid, PolyPull2* self,
               PerPackageData& packageData) {
    unsigned localtid = tid - galois::runtime::LL::getLeaderForThread(tid);
    unsigned coresCurrPackage;
    unsigned nodesPerCore;
    unsigned currPackage = galois::runtime::LL::getPackageForThread(tid);
    if (currPackage != partitionInfo.nPackages - 1) {
      coresCurrPackage = partitionInfo.coresPerPackage;
    } else {
      coresCurrPackage = partitionInfo.coresLastPackage;
    }

    nodesPerCore =
        partitionInfo.nodesPerPackage[currPackage] / coresCurrPackage;
    float diff;
    if (localtid != coresCurrPackage - 1) {
      for (unsigned i = localtid * nodesPerCore;
           i < (localtid + 1) * nodesPerCore; i++) {
        PR_next[i] = alpha2 * PR_next[i] + 1 - alpha2;
        /* diff = std::fabs(packageData.getLocal()->PR_curr[i] - PR_next[i]);
         self->max_delta.update(diff);
         self->sum_delta.update(diff);*/
      }
    } else {
      for (unsigned i = localtid * nodesPerCore;
           i < partitionInfo.nodesPerPackage[currPackage]; i++) {
        PR_next[i] = alpha2 * PR_next[i] + 1 - alpha2;
        /*diff = std::fabs(packageData.getLocal()->PR_curr[i] - PR_next[i]);
        self->max_delta.update(diff);
        self->sum_delta.update(diff);*/
      }
    }
  }

  void operator()(unsigned tid, unsigned numT) {
    float* PR_next = packageData.getLocal()->PR_next;
    damping(PR_next, tid, self, packageData);
    gbarrier.wait();
    if (galois::runtime::LL::isPackageLeader(tid)) {
      float* temp;
      temp                            = packageData.getLocal()->PR_curr;
      packageData.getLocal()->PR_curr = packageData.getLocal()->PR_next;
      packageData.getLocal()->PR_next = temp;

      bool* tempBit;
      tempBit                          = packageData.getLocal()->Bit_curr;
      packageData.getLocal()->Bit_curr = packageData.getLocal()->Bit_next;
      packageData.getLocal()->Bit_next = tempBit;

      packageData.getLocal()->next_reset();
    }
  }
};

void operator()(Graph& graph) {
  // nPackages = LL::getMaxPackages();
  unsigned coresPerPackage = galois::runtime::LL::getMaxCores() /
                             galois::runtime::LL::getMaxPackages();
  unsigned superPartition   = 2;
  unsigned nPackages        = (numThreads - 1) / coresPerPackage + 1;
  unsigned coresLastPackage = numThreads % coresPerPackage;
  if (coresLastPackage == 0)
    coresLastPackage = coresPerPackage;
  Barrier& gbarrier = galois::runtime::getSystemBarrier();

  unsigned nnodes = graph.size();
  std::cout << "nnodes:" << nnodes << " nedges:" << graph.sizeEdges()
            << std::endl;

  PartitionInfo partitionInfo(nnodes, numThreads, nPackages, coresPerPackage,
                              coresLastPackage);

  partitionInfo.partition(graph);

  unsigned max_nodes_per_package = 0, max_edges_per_package = 0;
  unsigned* nodesPerPackage = partitionInfo.nodesPerPackage;
  unsigned* edgesPerPackage = partitionInfo.edgesPerPackage;
  for (unsigned i = 0; i < nPackages; i++) {
    max_nodes_per_package = (max_nodes_per_package > nodesPerPackage[i])
                                ? max_nodes_per_package
                                : nodesPerPackage[i];
    max_edges_per_package = (max_edges_per_package > edgesPerPackage[i])
                                ? max_edges_per_package
                                : edgesPerPackage[i];
  }

  PerPackageData packageData(nnodes, max_nodes_per_package,
                             max_edges_per_package);

  galois::on_each(
      distributeEdges(graph, packageData, nodesPerPackage, edgesPerPackage));
  unsigned int iteration = 0;
  galois::StatTimer T("pure time");

  // partitionInfo.print();
  T.start();
  while (true) {
    galois::on_each(Process(packageData, partitionInfo, gbarrier, this));
    galois::on_each(Process2(packageData, partitionInfo, gbarrier, this));
    iteration += 1;

    float delta = max_delta.reduce();

    std::cout << "iteration: " << iteration
              << " sum delta: " << sum_delta.reduce() << " max delta: " << delta
              << "\n";
    // delta = 1;
    if (iteration >= maxIterations)
      break;

    max_delta.reset();
    sum_delta.reset();
    // galois::on_each(Copy(graph, packageData, nodesPerPackage,nPackages));
  }
  T.stop();
  if (iteration >= maxIterations) {
    std::cout << "Failed to converge\n";
  }
  galois::on_each(Copy(graph, packageData, nodesPerPackage, nPackages));

  partitionInfo.mfree();

  /*if (iteration & 1) {
    // Result already in right place
  } else {
    galois::do_all(graph, Copy(graph));
  }*/
}
}
;
* /
#endif

    struct DupPull { // still buggy

  struct LocalData {

    unsigned int nnodes;
    unsigned int max_nodes_per_package;
    unsigned int max_edges_per_package;

    bool* Bit_dup;
    bool* Bit_next;
    float* PR_dup;
    float* PR_next;
    unsigned int* nodeList;
    unsigned int* edgeList;
    unsigned int* outDegree;
    unsigned int* inDegree;
    LocalData(unsigned int nnodes1, unsigned int max_nodes_per_package1,
              unsigned int max_edges_per_package1)
        : nnodes(nnodes1), max_nodes_per_package(max_nodes_per_package1),
          max_edges_per_package(max_edges_per_package1){};

    void alloc() {
      using namespace galois::runtime::MM;
      /*Bit_curr = (bool * ) largeAlloc( max_nodes_per_package * sizeof(bool));
      Bit_next = (bool * ) largeAlloc( max_nodes_per_package * sizeof(bool));
      PR_curr = (float * ) largeAlloc( max_nodes_per_package * sizeof(float));
      PR_next = (float * ) largeAlloc( max_nodes_per_package * sizeof(float));
      nodeList = (int * ) largeAlloc( (nnodes + 1) * sizeof(int));
      edgeList = (int * ) largeAlloc( max_edges_per_package * sizeof(int));*/
      /*Bit_curr = (bool * ) std::malloc( max_nodes_per_package * sizeof(bool));
      Bit_next = (bool * ) std::malloc( max_nodes_per_package * sizeof(bool));
      PR_curr = (float * ) std::malloc( max_nodes_per_package * sizeof(float));
      PR_next = (float * ) std::malloc( max_nodes_per_package * sizeof(float));
      nodeList = (int * ) std::malloc( (nnodes + 1) * sizeof(int));
      edgeList = (int * ) std::malloc( max_edges_per_package * sizeof(int));*/
      Bit_dup  = (bool*)numa_alloc_local(nnodes * sizeof(bool));
      Bit_next = (bool*)numa_alloc_local(max_nodes_per_package * sizeof(bool));
      PR_dup   = (float*)numa_alloc_local(nnodes * sizeof(float));
      PR_next = (float*)numa_alloc_local(max_nodes_per_package * sizeof(float));
      nodeList =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));
      edgeList = (unsigned int*)numa_alloc_local(max_edges_per_package *
                                                 sizeof(unsigned int));
      outDegree =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));
      inDegree =
          (unsigned int*)numa_alloc_local((nnodes + 1) * sizeof(unsigned int));

      next_reset();
    }

    void mfree() {
      using namespace galois::runtime::MM;
      /*largeFree( (void *)Bit_curr, max_nodes_per_package * sizeof(bool));
      largeFree( (void *)Bit_next, max_nodes_per_package * sizeof(bool));
      largeFree( (void *)PR_curr, max_nodes_per_package * sizeof(float));
      largeFree( (void *)PR_next, max_nodes_per_package * sizeof(float));
      largeFree( (void *)nodeList, (nnodes + 1) * sizeof(int));
      largeFree( (void *)edgeList, max_edges_per_package * sizeof(int));*/
      /*std::free( (void *)Bit_curr);
      std::free( (void *)Bit_next);
      std::free( (void *)PR_curr);
      std::free( (void *)PR_next);
      std::free( (void *)nodeList);
      std::free( (void *)edgeList);*/
      numa_free((void*)Bit_dup, nnodes * sizeof(bool));
      numa_free((void*)Bit_next, max_nodes_per_package * sizeof(bool));
      numa_free((void*)PR_dup, nnodes * sizeof(float));
      numa_free((void*)PR_next, max_nodes_per_package * sizeof(float));
      numa_free((void*)nodeList, (nnodes + 1) * sizeof(unsigned int));
      numa_free((void*)edgeList, max_edges_per_package * sizeof(unsigned int));
      numa_free((void*)outDegree, (nnodes + 1) * sizeof(unsigned int));
      numa_free((void*)inDegree, (nnodes + 1) * sizeof(unsigned int));
    }

    void next_reset() {
      memset(PR_next, 0, max_nodes_per_package * sizeof(float));
      memset(Bit_next, 0, max_nodes_per_package * sizeof(bool));
      // printf(" %d %d %f %f\n",Bit_next[0], Bit_next[max_nodes_per_package -
      // 1], PR_next[0], PR_next[max_nodes_per_package - 1] );
    }
  };

  struct LNode {
    unsigned int outDegree;
    unsigned int inDegree;
    unsigned int id;
    float pagerank;
    float getPageRank() { return pagerank; }
    void setPageRank(float pr) { pagerank = pr; }
  };
  typedef galois::graphs::LC_InlineEdge_Graph<LNode, void>::with_no_lockable<
      true>::type ::with_numa_alloc<false>::type InnerGraph;

  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  /*typedef galois::graphs::LC_CSR_Graph<LNode,void>
    ::with_no_lockable<true>::type Graph;*/
  typedef Graph::GraphNode GNode;

  typedef galois::runtime::PerPackageStorage<LocalData> PerPackageData;

  typedef galois::runtime::Barrier Barrier;

  std::string name() const { return "duplicate pull"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<float> sum_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::NONE);
      unsigned int outs =
          std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                        g.edge_end(n, galois::MethodFlag::NONE));
      data.outDegree = outs;
      unsigned int ins =
          std::distance(g.in_edge_begin(n, galois::MethodFlag::NONE),
                        g.in_edge_end(n, galois::MethodFlag::NONE));
      data.inDegree = ins;
      GNode start   = *g.begin();
      data.id       = g.idFromNode(n); // n - start;
    }
  };

  struct Copy {
    Graph& graph;
    PerPackageData& packageData;
    unsigned int* nodesPerPackage;
    unsigned int* nodesStartPackage;
    unsigned int nPackages;
    Copy(Graph& g, PerPackageData& pData, unsigned int* nodesP,
         unsigned int nPackages)
        : graph(g), packageData(pData), nodesPerPackage(nodesP),
          nPackages(nPackages) {
      nodesStartPackage            = new unsigned int[nPackages + 1];
      nodesStartPackage[0]         = 0;
      nodesStartPackage[nPackages] = graph.size();
      for (int i = 1; i < nPackages; i++) {
        nodesStartPackage[i] = nodesStartPackage[i - 1] + nodesPerPackage[i];
      }
    }
    void operator()(unsigned tid, unsigned numT) {
      if (galois::runtime::LL::isPackageLeader(tid)) {
        unsigned int currPackage =
            galois::runtime::LL::getPackageForThread(tid);
        for (auto ii = graph.begin(), end = graph.end(); ii != end; ii++) {
          GNode src    = *ii;
          LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);

          if (sdata.id >= nodesStartPackage[currPackage] &&
              sdata.id < nodesStartPackage[currPackage + 1])
            sdata.pagerank =
                packageData.getLocal()
                    ->PR_next[sdata.id - nodesStartPackage[currPackage]];
        }

        packageData.getLocal()->mfree();
      }
    }
  };

  struct distributeEdges {
    Graph& graph;
    PerPackageData& packageData;
    unsigned int* nodesPerPackage;
    unsigned int* edgesPerPackage;

    distributeEdges(Graph& g, PerPackageData& p, unsigned int* nodes,
                    unsigned int* edges)
        : graph(g), packageData(p), nodesPerPackage(nodes),
          edgesPerPackage(edges){};

    void operator()(unsigned tid, unsigned numT) {
      if (galois::runtime::LL::isPackageLeader(tid)) {
        // printf("tid: %d\n", tid);
        packageData.getLocal()->alloc();

        unsigned int nnodes = graph.size();
        unsigned int currPackage =
            galois::runtime::LL::getPackageForThread(tid);
        unsigned int* outDegree = packageData.getLocal()->outDegree;
        unsigned int* inDegree  = packageData.getLocal()->inDegree;
        packageData.getLocal()->nodeList[nnodes] = edgesPerPackage[currPackage];
        // printf("tid: %d, edgesPerPackage: %d", tid,
        // edgesPerPackage[currPackage]);
        unsigned int rangeLow = 0;
        for (int i = 0; i < currPackage; i++) {
          rangeLow += nodesPerPackage[i];
        }
        unsigned int rangeHi = rangeLow + nodesPerPackage[currPackage];

        unsigned int edgeCount = 0;
        // std::cerr << "tid: " << tid <<"range:"<<rangeLow<<"-"<<rangeHi<<
        // std::endl; printf("work0.3 %d %d\n", rangeLow, rangeHi);
        unsigned int indegree_temp = 0;

        // std::cerr << "tid: " << tid <<"edgeCount:"<<edgeCount<< " inDegree
        // "<< indegree_temp<<std::endl;
        edgeCount = 0;
        for (auto ii = graph.begin(), end = graph.end(); ii != end; ii++) {
          GNode src    = *ii;
          LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
          // printf("%d %d\n", sdata.id, edgeCount);
          packageData.getLocal()->nodeList[sdata.id] = edgeCount;
          outDegree[sdata.id]                        = sdata.outDegree;
          inDegree[sdata.id]                         = sdata.inDegree;
          for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                    ej = graph.edge_end(src, galois::MethodFlag::NONE);
               jj != ej; ++jj) {
            GNode dst    = graph.getEdgeDst(jj);
            LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);

            if (ddata.id < rangeHi && ddata.id >= rangeLow) {
              packageData.getLocal()->edgeList[edgeCount] = ddata.id;
              edgeCount++;
            }
          }
        }
        // std::cerr << "tid1: " << tid << std::endl;
        // printf("work0.5\n");
        //===set node bit and PR==============
        for (unsigned i = 0; i < nnodes; i++) {
          packageData.getLocal()->PR_dup[i]  = 1.0;
          packageData.getLocal()->Bit_dup[i] = true;
        }
      }
    }
  };

  struct PartitionInfo {
    unsigned int numThreads, nnodes, nPackages, coresPerPackage,
        coresLastPackage;
    unsigned* nodesPerPackage;
    unsigned* nodesStartPackage;
    unsigned* edgesPerPackage;
    unsigned* nodesPerCore;
    unsigned* nodesStartCore;
    unsigned* nodesPerCoreLastPkg;
    unsigned* nodesStartCoreLastPkg;
    PartitionInfo(unsigned nnodes, unsigned numThreads, unsigned nPackages,
                  unsigned coresPerPackage, unsigned coresLastPackage)
        : nnodes(nnodes), numThreads(numThreads), nPackages(nPackages),
          coresPerPackage(coresPerPackage), coresLastPackage(coresLastPackage) {
      edgesPerPackage   = new unsigned[nPackages];
      nodesPerPackage   = new unsigned[nPackages];
      nodesStartPackage = new unsigned[nPackages + 1];

      nodesPerCore   = new unsigned[coresPerPackage];
      nodesStartCore = new unsigned[coresPerPackage];

      nodesPerCoreLastPkg   = new unsigned[coresLastPackage];
      nodesStartCoreLastPkg = new unsigned[coresLastPackage];

      memset(edgesPerPackage, 0, sizeof(unsigned) * nPackages);
      memset(nodesPerPackage, 0, sizeof(unsigned) * nPackages);
      memset(nodesStartPackage, 0, sizeof(unsigned) * nPackages);

      memset(nodesPerCore, 0, sizeof(unsigned) * coresPerPackage);
      memset(nodesStartCore, 0, sizeof(unsigned) * coresPerPackage);
      memset(nodesPerCoreLastPkg, 0, sizeof(unsigned) * coresLastPackage);
      memset(nodesStartCoreLastPkg, 0, sizeof(unsigned) * coresLastPackage);
    }

    void partitionByDegree(Graph& graph, unsigned numThreads,
                           unsigned* nodesPerThread, unsigned* edgesPerThread) {
      int n = graph.size();
      // int *degrees = new int [n];

      // parallel_for(intT i = 0; i < n; i++) degrees[i] =
      // GA.V[i].getInDegree();}

      unsigned int* accum = new unsigned int[numThreads];
      for (int i = 0; i < numThreads; i++) {
        accum[i]          = 0;
        nodesPerThread[i] = 0;
      }

      unsigned int averageDegree = graph.sizeEdges() / numThreads;
      std::cout << "averageDegree is " << averageDegree << std::endl;
      int counter = 0;
      for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        GNode src    = *ii;
        LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);
        accum[counter] += sdata.inDegree;
        nodesPerThread[counter]++;
        if ((accum[counter] >= averageDegree && counter < numThreads - 1) ||
            ii == ei - 1) {
          edgesPerThread[counter] = accum[counter];
          counter++;
        }
      }
      delete[] accum;
    }

    void subPartitionByDegree(Graph& graph, int nCores, unsigned* nodesPerCore,
                              unsigned* nodesStartCore) {
      unsigned n = graph.size();

      unsigned aveNodes = n / nCores;

      for (int i = 0; i < nCores - 1; i++) {
        nodesPerCore[i]   = aveNodes;
        nodesStartCore[i] = aveNodes * i;
      }
      nodesStartCore[nCores - 1] = aveNodes * (nCores - 1);
      nodesPerCore[nCores - 1]   = n - nodesStartCore[nCores - 1];
    }

    void partition(Graph& graph) {
      unsigned* nodesPerThread = new unsigned[numThreads];
      unsigned* edgesPerThread = new unsigned[numThreads];

      partitionByDegree(graph, numThreads, nodesPerThread, edgesPerThread);

      for (int i = 0; i < nPackages; i++) {
        int coresCurrPackage;
        if (i == nPackages - 1)
          coresCurrPackage = coresLastPackage;
        else
          coresCurrPackage = coresPerPackage;

        nodesPerPackage[i] = 0;
        edgesPerPackage[i] = 0;

        for (int j = 0; j < coresCurrPackage; j++) {
          nodesPerPackage[i] += nodesPerThread[coresPerPackage * i + j];
          edgesPerPackage[i] += edgesPerThread[coresPerPackage * i + j];
        }

        if (i > 0)
          nodesStartPackage[i] =
              nodesStartPackage[i - 1] + nodesPerPackage[i - 1];
      }
      nodesStartPackage[nPackages] =
          nodesStartPackage[nPackages - 1] + nodesPerPackage[nPackages - 1];

      subPartitionByDegree(graph, coresPerPackage, nodesPerCore,
                           nodesStartCore);
      subPartitionByDegree(graph, coresLastPackage, nodesPerCoreLastPkg,
                           nodesStartCoreLastPkg);

      // printf("nodesStartPackage: %d %d %d\n", nodesStartPackage[0],
      // nodesStartPackage[1], nodesStartPackage[2]);
      /*for(int i = 0; i < numThreads; i++)
      {
        printf("%d %d ", i, nodesPerThread[i]);
      }
      printf("\n");
      for(int i = 0; i < numThreads; i++)
      {
        printf("%d %d ", i, edgesPerThread[i]);
      }
      printf("\n");*/
      delete[] nodesPerThread;
      delete[] edgesPerThread;
    }

    void mfree() {
      delete[] edgesPerPackage;
      delete[] nodesPerPackage;
      delete[] nodesStartPackage;

      delete[] nodesPerCore;
      delete[] nodesStartCore;

      delete[] nodesPerCoreLastPkg;
      delete[] nodesStartCoreLastPkg;
    }

    void print() {
      printf("ePerPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, edgesPerPackage[i]);
        printf("\n");
      }
      printf("nPerPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, nodesPerPackage[i]);
        printf("\n");
      }
      printf("nstartPkg: ");
      for (int i = 0; i < nPackages; i++) {

        printf("%d %d ", i, nodesStartPackage[i]);
        printf("\n");
      }
      printf("nPerCore: ");
      for (int i = 0; i < coresPerPackage; i++) {

        printf("%d %d ", i, nodesPerCore[i]);
        printf("\n");
      }
      printf("nStartCore: ");
      for (int i = 0; i < coresPerPackage; i++) {

        printf("%d %d ", i, nodesStartCore[i]);
        printf("\n");
      }
      printf("nPerCoreLastPkg: ");
      for (int i = 0; i < coresLastPackage; i++) {

        printf("%d %d ", i, nodesPerCoreLastPkg[i]);
        printf("\n");
      }
      printf("nPerCorePkg: ");
      for (int i = 0; i < coresLastPackage; i++) {

        printf("%d %d", i, nodesStartCoreLastPkg[i]);
        printf("\n");
      }
    }
  };

  struct Process {
    DupPull* self;
    PerPackageData& packageData;
    PartitionInfo& partitionInfo;
    Barrier& gbarrier;
    Process(PerPackageData& p, PartitionInfo& partitionInfo, Barrier& gbarrier,
            DupPull* self)
        : packageData(p), partitionInfo(partitionInfo), gbarrier(gbarrier),
          self(self) {}
    void update(float* PR_next, unsigned offset, float value) {
      writeAdd(PR_next + offset, value);
      // PR_next[offset] += value;
    }
    void operator()(unsigned tid, unsigned numT) {
      unsigned leader      = galois::runtime::LL::getLeaderForThread(tid);
      unsigned currPackage = galois::runtime::LL::getPackageForThread(tid);

      unsigned nPackages = partitionInfo.nPackages;

      unsigned nodesCurrPackage = partitionInfo.nodesPerPackage[currPackage];
      unsigned coresCurrPackage = (currPackage == nPackages - 1)
                                      ? partitionInfo.coresLastPackage
                                      : partitionInfo.coresPerPackage;
      unsigned nodesPerCore = nodesCurrPackage / coresCurrPackage;
      unsigned threadStart  = (tid - leader) * nodesPerCore;
      unsigned threadEnd;
      if (tid - leader == coresCurrPackage - 1)
        threadEnd = nodesCurrPackage;
      else
        threadEnd = (tid - leader + 1) * nodesPerCore;

      unsigned activePackage = 0;
      bool *Bit_curr, *Bit_next;
      unsigned *edgelist, *nodelist, *outDegree, *inDegree;
      float *PR_curr, *PR_next;
      unsigned startOffset, packageOffset, nout;
      unsigned firstEdge, src, localnout;
      float sum, delta;

      unsigned* nodesStartPackage = partitionInfo.nodesStartPackage;
      for (int i = 0; i < nPackages; i++) {
        if (threadStart >= nodesStartPackage[i] &&
            threadStart < nodesStartPackage[i + 1]) {
          activePackage = i;
          startOffset   = nodesStartPackage[i];
          break;
        }
      }
      nodelist  = packageData.getLocal()->nodeList;
      edgelist  = packageData.getLocal()->edgeList;
      outDegree = packageData.getLocal()->outDegree;
      inDegree  = packageData.getLocal()->inDegree;
      Bit_curr  = packageData.getLocal()->Bit_dup;
      PR_curr   = packageData.getLocal()->PR_dup;

      PR_next  = packageData.getLocal()->PR_next;
      Bit_next = packageData.getLocal()->Bit_next;

      // gbarrier.wait();
      // printf("tid: %d start: %d end: %d \n", tid, threadStart, threadEnd);
      // gbarrier.wait();
      for (unsigned i = threadStart; i < threadEnd; i++) {
        sum = 0;
        if (nodelist[i] != nodelist[i + 1]) {
          for (unsigned j = nodelist[i]; j < nodelist[i + 1]; j++) {
            src   = edgelist[j];
            delta = PR_curr[src] / (float)outDegree[src];

            if (delta >= tolerance && Bit_curr[src]) {
              sum += delta;
              Bit_next[packageOffset] = true;
            }
          }
        }
        PR_next[i] = sum;
      }
    }
  };

  struct Process2 {
    DupPull* self;
    PerPackageData& packageData;
    PartitionInfo& partitionInfo;
    Barrier& gbarrier;
    Process2(PerPackageData& p, PartitionInfo& partitionInfo, Barrier& gbarrier,
             DupPull* self)
        : packageData(p), partitionInfo(partitionInfo), gbarrier(gbarrier),
          self(self) {}

    void damping(float* PR_next, unsigned tid, DupPull* self,
                 PerPackageData& packageData) {
      unsigned localtid = tid - galois::runtime::LL::getLeaderForThread(tid);
      unsigned coresCurrPackage;
      unsigned nodesPerCore;
      unsigned currPackage = galois::runtime::LL::getPackageForThread(tid);
      if (currPackage != partitionInfo.nPackages - 1) {
        coresCurrPackage = partitionInfo.coresPerPackage;
      } else {
        coresCurrPackage = partitionInfo.coresLastPackage;
      }

      nodesPerCore =
          partitionInfo.nodesPerPackage[currPackage] / coresCurrPackage;
      float diff;
      unsigned offset = partitionInfo.nodesStartPackage[currPackage];
      if (localtid != coresCurrPackage - 1) {
        for (unsigned i = localtid * nodesPerCore;
             i < (localtid + 1) * nodesPerCore; i++) {
          PR_next[i] = alpha2 * PR_next[i] + 1 - alpha2;
          diff       = std::fabs(packageData.getLocal()->PR_dup[i + offset] -
                           PR_next[i]);
          self->max_delta.update(diff);
          self->sum_delta.update(diff);
        }
      } else {
        for (unsigned i = localtid * nodesPerCore;
             i < partitionInfo.nodesPerPackage[currPackage]; i++) {
          PR_next[i] = alpha2 * PR_next[i] + 1 - alpha2;
          diff       = std::fabs(packageData.getLocal()->PR_dup[i + offset] -
                           PR_next[i]);
          self->max_delta.update(diff);
          self->sum_delta.update(diff);
        }
      }
    }

    void operator()(unsigned tid, unsigned numT) {
      float* PR_next = packageData.getLocal()->PR_next;
      damping(PR_next, tid, self, packageData);
      gbarrier.wait();
      if (galois::runtime::LL::isPackageLeader(tid)) {

        unsigned currPackage = galois::runtime::LL::getPackageForThread(tid);
        unsigned offset      = partitionInfo.nodesStartPackage[currPackage];
        unsigned nodesCurrPackage = partitionInfo.nodesPerPackage[currPackage];
        unsigned nPackages        = partitionInfo.nPackages;
        bool* Bit_curr            = packageData.getLocal()->Bit_dup;
        float* PR_curr            = packageData.getLocal()->PR_dup;
        for (unsigned i = 0; i < nPackages; i++) {
          if (i == currPackage) {
            memcpy(Bit_curr + offset, packageData.getLocal()->Bit_next,
                   nodesCurrPackage * sizeof(bool));
            memcpy(PR_curr + offset, packageData.getLocal()->PR_next,
                   nodesCurrPackage * sizeof(float));
          } else {
            memcpy(Bit_curr + partitionInfo.nodesStartPackage[i],
                   packageData.getRemoteByPkg(i)->Bit_next,
                   partitionInfo.nodesPerPackage[i] * sizeof(bool));
            memcpy(PR_curr + partitionInfo.nodesStartPackage[i],
                   packageData.getRemoteByPkg(i)->PR_next,
                   partitionInfo.nodesPerPackage[i] * sizeof(float));
          }
        }

        packageData.getLocal()->next_reset();
      }
    }
  };

  void operator()(Graph& graph) {
    // nPackages = LL::getMaxPackages();
    unsigned coresPerPackage = galois::runtime::LL::getMaxCores() /
                               galois::runtime::LL::getMaxPackages();

    unsigned nPackages        = (numThreads - 1) / coresPerPackage + 1;
    unsigned coresLastPackage = numThreads % coresPerPackage;
    if (coresLastPackage == 0)
      coresLastPackage = 10;
    Barrier& gbarrier = galois::runtime::getSystemBarrier();

    unsigned nnodes = graph.size();
    std::cout << "nnodes:" << nnodes << " nedges:" << graph.sizeEdges()
              << std::endl;

    PartitionInfo partitionInfo(nnodes, numThreads, nPackages, coresPerPackage,
                                coresLastPackage);

    partitionInfo.partition(graph);

    unsigned max_nodes_per_package = 0, max_edges_per_package = 0;
    unsigned* nodesPerPackage = partitionInfo.nodesPerPackage;
    unsigned* edgesPerPackage = partitionInfo.edgesPerPackage;
    for (unsigned i = 0; i < nPackages; i++) {
      max_nodes_per_package = (max_nodes_per_package > nodesPerPackage[i])
                                  ? max_nodes_per_package
                                  : nodesPerPackage[i];
      max_edges_per_package = (max_edges_per_package > edgesPerPackage[i])
                                  ? max_edges_per_package
                                  : edgesPerPackage[i];
    }

    PerPackageData packageData(nnodes, max_nodes_per_package,
                               max_edges_per_package);

    galois::on_each(
        distributeEdges(graph, packageData, nodesPerPackage, edgesPerPackage));
    unsigned int iteration = 0;
    galois::StatTimer T("pure time");

    // partitionInfo.print();
    T.start();
    while (true) {
      galois::on_each(Process(packageData, partitionInfo, gbarrier, this));
      galois::on_each(Process2(packageData, partitionInfo, gbarrier, this));
      iteration += 1;

      float delta = max_delta.reduce();

      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta.reduce()
                << " max delta: " << delta << "\n";
      // delta = 1;
      if (delta <= tolerance || iteration >= maxIterations)
        break;

      max_delta.reset();
      sum_delta.reset();
      // galois::on_each(Copy(graph, packageData, nodesPerPackage,nPackages));
    }
    T.stop();
    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }
    galois::on_each(Copy(graph, packageData, nodesPerPackage, nPackages));

    partitionInfo.mfree();

    /*if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }*/
  }
};

struct PushAlgo {
  struct LNode {
    float value[2];
    unsigned outDegree;
    float getPageRank() { return value[1]; }
    float getPageRank(unsigned int it) { return value[it & 1]; }
    void setPageRank(unsigned it, float v) { value[(it + 1) & 1] = v; }
  };
  typedef typename galois::graphs::LC_InlineEdge_Graph<LNode, void>::
      with_numa_alloc<true>::type ::with_no_lockable<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Push"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<size_t> small_delta;
  galois::GAccumulator<float> sum_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data    = g.getData(n, galois::MethodFlag::NONE);
      data.value[0]  = 1.0;
      data.value[1]  = 0.0;
      int outs       = std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                               g.edge_end(n, galois::MethodFlag::NONE));
      data.outDegree = outs;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.value[1] = data.value[0];
    }
  };

  struct Reset {
    Graph& g;
    unsigned iteration;
    Reset(Graph& g, unsigned i) : g(g), iteration(i) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::NONE);
      data.value[(iteration + 1) & 1] =
          data.value[(iteration + 1) & 1] * alpha2 + 1 - alpha2;
      data.value[iteration & 1] = 0.0;
    }
  };

  struct Process {
    PushAlgo* self;
    Graph& graph;
    unsigned int iteration;

    Process(PushAlgo* s, Graph& g, unsigned int i)
        : self(s), graph(g), iteration(i) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      LNode& sdata       = graph.getData(src, galois::MethodFlag::NONE);
      unsigned outDegree = sdata.outDegree;
      float delta        = sdata.getPageRank(iteration) / (float)outDegree;
      double sum         = 0;

      for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst    = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);
        if (delta > tolerance)
          writeAdd(&ddata.value[(iteration + 1) & 1], delta);
      }

      float diff = delta;

      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
      self->sum_delta.update(diff);
      // sdata.setPageRank(iteration, value);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;

    while (true) {
      galois::for_each(
          graph, Process(this, graph, iteration),
          galois::wl<galois::worklists::PerSocketChunkFIFO<256>>());
      galois::do_all(graph, Reset(graph, iteration));
      iteration += 1;

      float delta   = max_delta.reduce();
      size_t sdelta = small_delta.reduce();

      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta.reduce()
                << " max delta: " << delta << " small delta: " << sdelta << " ("
                << sdelta / (float)graph.size() << ")"
                << "\n";

      if (delta <= tolerance || iteration >= maxIterations)
        break;
      max_delta.reset();
      small_delta.reset();
      sum_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }
  }
};

//-------------michael's code end-----------------------

/* ------------------------- Joyce's codes start ------------------------- */
//---------- parallel synchronous algorithm (original copy: PullAlgo2, readGraph
// is re-written.)

struct Synch {
  struct LNode {
    float value[2];
    int id;
    unsigned int nout;
    float getPageRank() { return value[1]; }
    float getPageRank(unsigned int it) { return value[it & 1]; }
    void setPageRank(unsigned it, float v) { value[(it + 1) & 1] = v; }
  };

  typedef typename galois::graphs::LC_InlineEdge_Graph<LNode, void>::
      with_numa_alloc<true>::type ::with_no_lockable<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return "Synch"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.value[0] = 1.0;
      data.value[1] = 1.0;
      int outs      = std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                               g.edge_end(n, galois::MethodFlag::NONE));
      data.nout     = outs;
      data.id       = idcount++;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    Synch* self;
    Graph& graph;
    unsigned int iteration;

    Process(Synch* s, Graph& g, unsigned int i)
        : self(s), graph(g), iteration(i) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {

      LNode& sdata = graph.getData(src, galois::MethodFlag::NONE);

      // std::cout<<sdata.id<<" picked up...\n";
      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.in_edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst    = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);
        sum += ddata.getPageRank(iteration) / ddata.nout;
        // std::cout<<"- id: "<<ddata.id<<"\n";
      }

      float value = (1.0 - alpha) * sum + alpha;
      float diff  = std::fabs(value - sdata.getPageRank(iteration));
      sdata.setPageRank(iteration, value);
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;

    while (true) {
      galois::for_each(graph, Process(this, graph, iteration));
      iteration += 1;

      float delta   = max_delta.reduce();
      size_t sdelta = small_delta.reduce();

      std::cout << "iteration: " << iteration << " max delta: " << delta
                << " small delta: " << sdelta << " ("
                << sdelta / (float)graph.size() << ")"
                << "\n";

      if (delta <= tolerance || iteration >= maxIterations) {
        break;
      }
      max_delta.reset();
      small_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }
  }
};

//---------- parallel prioritized asynchronous algorithm (max. residual)
struct PrtRsd {

  struct LNode {
    float pagerank;
    int id;
    float residual;
    unsigned int nout;
    float getPageRank() { return pagerank; }
    float getResidual() { return residual; }
  };

  typedef galois::graphs::LC_InlineEdge_Graph<LNode, void>::with_numa_alloc<
      true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "PrtRsd"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.pagerank = (1.0 - alpha2);
      data.residual = 0.0;
      data.id       = idcount++;
      int outs      = std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                               g.edge_end(n, galois::MethodFlag::NONE));
      data.nout     = outs;
    }
  };

  struct Process1 {
    PrtRsd* self;
    Graph& graph;

    Process1(PrtRsd* s, Graph& g) : self(s), graph(g) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      LNode& data = graph.getData(src);
      // for each out-going neighbour, add residuals
      for (auto jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej;
           ++jj) {
        GNode dst      = graph.getInEdgeDst(jj);
        LNode& ddata   = graph.getData(dst);
        ddata.residual = (float)ddata.residual + (float)1 / data.nout;
      }
    }
  }; //--- end of Process1

  struct Process2 {
    PrtRsd* self;
    Graph& graph;

    Process2(PrtRsd* s, Graph& g) : self(s), graph(g) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      // scale the residual
      LNode& data   = graph.getData(src, galois::MethodFlag::NONE);
      data.residual = alpha2 * (1 - alpha2) * data.residual;
    }
  }; //--- end of Process2

  // define priority
  typedef std::pair<int, GNode> UpdateRequest;
  struct UpdateRequestIndexer : public std::unary_function<UpdateRequest, int> {
    int operator()(const UpdateRequest& val) const { return val.first; }
  };

  struct Process3 {
    PrtRsd* self;
    Graph& graph;
    galois::Statistic& pre;
    galois::Statistic& post;

    Process3(PrtRsd* s, Graph& g, galois::Statistic& _pre,
             galois::Statistic& _post)
        : self(s), graph(g), pre(_pre), post(_post) {}

    void operator()(const UpdateRequest& srcRq,
                    galois::UserContext<UpdateRequest>& ctx) {
      GNode src   = srcRq.second;
      LNode* node = &graph.getData(src, galois::MethodFlag::NONE);
      if ((node->residual < tolerance) ||
          (amp * (int)node->residual != srcRq.first)) {
        post += 1;
        return;
      }
      node = &graph.getData(src);

      // update pagerank (consider each in-coming edge)
      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.in_edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst    = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);
        sum += ddata.getPageRank() / ddata.nout;
      }

      double lpr      = alpha2 * sum + (1 - alpha2);
      double lres     = node->residual * alpha2 / node->nout;
      unsigned nopush = 0;

      // update residual (consider each out-going edge)
      for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst    = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);
        float oldR   = ddata.residual;
        ddata.residual += lres;
        if (ddata.residual > tolerance &&
            ((int)oldR != (int)ddata.residual || oldR < tolerance)) {
          ctx.push(
              std::make_pair(amp * (int)ddata.residual, dst)); // max residual
        } else {
          ++nopush;
        }
      }
      if (nopush)
        pre += nopush;

      node->pagerank = lpr;
      node->residual = 0.0;
    }

  }; //--- end of Process3

  void operator()(Graph& graph) {
    galois::Statistic pre("PrePrune");
    galois::Statistic post("PostPrune");
    galois::for_each(graph, Process1(this, graph));
    galois::for_each(graph, Process2(this, graph));
    std::cout << "tolerance: " << tolerance << ", amp2: " << amp2 << "\n";
    using namespace galois::worklists;
    typedef PerSocketChunkLIFO<4> PSchunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer, PSchunk> OBIM;
    typedef WorkListTracker<UpdateRequestIndexer, OBIM> dOBIM;
    galois::InsertBag<UpdateRequest> initialWL;
    galois::do_all(graph, [&initialWL, &graph](GNode src) {
      LNode& data = graph.getData(src);
      if (data.residual > tolerance) {
        initialWL.push_back(
            std::make_pair(amp * (int)data.residual, src)); // max residual
      }
    });
    galois::StatTimer T("InnerTime");
    T.start();
    galois::for_each(initialWL, Process3(this, graph, pre, post),
                     galois::wl<dOBIM>(), galois::loopname("mainloop"));
    T.stop();
  }
};

//---------- parallel prioritized asynchronous algorithm (degree biased)
struct PrtDeg {

  struct LNode {
    float pagerank;
    int id;
    float residual;
    unsigned int nout;
    unsigned int deg;
    float getPageRank() { return pagerank; }
    float getResidual() { return residual; }
  };

  typedef galois::graphs::LC_InlineEdge_Graph<LNode, void>::with_numa_alloc<
      true>::type ::with_no_lockable<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "PrtDeg"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data   = g.getData(n, galois::MethodFlag::NONE);
      data.pagerank = (1.0 - alpha2);
      data.residual = 0.0;
      data.id       = idcount++;
      int outs      = std::distance(g.edge_begin(n, galois::MethodFlag::NONE),
                               g.edge_end(n, galois::MethodFlag::NONE));
      data.nout     = outs;
      int ins  = std::distance(g.in_edge_begin(n, galois::MethodFlag::NONE),
                              g.in_edge_end(n, galois::MethodFlag::NONE));
      data.deg = outs + ins;
    }
  };

  struct Process1 {
    PrtDeg* self;
    Graph& graph;

    Process1(PrtDeg* s, Graph& g) : self(s), graph(g) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      LNode& data = graph.getData(src, galois::MethodFlag::NONE);
      // for each out-going neighbour, add residuals
      for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst      = graph.getInEdgeDst(jj);
        LNode& ddata   = graph.getData(dst, galois::MethodFlag::NONE);
        ddata.residual = (float)ddata.residual + (float)1 / data.nout;
      }
    }
  }; //--- end of Process1

  struct Process2 {
    PrtDeg* self;
    Graph& graph;

    Process2(PrtDeg* s, Graph& g) : self(s), graph(g) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      // scale the residual
      LNode& data   = graph.getData(src, galois::MethodFlag::NONE);
      data.residual = alpha2 * (1 - alpha2) * data.residual;
    }
  }; //--- end of Process2

  // define priority
  typedef std::pair<int, GNode> UpdateRequest;
  struct UpdateRequestIndexer : public std::unary_function<UpdateRequest, int> {
    int operator()(const UpdateRequest& val) const { return val.first; }
  };

  struct Process3 {
    PrtDeg* self;
    Graph& graph;

    Process3(PrtDeg* s, Graph& g) : self(s), graph(g) {}

    void operator()(const UpdateRequest& srcRq,
                    galois::UserContext<UpdateRequest>& ctx) {
      GNode src   = srcRq.second;
      LNode* node = &graph.getData(src, galois::MethodFlag::NONE);
      int tmp     = (*node).residual * amp / (*node).deg; // degree biased
      if (tmp != srcRq.first) {
        return;
      } else if ((*node).residual < tolerance) {
        std::cout << "amp should be adjusted... results are not reliable... "
                  << tmp << " " << srcRq.first << " " << (*node).residual
                  << "\n";
        return;
      }

      // update pagerank (consider each in-coming edge)
      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.in_edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst    = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);
        sum += ddata.getPageRank() / ddata.nout;
      }
      node->pagerank = alpha2 * sum + (1 - alpha2);

      // update residual (consider each out-going edge)
      for (auto jj = graph.edge_begin(src, galois::MethodFlag::NONE),
                ej = graph.edge_end(src, galois::MethodFlag::NONE);
           jj != ej; ++jj) {
        GNode dst    = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::NONE);
        ddata.residual =
            (float)ddata.residual + (float)node->residual * alpha2 / node->nout;
        if (ddata.residual > tolerance) {
          ctx.push(std::make_pair(ddata.residual * amp / ddata.deg,
                                  dst)); // degree biased
        }
      }
      node->residual = 0.0;
    }

  }; //--- end of Process3

  void operator()(Graph& graph) {

    galois::for_each(graph, Process1(this, graph));
    galois::for_each(graph, Process2(this, graph));
    std::cout << "tolerance: " << tolerance << ", amp: " << amp << "\n";
    using namespace galois::worklists;
    typedef PerSocketChunkLIFO<16> PSchunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer, PSchunk> OBIM;
    galois::InsertBag<UpdateRequest> initialWL;
    for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      GNode src   = *ii;
      LNode& data = graph.getData(src, galois::MethodFlag::NONE);
      if (data.residual > tolerance) {
        initialWL.push_back(std::make_pair(data.residual * amp / data.deg,
                                           src)); // degree biased
      }
    }
    galois::for_each(initialWL, Process3(this, graph), galois::wl<OBIM>());
  }
};
/* ------------------------- Joyce's codes end ------------------------- */

//! Transpose in-edges to out-edges
//![WriteGraph]
static void precomputePullData() {
  typedef galois::graphs::LC_CSR_Graph<size_t, void>::with_no_lockable<
      true>::type InputGraph;
  typedef InputGraph::GraphNode InputNode;
  typedef galois::graphs::FileGraphWriter OutputGraph;
  // typedef OutputGraph::GraphNode OutputNode;

  InputGraph input;
  OutputGraph output;
  galois::graphs::readGraph(input, filename);

  size_t node_id = 0;
  for (auto ii = input.begin(), ei = input.end(); ii != ei; ++ii) {
    InputNode src      = *ii;
    input.getData(src) = node_id++;
  }

  output.setNumNodes(input.size());
  output.setNumEdges(input.sizeEdges());
  output.setSizeofEdgeData(sizeof(float));
  output.phase1();

  for (auto ii = input.begin(), ei = input.end(); ii != ei; ++ii) {
    InputNode src = *ii;
    size_t sid    = input.getData(src);
    assert(sid < input.size());

    // size_t num_neighbors = std::distance(input.edge_begin(src),
    // input.edge_end(src));

    for (auto jj = input.edge_begin(src), ej = input.edge_end(src); jj != ej;
         ++jj) {
      InputNode dst = input.getEdgeDst(jj);
      size_t did    = input.getData(dst);
      assert(did < input.size());

      output.incrementDegree(did);
    }
  }

  output.phase2();
  std::vector<float> edgeData;
  edgeData.resize(input.sizeEdges());

  for (auto ii = input.begin(), ei = input.end(); ii != ei; ++ii) {
    InputNode src = *ii;
    size_t sid    = input.getData(src);
    assert(sid < input.size());

    size_t num_neighbors =
        std::distance(input.edge_begin(src), input.edge_end(src));

    float w = 1.0 / num_neighbors;
    for (auto jj = input.edge_begin(src), ej = input.edge_end(src); jj != ej;
         ++jj) {
      InputNode dst = input.getEdgeDst(jj);
      size_t did    = input.getData(dst);
      assert(did < input.size());

      size_t idx    = output.addNeighbor(did, sid);
      edgeData[idx] = w;
    }
  }

  float* t = output.finish<float>();
  std::uninitialized_copy(std::make_move_iterator(edgeData.begin()),
                          std::make_move_iterator(edgeData.end()), t);

  output.toFile(outputPullFilename);
  std::cout << "Wrote " << outputPullFilename << "\n";
}
//![WriteGraph]

//! Make values unique
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
  typedef typename Graph::GraphNode GNode;
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

template <typename Algo>
void run() {
  typedef typename Algo::Graph Graph;

  Algo algo;
  Graph graph;
  std::cout << "Reading graph \n";
  algo.readGraph(graph);
  std::cout << "preAllocating \n";
  galois::preAlloc(numThreads +
                   (graph.size() * sizeof(typename Graph::node_data_type)) /
                       galois::runtime::MM::hugePageSize);
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  std::cout << "Target max delta: " << tolerance << "\n";
  T.start();
  galois::do_all(graph, typename Algo::Initialize(graph));
  algo(graph);
  T.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify)
    printTop(graph, 10);
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;

  if (outputPullFilename.size()) {
    precomputePullData();
    return 0;
  }

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
  case Algo::pull:
    run<PullAlgo>();
    break;
  case Algo::pull2:
    run<PullAlgo2>();
    break;
  case Algo::polypush:
    run<PolyPush>();
    break;
  case Algo::polypull:
    run<PolyPull>();
    break;
  // case Algo::polypull2: run<PolyPull2>(); break;
  case Algo::duppull:
    run<DupPull>();
    break;
  case Algo::push:
    run<PushAlgo>();
    break;
  case Algo::synch:
    run<Synch>();
    break;
  case Algo::prt_rsd:
    run<PrtRsd>();
    break;
  case Algo::prt_deg:
    run<PrtDeg>();
    break;
  case Algo::ligra:
    run<LigraAlgo<false>>();
    break;
  case Algo::ligraChi:
    run<LigraAlgo<true>>();
    break;
  case Algo::graphlab:
    run<GraphLabAlgo<false, false>>();
    break;
  case Algo::graphlabAsync:
    run<GraphLabAlgo<true, true>>();
    break;
  case Algo::pagerankWorklist:
    run<PagerankDelta>();
    break;
  case Algo::serial:
    run<SerialAlgo>();
    break;
  default:
    std::cerr << "Unknown algorithm\n";
    abort();
  }
  T.stop();

  return 0;
}
