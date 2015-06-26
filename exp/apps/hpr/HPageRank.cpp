/** Page rank application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 */


#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/LC_CSR_Graph.h"
#include "Galois/Graph/Util.h"
#include "Lonestar/BoilerPlate.h"

#include "cuda/hpr_cuda.h"

#include <iostream>
#include <typeinfo>
#include <algorithm>

static const char* const name = "Page Rank - Distributed Heterogeneous";
static const char* const desc = "Computes PageRank on Distributed Galois.  Uses pull algorithm, takes the pre-transposed graph.";
static const char* const url = 0;

enum Personality {
  CPU,
  GPU_CUDA,
  GPU_OPENCL
};

namespace cll = llvm::cl;
static cll::opt<Personality> personality ("personality", cll::desc("Personality"), 
					  cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), 
						      clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), 
						      clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"),
						      clEnumValEnd), 
					  cll::init(CPU));
static cll::opt<std::string> inputFile (cll::Positional, cll::desc("<input file (transpose)>"), cll::Required);
static cll::opt<unsigned int> maxIterations ("maxIterations", cll::desc("Maximum iterations"), cll::init(2));

struct LNode {
  float value;
  unsigned int nout;
};

typedef Galois::Graph::LC_CSR_Graph<LNode, void> Graph;
typedef typename Graph::GraphNode GNode;

// Constants for page Rank Algo.
//! d is the damping factor. Alpha is the prob that user will do a random jump, i.e., 1 - d
static const double alpha = (1.0 - 0.85);

//! maximum relative change until we deem convergence
static const double TOLERANCE = 0.1;


struct pGraph {
  Graph& g;
  unsigned g_offset; // LID + g_offset = GID
  unsigned numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes) are replicas
  unsigned numNodes; // number of nodes (may differ from g.size() to simplify loading)
  // [numNodes, g.size()) should be ignored
  std::vector<unsigned> L2G; // GID = L2G[LID - numOwned]
  unsigned id; // my hostid
  std::vector<unsigned> lastNodes; //[ [i - 1], [i]) -> Node owned by host i
  unsigned getHost(unsigned node) { // node is GID
    return *std::upper_bound(lastNodes.begin(), lastNodes.end(), node);
  }
  unsigned G2L(unsigned GID) {
    auto ii = std::find(L2G.begin(), L2G.end(), GID);
    assert(ii != L2G.end());
    return std::distance(L2G.begin(), ii) + numOwned;
  }

  pGraph(Graph& _g) :g(_g) {}
};

pGraph loadGraph(std::string file, unsigned hostID, unsigned numHosts, Graph& out) {
  pGraph retval{out};
  Galois::Graph::FileGraph fg;
  fg.fromFile(file);
  auto p = Galois::block_range(0UL, fg.size(), hostID, numHosts);
  retval.g_offset = p.first;
  retval.numOwned = p.second - p.first;
  std::vector<unsigned> perm(fg.size(), ~0); //[i (orig)] -> j (final)
  unsigned nextSlot = 0;
  std::cout << fg.size() << " " << p.first << " " << p.second << "\n";
  //Fill our partition
  for (unsigned i = p.first; i < p.second; ++i)
    perm[i] = nextSlot++;
  //find ghost cells
  for (auto ii = fg.begin() + p.first; ii != fg.begin() + p.second; ++ii) {
    for (auto jj = fg.edge_begin(*ii); jj != fg.edge_end(*ii); ++jj) {
      //std::cout << *ii << " " << *jj << " " << nextSlot << " " << perm.size() << "\n";
      //      assert(*jj < perm.size());
      auto dst = fg.getEdgeDst(jj);
      if (perm.at(dst) == ~0) {
        perm[dst] = nextSlot++;
        retval.L2G.push_back(dst);
      }
    }
  }
  retval.numNodes = nextSlot;
  //Fill remainder of graph since permute doesn't support truncating
  for (auto ii = fg.begin(); ii != fg.end(); ++ii)
    if (perm[*ii] == ~0)
      perm[*ii] = nextSlot++;
  std::cout << nextSlot << " " << fg.size() << "\n";
  assert(nextSlot == fg.size());
  //permute graph
  Galois::Graph::FileGraph fg2;
  Galois::Graph::permute<void>(fg, perm, fg2);
  Galois::Graph::readGraph(retval.g, fg2);
  return retval;
}


struct InitializeGraph {
  Graph* g;

  void static go(Graph& _g)
  {
    Galois::do_all(_g.begin(), _g.end(), InitializeGraph{&_g}, Galois::loopname("init"));
  }

  void operator()(GNode src) const {
    LNode& sdata = g->getData(src);
    sdata.value = 1.0 - alpha;
    sdata.nout = 2; // FIXME
  }
};

struct PageRank {
  Graph* g;

  void static go(Graph& _g, unsigned num)
  {
    Galois::do_all(_g.begin(), _g.begin() + num, PageRank{&_g}, Galois::loopname("Page Rank"));
  }
  
  void operator() (GNode src) const {
    double sum = 0;
    LNode& sdata = g->getData(src);
    for (auto jj = g->edge_begin(src), ej = g->edge_end(src); jj != ej; ++jj) {
      GNode dst = g->getEdgeDst(jj);
      LNode& ddata = g->getData(dst);
      sum += ddata.value / ddata.nout;
    }
    float value = (1.0 - alpha) * sum + alpha;
    float diff = std::fabs(value - sdata.value);
    sdata.value = value;
  }
};


// [hostid] -> vector of GID that host has replicas of
std::vector<std::vector<unsigned> > remoteReplicas;
// [hostid] -> remote pGraph Structure (locally invalid)
std::vector<pGraph*> magicPointer;

void setRemotePtr(uint32_t hostID, pGraph* p) {
  if (hostID >= magicPointer.size())
    magicPointer.resize(hostID + 1);
  magicPointer[hostID] = p;
}

void recvNodeStatic(unsigned GID, uint32_t hostID) {
  if (hostID >= remoteReplicas.size())
    remoteReplicas.resize(hostID + 1);
  remoteReplicas[hostID].push_back(GID);
}

void setNodeValue(pGraph* p, unsigned GID, float v) {
  p->g.getData(p->G2L(GID)).value = v;
}

void sendGhostCells(Galois::Runtime::NetworkInterface& net, pGraph& g) {
  for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
    for (auto n : remoteReplicas[x]) {
      net.sendAlt(x, setNodeValue, magicPointer[x], n, g.g.getData(n - g.g_offset).value);
    }
  }
}

void loadGraphNonCPU(pGraph &g) {
  assert(personality != CPU);
  
}

int main(int argc, char** argv) {
    LonestarStart (argc, argv, name, desc, url);

    std::cout << "Personality is " << personality << std::endl;

    Galois::StatManager statManager;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    auto& barrier = Galois::Runtime::getSystemBarrier();

    Graph rg;
    pGraph g = loadGraph(inputFile, Galois::Runtime::NetworkInterface::ID, Galois::Runtime::NetworkInterface::Num, rg);

    loadGraphNonCPU(g);

    //local initialization
    if(personality == CPU) {
      InitializeGraph::go(g.g); /* dispatch to appropriate device */
    } else if(personality == GPU_CUDA) {
      test_cuda();
      return 1;
    }


    barrier.wait();

    //send pGraph pointers
    for (uint32_t x = 0; x < Galois::Runtime::NetworkInterface::Num; ++x)
      net.sendAlt(x, setRemotePtr, Galois::Runtime::NetworkInterface::ID, &g);

    //Ask for cells
    for (auto GID : g.L2G)
      net.sendAlt(g.getHost(GID), recvNodeStatic, GID, Galois::Runtime::NetworkInterface::ID);
    barrier.wait();

    for (int i = 0; i < maxIterations; ++i) {

      std::cout << "Staring " << i << "\n";

      //communicate ghost cells
      sendGhostCells(net, g);
      barrier.wait();

      std::cout << "Starting PR\n";

      //Do pagerank
      PageRank::go(rg, g.numOwned);
      barrier.wait();
    }

    return 0;
}





