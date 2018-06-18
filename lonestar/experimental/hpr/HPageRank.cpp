/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Graph/FileGraph.h"
#include "galois/Graph/LC_CSR_Graph.h"
#include "galois/Graph/Util.h"
#include "Lonestar/BoilerPlate.h"

#include "PGraph.h"
#include "cuda/hpr_cuda.h"
#include "cuda/cuda_mtypes.h"
#include "hpr.h"

#include "opencl/OpenCLPrBackend.h"

#include <iostream>
#include <typeinfo>
#include <algorithm>

#define _HETERO_DEBUG_ 0

static const char* const name       = "Page Rank - Distributed Heterogeneous";
static const char* const name_short = "HPageRank";
static const char* const desc =
    "Computes PageRank on Distributed Galois.  Uses pull algorithm, takes the "
    "pre-transposed graph.";
static const char* const url = 0;

enum Personality { CPU, GPU_CUDA, GPU_OPENCL };
std::string personality_str(Personality p) {
  switch (p) {
  case CPU:
    return "CPU";
  case GPU_CUDA:
    return "GPU_CUDA";
  case GPU_OPENCL:
    return "GPU_OPENCL";
  }
  assert(false && "Invalid personality");
  return "";
}

namespace cll = llvm::cl;
static cll::opt<Personality>
    personality("personality", cll::desc("Personality"),
                cll::values(clEnumValN(CPU, "cpu", "Galois CPU"),
                            clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"),
                            clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"),
                            clEnumValEnd),
                cll::init(CPU));
static cll::opt<std::string> inputFile(cll::Positional,
                                       cll::desc("<input file (transpose)>"),
                                       cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(4));
static cll::opt<bool>
    verify("verify",
           cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"),
           cll::init(true));
static cll::opt<int> gpudevice(
    "gpu",
    cll::desc("Select GPU to run on, default is to choose automatically"),
    cll::init(-1));
static cll::opt<float> cldevice(
    "cldevice",
    cll::desc(
        "Select OpenCL device to run on , default is 0.0 (OpenCL backend)"),
    cll::init(0.0));
static cll::opt<std::string>
    personality_set("pset",
                    cll::desc("String specifying personality for each host. "
                              "'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"),
                    cll::init(""));

///////////////////
///////////////////////////////////////////
enum BSP_FIELD_NAMES { PR_VAL_FIELD = 0 };
struct LNode {
  unsigned int bsp_version;
  float value[2]; /*ID=0*/
  unsigned int nout;
  void swap_version(unsigned int field_id) { bsp_version ^= 1 << field_id; }
  unsigned current_version(unsigned int field_id) {
    return (bsp_version & (1 << field_id)) != 0;
  }
  unsigned next_version(unsigned int field_id) {
    return (~bsp_version & (1 << field_id)) != 0;
  }
};
/*struct LNode {
   int bsp_version;
   float value[2]; ID=0
   unsigned int nout;
   void swap_version(int field_id){
      bsp_version ^= 1<<field_id;
   }
   int current_version(int field_id){
      return bsp_version& (1<<field_id);
   }
   int next_version(int field_id){
      return !(bsp_version&(1<<field_id));
   }
};*/
//////////////////////////////////////////////////////////////////////////////////////
typedef galois::graphs::LC_CSR_Graph<LNode, void> Graph;
typedef typename Graph::GraphNode GNode;
//////////////////////////////////////////////////////////////////////////////////////
typedef galois::opencl::LC_LinearArray_Graph<galois::opencl::Array, LNode, void>
    DeviceGraph;

struct CUDA_Context* cuda_ctx;
struct OPENCL_Context<DeviceGraph> cl_ctx;

/*********************************************************************************
 *
 **********************************************************************************/
struct InitializeGraph {
  pGraph<Graph>* g;
  void static go(pGraph<Graph>& _g) {
    galois::do_all(_g.g.begin(), _g.g.begin() + _g.numOwned,
                   InitializeGraph{&_g}, galois::loopname("init"));
  }
  void operator()(GNode src) const {
    LNode& sdata   = g->g.getData(src);
    sdata.value[0] = 1.0 - alpha; // sdata.value[0]
    sdata.value[1] = 0;           // sdata.value[1]
    for (auto nbr = g->g.edge_begin(src); nbr != g->g.edge_end(src); ++nbr) {
      __sync_fetch_and_add(&g->g.getData(g->g.getEdgeDst(*nbr)).nout, 1);
    }
  }
};
/*********************************************************************************
 * CPU PageRank operator implementation.
 **********************************************************************************/
struct WriteBack {
  pGraph<Graph>* g;
  void static go(pGraph<Graph>& _g) {
    galois::do_all(_g.g.begin(), _g.g.begin() + _g.numOwned, WriteBack{&_g},
                   galois::loopname("Writeback"));
  }
  void operator()(GNode src) const {
    LNode& sdata = g->g.getData(src);
    sdata.swap_version(BSP_FIELD_NAMES::PR_VAL_FIELD);
  }
};
struct PageRank {
  pGraph<Graph>* g;

  void static go(pGraph<Graph>& _g) {
    galois::do_all(_g.g.begin(), _g.g.begin() + _g.numOwned, PageRank{&_g},
                   galois::loopname("Page Rank"));
    // Do commit
    galois::do_all(_g.g.begin(), _g.g.begin() + _g.numOwned,
                   [&](GNode src) {
                     _g.g.getData(src).swap_version(
                         BSP_FIELD_NAMES::PR_VAL_FIELD);
                   },
                   galois::loopname("PR-Commit"));
  }

  void operator()(GNode src) const {
    double sum   = 0;
    LNode& sdata = g->g.getData(src);
    for (auto jj = g->g.edge_begin(src), ej = g->g.edge_end(src); jj != ej;
         ++jj) {
      GNode dst    = g->g.getEdgeDst(jj);
      LNode& ddata = g->g.getData(dst);
      sum += ddata.value[ddata.current_version(BSP_FIELD_NAMES::PR_VAL_FIELD)] /
             ddata.nout;
    }
    float value = (1.0 - alpha) * sum + alpha;
    float diff  = std::fabs(
        value -
        sdata.value[sdata.current_version(BSP_FIELD_NAMES::PR_VAL_FIELD)]);
    sdata.value[sdata.next_version(BSP_FIELD_NAMES::PR_VAL_FIELD)] = value;
  }
};
/*********************************************************************************
 *
 **********************************************************************************/

// [hostid] -> vector of GID that host has replicas of
std::vector<std::vector<unsigned>> remoteReplicas;
// [hostid] -> remote pGraph Structure (locally invalid)
std::vector<pGraph<Graph>*> magicPointer;
/*********************************************************************************
 *
 **********************************************************************************/

void setRemotePtr(uint32_t hostID, pGraph<Graph>* p) {
  if (hostID >= magicPointer.size())
    magicPointer.resize(hostID + 1);
  magicPointer[hostID] = p;
}
/*********************************************************************************
 *
 **********************************************************************************/

void recvNodeStatic(unsigned GID, uint32_t hostID) {
  if (hostID >= remoteReplicas.size())
    remoteReplicas.resize(hostID + 1);
  remoteReplicas[hostID].push_back(GID);
}
/*********************************************************************************
 *
 **********************************************************************************/

void setNodeValue(pGraph<Graph>* p, unsigned GID, float v) {
  switch (personality) {
  case CPU:
    p->g.getData(p->G2L(GID))
        .value[p->g.getData(p->G2L(GID))
                   .current_version(BSP_FIELD_NAMES::PR_VAL_FIELD)] = v;
    break;
  case GPU_CUDA:
    setNodeValue_CUDA(cuda_ctx, p->G2L(GID), v);
    break;
  case GPU_OPENCL:
    cl_ctx.getData(p->G2L(GID))
        .value[p->g.getData(p->G2L(GID))
                   .current_version(BSP_FIELD_NAMES::PR_VAL_FIELD)] = v;
    break;
  default:
    break;
  }
}
/*********************************************************************************
 *
 **********************************************************************************/
// could be merged with setNodeValue, but this is one-time only ...
void setNodeAttr(pGraph<Graph>* p, unsigned GID, unsigned nout) {
  switch (personality) {
  case CPU:
    p->g.getData(p->G2L(GID)).nout = nout;
    break;
  case GPU_CUDA:
    setNodeAttr_CUDA(cuda_ctx, p->G2L(GID), nout);
    break;
  case GPU_OPENCL:
    cl_ctx.getData(p->G2L(GID)).nout = nout;
    break;
  default:
    break;
  }
}
/*********************************************************************************
 *
 **********************************************************************************/
// send values for nout calculated on my node
void setNodeAttr2(pGraph<Graph>* p, unsigned GID, unsigned nout) {
  auto LID = GID - p->g_offset;
  // printf("%d setNodeAttrs2 GID: %u nout: %u LID: %u\n", p->id, GID, nout,
  // LID);
  switch (personality) {
  case CPU:
    p->g.getData(LID).nout += nout;
    break;
  case GPU_CUDA:
    setNodeAttr2_CUDA(cuda_ctx, LID, nout);
    break;
  case GPU_OPENCL:
    cl_ctx.getData(LID).nout += nout;
    break;
  default:
    break;
  }
}
/*********************************************************************************
 *
 **********************************************************************************/
void sendGhostCellAttrs2(galois::runtime::NetworkInterface& net,
                         pGraph<Graph>& g) {
  for (auto n = g.g.begin() + g.numOwned; n != g.g.begin() + g.numNodes; ++n) {
    auto l2g_ndx = std::distance(g.g.begin(), n) - g.numOwned;
    auto x       = g.getHost(g.L2G[l2g_ndx]);
    // printf("%d: sendAttr2 GID: %d own: %d\n", g.id, g.L2G[l2g_ndx], x);
    switch (personality) {
    case CPU:
      net.sendAlt(x, setNodeAttr2, magicPointer[x], g.L2G[l2g_ndx],
                  g.g.getData(*n).nout);
      break;
    case GPU_CUDA:
      net.sendAlt(x, setNodeAttr2, magicPointer[x], g.L2G[l2g_ndx],
                  getNodeAttr2_CUDA(cuda_ctx, *n));
      break;
    case GPU_OPENCL:
      net.sendAlt(x, setNodeAttr2, magicPointer[x], g.L2G[l2g_ndx],
                  cl_ctx.getData(*n).nout);
      break;
    default:
      assert(false);
      break;
    }
  }
}

/*********************************************************************************
 *
 **********************************************************************************/

void sendGhostCellAttrs(galois::runtime::NetworkInterface& net,
                        pGraph<Graph>& g) {
  for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
    for (auto n : remoteReplicas[x]) {
      /* no per-personality needed but until nout is
       fixed for CPU and OpenCL ... */

      switch (personality) {
      case CPU:
        net.sendAlt(x, setNodeAttr, magicPointer[x], n,
                    g.g.getData(n - g.g_offset).nout);
        break;
      case GPU_CUDA:
        net.sendAlt(x, setNodeAttr, magicPointer[x], n,
                    getNodeAttr_CUDA(cuda_ctx, n - g.g_offset));
        break;
      case GPU_OPENCL:
        net.sendAlt(x, setNodeAttr, magicPointer[x], n,
                    cl_ctx.getData(n - g.g_offset).nout);
        break;
      default:
        assert(false);
        break;
      }
    }
  }
}
/*********************************************************************************
 * Send ghost-cell updates to all hosts that require it. Go over all the
 *remotereplica arrays, and for each array, go over all the elements and send it
 *to the host 'x'. Note that we use the magicPointer array to obtain the
 *reference of the graph object where the node data is to be set.
 **********************************************************************************/
void sendGhostCells(galois::runtime::NetworkInterface& net, pGraph<Graph>& g) {
  for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
    for (auto n : remoteReplicas[x]) {
      switch (personality) {
      case CPU:
        net.sendAlt(
            x, setNodeValue, magicPointer[x], n,
            g.g.getData(n - g.g_offset)
                .value[g.g.getData(n - g.g_offset)
                           .current_version(BSP_FIELD_NAMES::PR_VAL_FIELD)]);
        break;
      case GPU_CUDA:
        net.sendAlt(x, setNodeValue, magicPointer[x], n,
                    getNodeValue_CUDA(cuda_ctx, n - g.g_offset));
        break;
      case GPU_OPENCL:
        net.sendAlt(
            x, setNodeValue, magicPointer[x], n,
            cl_ctx.getData((n - g.g_offset))
                .value[g.g.getData(n - g.g_offset)
                           .current_version(BSP_FIELD_NAMES::PR_VAL_FIELD)]);
        break;
      default:
        assert(false);
        break;
      }
    }
  }
}
/*********************************************************************************
 *
 **********************************************************************************/

MarshalGraph pGraph2MGraph(pGraph<Graph>& g) {
  MarshalGraph m;

  m.nnodes    = g.numNodes;
  m.nedges    = g.numEdges;
  m.nowned    = g.numOwned;
  m.g_offset  = g.g_offset;
  m.id        = g.id;
  m.row_start = (index_type*)calloc(m.nnodes + 1, sizeof(index_type));
  m.edge_dst  = (index_type*)calloc(m.nedges, sizeof(index_type));

  // TODO: initialize node_data and edge_data
  m.node_data = NULL;
  m.edge_data = NULL;

  // pinched from Rashid's LC_LinearArray_Graph.h

  size_t edge_counter = 0, node_counter = 0;
  for (auto n = g.g.begin(); n != g.g.end() && *n != m.nnodes;
       n++, node_counter++) {
    m.row_start[node_counter] = edge_counter;
    if (*n < g.numOwned) {
      for (auto e = g.g.edge_begin(*n); e != g.g.edge_end(*n); e++) {
        if (g.g.getEdgeDst(e) < g.numNodes)
          m.edge_dst[edge_counter++] = g.g.getEdgeDst(e);
      }
    }
  }

  m.row_start[node_counter] = edge_counter;
  m.nedges                  = edge_counter;
  return m;
}
/*********************************************************************************
 *
 **********************************************************************************/

void loadGraphNonCPU(pGraph<Graph>& g) {
  MarshalGraph m;
  assert(personality != CPU);
  switch (personality) {
  case GPU_CUDA:
    m = pGraph2MGraph(g);
    load_graph_CUDA(cuda_ctx, m);
    break;
  case GPU_OPENCL:
    cl_ctx.loadGraphNonCPU(g.g, g.numOwned, g.numEdges,
                           g.numNodes - g.numOwned);
    break;
  default:
    assert(false);
    break;
  }
  // TODO cleanup marshalgraph, leaks memory!
}

/*********************************************************************************
 *
 **********************************************************************************/
void inner_main() {
  auto& net = galois::runtime::getSystemNetworkInterface();
  galois::StatManager statManager;
  auto& barrier             = galois::runtime::getSystemBarrier();
  const unsigned my_host_id = galois::runtime::NetworkInterface::ID;
  galois::Timer T_total, T_graph_load, T_pagerank, T_graph_init,
      T_postSyncKernel;
  T_total.start();
  // Parse arg string when running on multiple hosts and update/override
  // personality with corresponding value.
  if (personality_set.length() == galois::runtime::NetworkInterface::Num) {
    switch (personality_set.c_str()[galois::runtime::NetworkInterface::ID]) {
    case 'g':
      personality = GPU_CUDA;
      break;
    case 'o':
      personality = GPU_OPENCL;
      break;
    case 'c':
    default:
      personality = CPU;
      break;
    }
  }
  fprintf(stderr, "Pre-barrier - Host: %d, Personality %s\n",
          galois::runtime::NetworkInterface::ID,
          personality_str(personality).c_str());
  barrier.wait();
  fprintf(stderr, "Post-barrier - Host: %d, Personality %s\n",
          galois::runtime::NetworkInterface::ID,
          personality_str(personality).c_str());
  //   Graph rg;
  T_graph_load.start();
  pGraph<Graph> g;
  g.loadGraph(inputFile);

  if (personality == GPU_CUDA) {
    cuda_ctx = get_CUDA_context(galois::runtime::NetworkInterface::ID);
    if (!init_CUDA_context(cuda_ctx, gpudevice))
      return;
  } else if (personality == GPU_OPENCL) {
    galois::opencl::cl_env.init(cldevice);
  }

  if (personality != CPU)
    loadGraphNonCPU(g);
#if _HETERO_DEBUG_
  std::cout << g.id << " graph loaded\n";
#endif

  T_graph_load.stop();

  T_graph_init.start();

  // local initialization
  if (personality == CPU) {
    InitializeGraph::go(g);
  } else if (personality == GPU_CUDA) {
    initialize_graph_cuda(cuda_ctx);
  } else if (personality == GPU_OPENCL) {
    cl_ctx.init(g.numOwned, g.numNodes);
  }

#if _HETERO_DEBUG_
  std::cout << g.id << " initialized\n";
#endif
  barrier.wait();

  // send pGraph pointers
  for (uint32_t x = 0; x < galois::runtime::NetworkInterface::Num; ++x)
    net.sendAlt(x, setRemotePtr, galois::runtime::NetworkInterface::ID, &g);

  // Ask for cells
  for (auto GID : g.L2G)
    net.sendAlt(g.getHost(GID), recvNodeStatic, GID,
                galois::runtime::NetworkInterface::ID);
#if _HETERO_DEBUG_
  std::cout << "[" << my_host_id << "]:ask for remote replicas\n";
#endif
  barrier.wait();

  // send out partial contributions for nout from local -> ghost
  sendGhostCellAttrs2(net, g);
  barrier.wait();

  // send final nout values to remote replicas
#if _HETERO_DEBUG_
  std::cout << "[" << my_host_id << "]:ask for ghost cell attrs\n";
#endif
  sendGhostCellAttrs(net, g);
  barrier.wait();
  T_graph_init.stop();

  std::cout << "[" << my_host_id << "] Starting PageRank"
            << "\n";
  T_pagerank.start();
  T_postSyncKernel.start();
  for (int i = 0; i < maxIterations; ++i) {
#if _HETERO_DEBUG_
    std::cout << "Starting " << i << "\n";
#endif
    // communicate ghost cells
    sendGhostCells(net, g);
    barrier.wait();
#if _HETERO_DEBUG_
    std::cout << "Starting PR\n";
#endif
    // Do pagerank
    switch (personality) {
    case CPU:
      PageRank::go(g);
      //         WriteBack::go(g);
      break;
    case GPU_OPENCL:
      cl_ctx(g.numOwned);
      break;
    case GPU_CUDA:
      pagerank_cuda(cuda_ctx);
      break;
    default:
      break;
    }
    barrier.wait();
  }

  T_pagerank.stop();
  // Final synchronization to ensure that all the nodes are updated.
  sendGhostCells(net, g);
  barrier.wait();
  T_postSyncKernel.stop();

  if (verify) {
    std::stringstream ss;
    ss << personality_str(personality) << "_" << my_host_id << "_of_"
       << galois::runtime::NetworkInterface::Num << "_page_ranks.csv";
    std::ofstream out_file(ss.str());
    switch (personality) {
    case CPU: {
      for (auto n = g.g.begin(); n != g.g.begin() + g.numOwned; ++n) {
        out_file << *n + g.g_offset << ", "
                 << g.g.getData(*n).value[g.g.getData(*n).current_version(
                        BSP_FIELD_NAMES::PR_VAL_FIELD)]
                 << ", " << g.g.getData(*n).nout << "\n";
      }
      break;
    }
    case GPU_OPENCL: {
      for (int n = 0; n < g.numOwned; ++n) {
        out_file << n + g.g_offset << ", "
                 << cl_ctx.getData(n).value[cl_ctx.getData(n).current_version(
                        BSP_FIELD_NAMES::PR_VAL_FIELD)]
                 << ", " << cl_ctx.getData(n).nout << "\n";
      }
      break;
    }
    case GPU_CUDA:
      for (int n = 0; n < g.numOwned; n++) {
        out_file << n + g.g_offset << ", " << getNodeValue_CUDA(cuda_ctx, n)
                 << ", " << getNodeAttr_CUDA(cuda_ctx, n) << "\n";
      }
      break;
    }
    out_file.close();
  }
  T_total.stop();
  std::cout << "[" << galois::runtime::NetworkInterface::ID << "]"
            << " Total : " << T_total.get()
            << " Loading : " << T_graph_load.get()
            << " Init : " << T_graph_init.get() << " PageRank ("
            << maxIterations << " iteration) : " << T_pagerank.get()
            << " (msec)\n";
  //   if(my_host_id == 0)
  {
    fprintf(stderr, "HEADER2,filename,TotalTime,Loadtime,InitTime,KernelTime, "
                    "KernelPostSyncTime\n");
    fprintf(stderr, "STAT2,%s,%ld,%ld,%ld,%ld,%ld\n",
            inputFile.getValue().c_str(), T_total.get(), T_graph_load.get(),
            T_graph_init.get(), T_pagerank.get(), T_postSyncKernel.get());
  }
  std::cout << "Terminated on [ " << my_host_id << " ]\n";
  net.terminate();
  std::cout.flush();
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  // auto& net = galois::runtime::getSystemNetworkInterface();
  inner_main();
  return 0;
}
