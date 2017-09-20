/** SSSP application -*- C++ -*-
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
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Graphs/FileGraph.h"
#include "galois/Graphs/LC_CSR_Graph.h"
#include "galois/Graphs/Util.h"
#include "Lonestar/BoilerPlate.h"
#include "PGraph.h"
#include "OpenCL/LC_LinearArray_Graph.h"
#include "OpenCL/CLWrapper.h"

#include <iostream>
#include <typeinfo>
#include <algorithm>

//#include "CL_Header.h"
#include "opencl/CLSSSP.h"

#define _HETERO_DEBUG_ 0

static const char* const name_short = "HSSSP";
static const char* const name = "SSSP - Distributed Heterogeneous";
static const char* const desc = "Bellman-Ford SSSP on Distributed Galois.";
static const char* const url = 0;

enum Personality {
   CPU, GPU_CUDA, GPU_OPENCL
};
std::string personality_str(Personality p) {
   switch (p) {
   case CPU:
      return "CPU";
   case GPU_CUDA:
      return "GPU_CUDA";
   case GPU_OPENCL:
      return "GPU_OPENCL";
   }
   assert(false&& "Invalid personality");
   return "";
}

namespace cll = llvm::cl;
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), clEnumValEnd),
      cll::init(CPU));
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file (transpose)>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(4));
static cll::opt<unsigned int> src_node("srcNodeId", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<float> cldevice("cldevice", cll::desc("Select OpenCL device to run on , default is 0.0 (OpenCL backend)"), cll::init(0.0));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));
////////////////////////////////////////////
enum BSP_FIELD_NAMES {
   SSSP_DIST_FIELD = 0
};
struct NodeData {
   int bsp_version;
   int dist[2];/*ID=0*/
   void swap_version(unsigned int field_id) {
      bsp_version ^= 1 << field_id;
   }
   unsigned current_version(unsigned int field_id) {
      return (bsp_version & (1 << field_id)) != 0;
   }
   unsigned next_version(unsigned int field_id) {
      return (~bsp_version & (1 << field_id)) != 0;
   }
};
////////////////////////////////////////////
typedef NodeData NodeDataType;
typedef galois::graphs::LC_CSR_Graph<NodeDataType, unsigned int> Graph;
typedef pGraph<Graph> PGraph;
typedef typename Graph::GraphNode GNode;
bool hasChanged = false;

//////////////////////////////////////////////////////////////////////////////////////
//struct CUDA_Context *cuda_ctx;
typedef galois::OpenCL::LC_LinearArray_Graph<galois::OpenCL::Array, NodeData, unsigned int> DeviceGraph;
struct OPENCL_Context<DeviceGraph> dOp;
/*********************************************************************************
 *
 **********************************************************************************/
struct InitializeGraph {
   Graph* g;
   void static go(Graph& _g, unsigned num) {
      galois::do_all(_g.begin(), _g.begin() + num, InitializeGraph { &_g }, galois::loopname("init"));
   }
   void operator()(GNode src) const {
      NodeDataType & sdata = g->getData(src);
      //TODO RK : Fix this to not initialize both fields.
      sdata.dist[0] = sdata.dist[1] = std::numeric_limits<int>::max() / 2;
   }
};
/*********************************************************************************
 * CPU PageRank operator implementation.
 **********************************************************************************/
struct PrintNodes {
   Graph * g;
   void static go(Graph& _g, unsigned num) {
      galois::do_all(_g.begin(), _g.begin() + num, PrintNodes { &_g }, galois::loopname("PrintNodes"));
   }
   void operator()(GNode src) const {
      NodeDataType & sdata = g->getData(src);
      fprintf(stderr, "GraphPrint:: %d [%d, %d]\n", src, sdata.dist[0], sdata.dist[1]);
   }
};
/************************************************************
 *
 *************************************************************/
struct SSSP {
   Graph* g;

   void static go(Graph& _g, unsigned num) {
      //Perform computation
      galois::do_all(_g.begin(), _g.begin() + num, SSSP { &_g }, galois::loopname("SSSP"));
      //Do commit
      galois::do_all(_g.begin(), _g.begin() + num, [&](GNode src){_g.getData(src).swap_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD);}, galois::loopname("SSSP-Commit"));
   }

   void operator()(GNode src) const {
      NodeDataType& sdata = g->getData(src);
      int min_dist = sdata.dist[sdata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)];
      for (auto jj = g->edge_begin(src), ej = g->edge_end(src); jj != ej; ++jj) {
         GNode dst = g->getEdgeDst(jj);
         NodeDataType & ddata = g->getData(dst);
         int ddst = ddata.dist[ddata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)];
         int ewt = g->getEdgeData(jj);
         min_dist = std::min(min_dist, ewt + ddst);
//         fprintf(stderr, "Edge::[src: %d, dst: %d, ewt: %d [old: %d -> new: %d] ]\n", src, dst, ewt, sdata.dist[current(current_phase)], min_dist);

      }
      if (min_dist < sdata.dist[sdata.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]) {
         sdata.dist[sdata.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = min_dist;
         hasChanged = true;
      }

   }
};
/*********************************************************************************
 *
 **********************************************************************************/
// [hostid] -> vector of GID that host has replicas of
std::vector<std::vector<unsigned> > remoteReplicas;
// [hostid] -> remote pGraph Structure (locally invalid)
std::vector<PGraph*> magicPointer;
/*********************************************************************************
 *
 **********************************************************************************/

void setRemotePtr(uint32_t hostID, PGraph* p) {
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

void setNodeValue(PGraph* p, unsigned GID, int v) {
   if (p->g.getData(p->G2L(GID)).dist[p->g.getData(p->G2L(GID)).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] > v) {
//      fprintf(stderr, "Lowered-ghost[%d, %d->%d]", p->G2L(GID), p->g.getData(p->G2L(GID)),v);
   }
//   else{
//      return;
//   }
   switch (personality) {
   case CPU:
      p->g.getData(p->G2L(GID)).dist[p->g.getData(p->G2L(GID)).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = v;
      break;
   case GPU_CUDA:
//      setNodeValue_CUDA(cuda_ctx, p->G2L(GID), v);
      break;
   case GPU_OPENCL:
      dOp.getData(p->G2L(GID)).dist[dOp.getData(p->G2L(GID)).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = v;
      break;
   default:
      break;
   }
}
/*********************************************************************************
 * Send ghost-cell updates to all hosts that require it. Go over all the remote replica
 * arrays, and for each array, go over all the elements and send it to the host 'x'.
 * Note that we use the magicPointer array to obtain the reference of the graph object
 * where the node data is to be set.
 **********************************************************************************/

void sendGhostCells(galois::runtime::NetworkInterface& net, PGraph& g) {
   for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
      for (auto n : remoteReplicas[x]) {
         switch (personality) {
         case CPU:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, g.g.getData(n - g.g_offset).dist[g.g.getData(n - g.g_offset).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
            break;
         case GPU_CUDA:
//            net.sendAlt(x, setNodeValue, magicPointer[x], n, (int) getNodeValue_CUDA(cuda_ctx, n - g.g_offset));
            break;
         case GPU_OPENCL:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, dOp.getData(n - g.g_offset).dist[dOp.getData(n - g.g_offset).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
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
/*

MarshalGraph pGraph2MGraph(PGraph &g) {
   MarshalGraph m;

   m.nnodes = g.numNodes;
   m.nedges = g.numEdges;
   m.nowned = g.numOwned;
   m.g_offset = g.g_offset;
   m.id = g.id;
   m.row_start = (index_type *) calloc(m.nnodes + 1, sizeof(index_type));
   m.edge_dst = (index_type *) calloc(m.nedges, sizeof(index_type));

   // TODO: initialize node_data and edge_data
   m.node_data = NULL;
   m.edge_data = NULL;

   // pinched from Rashid's LC_LinearArray_Graph.h

   size_t edge_counter = 0, node_counter = 0;
   for (auto n = g.g.begin(); n != g.g.end() && *n != m.nnodes; n++, node_counter++) {
      m.row_start[node_counter] = edge_counter;
      if (*n < g.numOwned) {
         for (auto e = g.g.edge_begin(*n); e != g.g.edge_end(*n); e++) {
            if (g.g.getEdgeDst(e) < g.numNodes)
               m.edge_dst[edge_counter++] = g.g.getEdgeDst(e);
         }
      }
   }

   m.row_start[node_counter] = edge_counter;
   m.nedges = edge_counter;
   return m;
}
*/
/*********************************************************************************
 *
 **********************************************************************************/

void loadGraphNonCPU(PGraph &g) {
//   MarshalGraph m;
   assert(personality != CPU);
   switch (personality) {
   case GPU_CUDA:
//      m = pGraph2MGraph(g);
//      load_graph_CUDA(cuda_ctx, m);
      break;
   case GPU_OPENCL:
//      dOp.loadGraphNonCPU(g.g, g.numOwned, g.numEdges, g.numNodes - g.numOwned);
      dOp.loadGraphNonCPU(g);
      break;
   default:
      assert(false);
      break;
   }
   // TODO cleanup marshalgraph, leaks memory!
}

void recvChangeFlag(bool otherFlag) {
   hasChanged |= otherFlag;
}
void sendChangeFlag(galois::runtime::NetworkInterface & net) {
   for (uint32_t x = 0; x < galois::runtime::NetworkInterface::Num; ++x)
      net.sendAlt(x, recvChangeFlag, hasChanged);
}
/*********************************************************************************
 *
 **********************************************************************************/
void inner_main() {
   auto& net = galois::runtime::getSystemNetworkInterface();
   galois::StatManager statManager;
   galois::Timer alg_timer, kernel_timer, init_timer;
   float alg_time=0, kernel_time=0, init_time=0;

   auto& barrier = galois::runtime::getHostBarrier() ;//getSystemBarrier();
   const unsigned my_host_id = galois::runtime::NetworkInterface::ID;
   //Parse arg string when running on multiple hosts and update/override personality
   //with corresponding value.
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
   barrier.wait();
   init_timer.start();
   alg_timer.start();
   PGraph g;
   g.loadGraph(inputFile);

   if (personality == GPU_CUDA) {
//      cuda_ctx = get_CUDA_context(galois::runtime::NetworkInterface::ID);
//      if (!init_CUDA_context(cuda_ctx, gpudevice))
//         return;
   } else if (personality == GPU_OPENCL) {
      galois::OpenCL::cl_env.init(cldevice.Value);
   }
   if (personality != CPU)
      loadGraphNonCPU(g);
#if _HETERO_DEBUG_
   std::cout << g.id << " graph loaded\n";
#endif
//   dSSSP dOp;
//   OPENCL_Context<Graph> dOp;

   //local initialization
   if (personality == CPU) {
      InitializeGraph::go(g.g, g.numNodes);
      if (g.getHost((src_node.Value)) == galois::runtime::NetworkInterface::ID) {
         fprintf(stderr, "Initialized src %d to zero at part %d\n", src_node.Value, galois::runtime::NetworkInterface::ID);
         NodeData & ndata = g.g.getData(src_node);
         ndata.dist[ndata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = 0;
         ndata.dist[ndata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = 0;
      }
   } else if (personality == GPU_CUDA) {
//      initialize_graph_cuda(cuda_ctx);
   } else if (personality == GPU_OPENCL) {
      dOp.init(g.numOwned, g.numNodes);
      if (g.getHost((src_node.Value)) == galois::runtime::NetworkInterface::ID) {
         fprintf(stderr, "Initialized src %d to zero at part %d\n", src_node.Value, galois::runtime::NetworkInterface::ID);
         dOp.getData(src_node).dist[dOp.getData(src_node).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = 0;
      }
   }
#if _HETERO_DEBUG_
   std::cout << g.id << " initialized\n";
#endif
   barrier.wait();
   init_timer.stop();
   init_time = init_timer.get();
   //send pGraph pointers
   for (uint32_t x = 0; x < galois::runtime::NetworkInterface::Num; ++x)
      net.sendAlt(x, setRemotePtr, galois::runtime::NetworkInterface::ID, &g);

   //Ask for cells
   for (auto GID : g.L2G)
      net.sendAlt(g.getHost(GID), recvNodeStatic, GID, galois::runtime::NetworkInterface::ID);
#if _HETERO_DEBUG_
   std::cout << "["<<my_host_id<< "]:ask for remote replicas\n";
#endif
   barrier.wait();
   // send final nout values to remote replicas
#if _HETERO_DEBUG_
   std::cout << "["<<my_host_id<< "]:ask for ghost cell attrs\n";
#endif

   kernel_timer.start();
   for (int i = 0; i < maxIterations; ++i) {
      //communicate ghost cells
      sendGhostCells(net, g);
      barrier.wait();
      //Do pagerank
      switch (personality) {
      case CPU:
         SSSP::go(g.g, g.numOwned);
//         PrintNodes::go(g.g, g.numOwned);
         break;
      case GPU_OPENCL:
         dOp(g.numOwned, hasChanged);
         break;
      case GPU_CUDA:
//         pagerank_cuda(cuda_ctx);
         break;
      default:
         break;
      }
      sendChangeFlag(net);
      barrier();
      if (!hasChanged) {
         fprintf(stderr, "Terminating after %d steps\n", i);
         break;
      }
      hasChanged = false;
   }
   kernel_timer.stop();
   kernel_time = kernel_timer.get();
//   if (hasChanged)
      fprintf(stderr, "Terminating after max=%d steps\n", maxIterations.Value);
   //Final synchronization to ensure that all the nodes are updated.
   sendGhostCells(net, g);
   barrier.wait();
   alg_timer.stop();
   alg_time = alg_timer.get();
////////////////////////////////////////////////////////////////////////////////////////////////////////////
   if (verify) {
      std::stringstream ss;
      ss << personality_str(personality) << "_" << my_host_id << "_of_" << galois::runtime::NetworkInterface::Num << "_distances.csv";
      std::ofstream out_file(ss.str());
      switch (personality) {
      case CPU: {
         int max_dist = 0;
         int max_dist_next = 0;
         for (auto n = g.g.begin(); n != g.g.begin() + g.numOwned; ++n) {
            NodeData & ndata = g.g.getData(*n);
            max_dist = std::max(max_dist, ndata.dist[ndata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
            max_dist_next = std::max(max_dist, ndata.dist[ndata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
            out_file << *n + g.g_offset << ", " << ndata.dist[ndata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] << ","
                  << ndata.dist[ndata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] << "\n";
         }
         fprintf(stderr, "Maxdist - %d, Next - %d \n", max_dist, max_dist_next);
         break;
      }
      case GPU_OPENCL: {
         int max_dist = 0;
         for (int n = 0; n < g.numOwned; ++n) {
            max_dist = std::max(max_dist, dOp.getData(n).dist[dOp.getData(n).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
            out_file << n + g.g_offset << ", " << dOp.getData(n).dist[dOp.getData(n).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] << "\n";
         }
         fprintf(stderr, "Maxdist - %d\n", max_dist);
//         dGraph.print_graph();
         break;
      }
      case GPU_CUDA:
         /*for (int n = 0; n < g.numOwned; n++) {
            out_file << n + g.g_offset << ", " << getNodeValue_CUDA(cuda_ctx, n) << ", " << getNodeAttr_CUDA(cuda_ctx, n) << "\n";
         }*/
         break;
      }
      out_file.close();

   }
   fprintf(stderr, "HEADER2,app,input,pset,id,alg_time, init_time, kernel_time\n");
   fprintf(stderr, "STAT2,%s,%s,%s,%d,%6.6g,%6.6g,%6.6g\n", name_short, inputFile.c_str(),personality_set.c_str(), my_host_id,
         alg_time, init_time, kernel_time);

   std::cout.flush();

}
/********************************************************************************************
 *
 ********************************************************************************************/
int main(int argc, char** argv) {
   LonestarStart(argc, argv, name, desc, url);
   galois::OpenCL::CLContext * ctx = galois::OpenCL::getCLContext();
   auto& net = galois::runtime::getSystemNetworkInterface();
   inner_main();
   return 0;
}
