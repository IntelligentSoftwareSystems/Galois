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

#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/LC_CSR_Graph.h"
#include "Galois/Graph/Util.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Bag.h"
#include "PGraph.h"
#include "opencl/LC_LinearArray_Graph.h"
#include "cuda/hsssp_cuda.h"
#include "cuda/cuda_mtypes.h"
#include "opencl/CLWrapper.h"

#include <iostream>
#include <typeinfo>
#include <algorithm>
#include "opencl/CLSSSP.h"

#define _HETERO_DEBUG_ 0

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
   SSSP_DIST_FIELD = 0, WORKLIST = 1
};
/*****************************************************************
 *
 *****************************************************************/
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
/*****************************************************************
 *
 *****************************************************************/
struct BSPWorklist {
   //TODO RK : Fix template types to use GNode instead.
   typedef typename Galois::InsertBag<unsigned> WLType;
   WLType bags[2];
   int bsp_version;
   void swap_version(unsigned int field_id) {
      bsp_version ^= 1 << field_id;
   }
   unsigned current_version(unsigned int field_id) {
      return (bsp_version & (1 << field_id)) != 0;
   }
   unsigned next_version(unsigned int field_id) {
      return (~bsp_version & (1 << field_id)) != 0;
   }
   WLType & current() {
      return bags[current_version(BSP_FIELD_NAMES::WORKLIST)];
   }
   WLType & next() {
      return bags[next_version(BSP_FIELD_NAMES::WORKLIST)];
   }

} worklist;
std::vector<BSPWorklist *> remote_wl;
/*****************************************************************
 *
 *****************************************************************/
typedef NodeData NodeDataType;
typedef Galois::Graph::LC_CSR_Graph<NodeDataType, unsigned int> Graph;
typedef pGraph<Graph> PGraph;
typedef typename Graph::GraphNode GNode;
bool hasChanged = false;

//////////////////////////////////////////////////////////////////////////////////////
struct CUDA_Context *cuda_ctx;
typedef Galois::OpenCL::LC_LinearArray_Graph<Galois::OpenCL::Array, NodeData, unsigned int> DeviceGraph;
struct OPENCL_Context<DeviceGraph> dOp;
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
struct InitializeGraph {
   Graph* g;
   void static go(Graph& _g, unsigned num) {
      Galois::do_all(_g.begin(), _g.begin() + num, InitializeGraph { &_g }, Galois::loopname("init"));
   }
   void operator()(GNode src) const {
      NodeDataType & sdata = g->getData(src);
      //TODO RK : Fix this to not initialize both fields.
      sdata.dist[0] = sdata.dist[1] = std::numeric_limits<int>::max() / 4;
      sdata.bsp_version = 0;
   }
};
/*********************************************************************************
 * CPU PageRank operator implementation.
 **********************************************************************************/
struct PrintNodes {
   PGraph * g;
   void static go(PGraph& _g, unsigned num) {
      Galois::do_all(_g.g.begin(), _g.g.begin() + num, PrintNodes { &_g }, Galois::loopname("PrintNodes"));
   }
   void operator()(GNode src) const {
      NodeDataType & sdata = g->g.getData(src);
      fprintf(stderr, "GraphPrint:: %d [curr=%d, next=%d]\n", g->local2Global(src), sdata.dist[sdata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)],
            sdata.dist[sdata.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
   }
};
/************************************************************
 *
 *************************************************************/
void setNodeValue(PGraph* p, unsigned GID, int v) {
   switch (personality) {
   case CPU: {
      NodeData & nd = p->g.getData(p->G2L(GID));
      nd.dist[nd.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = std::min(v, nd.dist[nd.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
      nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = std::min(v, nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
   }
      break;
   case GPU_CUDA:
      setNodeValue_CUDA(cuda_ctx, p->G2L(GID), v);
      break;
   case GPU_OPENCL:
      dOp.getData(p->G2L(GID)).dist[dOp.getData(p->G2L(GID)).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = v;
      break;
   default:
      break;
   }
}

/*********************************************************************************
 *
 **********************************************************************************/

/*********************************************************************************
 * Send ghost-cell updates to all hosts that require it. Go over all the remote replica
 * arrays, and for each array, go over all the elements and send it to the host 'x'.
 * Note that we use the magicPointer array to obtain the reference of the graph object
 * where the node data is to be set.
 **********************************************************************************/

void sync_distances(Galois::Runtime::NetworkInterface& net, PGraph& g) {
   for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
      for (auto n : remoteReplicas[x]) {
         auto lid = g.G2L(n);
         switch (personality) {
         case CPU: {
            NodeData & nd = g.g.getData(lid);
            net.sendAlt(x, setNodeValue, magicPointer[x], n, nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
         }
            break;
         case GPU_CUDA:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, (int) getNodeValue_CUDA(cuda_ctx, n - g.g_offset));
            break;
         case GPU_OPENCL: {
            NodeData & nd = dOp.getData(lid);
            net.sendAlt(x, setNodeValue, magicPointer[x], n, nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
         }
            break;
         default:
            assert(false);
            break;
         }
      }
   }
}
/****************************************************************************************
 *
 ****************************************************************************************/
void setNextDistance(PGraph* p, unsigned GID, int v) {
   switch (personality) {
   case CPU: {
      NodeData & nd = p->g.getData(p->G2L(GID));
      nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = std::min(v, nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
   }
      break;
   case GPU_CUDA:
      setNodeValue_CUDA(cuda_ctx, p->G2L(GID), v);
      break;
   case GPU_OPENCL: {
      NodeData & nd = dOp.getData(p->G2L(GID));
      nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = std::min(v, nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
   }
      break;
   default:
      break;
   }
}
/****************************************************************************************
 *
 ****************************************************************************************/
void sendNextDistances(Galois::Runtime::NetworkInterface& net, PGraph& g) {
   for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
      for (auto n : remoteReplicas[x]) {
         switch (personality) {
         case CPU: {
            NodeData & nd = g.g.getData(n - g.g_offset);
            net.sendAlt(x, setNextDistance, magicPointer[x], n, nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
         }
            break;
         case GPU_CUDA:
            net.sendAlt(x, setNextDistance, magicPointer[x], n, (int) getNodeValue_CUDA(cuda_ctx, n - g.g_offset));
            break;
         case GPU_OPENCL: {
            NodeData & nd = dOp.getData(n - g.g_offset);
            net.sendAlt(x, setNextDistance, magicPointer[x], n, nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)]);
         }
            break;
         default:
            assert(false);
            break;
         }
      }
   }
}
/****************************************************************************************
 * Send ghost-cells to owner nodes.
 ****************************************************************************************/
void sendGhostCellDistances(Galois::Runtime::NetworkInterface& net, PGraph & g) {
   for (auto n = g.ghost_begin(); n != g.ghost_end(); ++n) {
      auto l2g_ndx = std::distance(g.g.begin(), n) - g.numOwned;
      auto x = g.getHost(g.L2G[l2g_ndx]);
      switch (personality) {
      case CPU: {
         NodeData & nd = g.g.getData(*n);
         int msg = nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)];
         net.sendAlt(x, setNextDistance, magicPointer[x], g.L2G[l2g_ndx], msg);
      }
         break;
      case GPU_CUDA:
//         net.sendAlt(x, setNextDistance, magicPointer[x], g.L2G[l2g_ndx], getNodeAttr2_CUDA(cuda_ctx, *n));
         break;
      case GPU_OPENCL: {
         NodeData & nd = dOp.getData(*n);
         int msg = nd.dist[nd.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)];
         net.sendAlt(x, setNextDistance, magicPointer[x], g.L2G[l2g_ndx], msg);
      }
         break;
      default:
         assert(false);
         break;
      }
   }
}
/****************************************************************************************
 *
 ****************************************************************************************/
struct SSSP_PUSH_Commit {
   Graph * g;
   void operator()(GNode src) const {
      g->getData(src).dist[0] = g->getData(src).dist[1] = std::min(g->getData(src).dist[0], g->getData(src).dist[1]);
      g->getData(src).swap_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD);
   }
};
/****************************************************************************************
 *
 ****************************************************************************************/
void pushToWorklist(BSPWorklist * wl, PGraph * g, unsigned item) {
   const unsigned id = Galois::Runtime::NetworkInterface::ID;
   unsigned local_id = g->G2L(item);
//   fprintf(stderr, "[%d] push_to_worklist:: -> local_id:%d, gid:%d\n", id, local_id, item);
   wl->current().push(local_id);
}
struct SSSP {
   Graph* g;
   void static go(PGraph& _g, unsigned num) {
      auto &net = Galois::Runtime::getSystemNetworkInterface();
      const unsigned id = Galois::Runtime::NetworkInterface::ID;

      //Perform computation
//      Galois::do_all(_g.begin(), _g.end(), SSSP { &_g.g }, Galois::loopname("SSSP"));
      Galois::do_all(worklist.current().begin(), worklist.current().end(), SSSP { &_g.g }, Galois::loopname("SSSP"));
      if (false) {
         fprintf(stderr, "[%d]Worklist::[", id);
         for (auto it = worklist.next().begin(); it != worklist.next().end(); ++it) {
            fprintf(stderr, "%d", *it);
         }
         fprintf(stderr, "]\n");
      }
      /////////////////////////////
      Galois::Runtime::getSystemBarrier()();
      sendGhostCellDistances(net, _g);
      sync_distances(Galois::Runtime::getSystemNetworkInterface(), _g);
      Galois::Runtime::getSystemBarrier()();
      Galois::do_all(_g.begin(), _g.end(), SSSP_PUSH_Commit { &_g.g }, Galois::loopname("SSSP-Commit"));
      /*fprintf(stderr, "[%d]WORKLIST-Size::[%u,%u]\n", id, std::distance(worklist.current().begin(), worklist.current().end()),
       std::distance(worklist.next().begin(), worklist.next().end()));*/
      //Sync. worklists
      {
         worklist.current().clear();
         for (auto it = worklist.next().begin(); it != worklist.next().end(); ++it) {
            auto gid = _g.local2Global(*it);
            auto owner = _g.getHost(gid);
//            fprintf(stderr, "[%d] worklist filter:: -> item:%d, owner:%d, gid:%d\n", id, *it, owner, gid);
            if (owner != id) {
               net.sendAlt(owner, pushToWorklist, remote_wl[owner], magicPointer[owner], gid);
            } else {
               pushToWorklist(&worklist, &_g, gid);
            }
         }
         worklist.next().clear();
      }
      Galois::Runtime::getSystemBarrier()();
//      worklist.swap_version(BSP_FIELD_NAMES::WORKLIST);
//      worklist.next().clear();
   }
   void operator()(GNode src) const {
//      GNode src = g->begin()+_src;
      const unsigned id = Galois::Runtime::NetworkInterface::ID;
      NodeDataType& sdata = g->getData(src);
      int sdist = sdata.dist[sdata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)];
      for (auto jj = g->edge_begin(src), ej = g->edge_end(src); jj != ej; ++jj) {
         GNode dst = g->getEdgeDst(jj);
         NodeDataType & ddata = g->getData(dst);
         int *ddst = &ddata.dist[ddata.next_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)];
         int old_dist = *ddst;
         int new_dist = g->getEdgeData(jj) + sdist;
         while (new_dist < (old_dist = *ddst)) {
            if (__sync_bool_compare_and_swap(ddst, old_dist, new_dist)) {
//               fprintf(stderr, "[%d] Updated [%d,%d, from:%d to:%d]\n", id, src, dst, old_dist, new_dist);
               worklist.next().push(dst);
               hasChanged = true;
               break;
            }
         }
      }
   }
};
/*********************************************************************************
 *
 **********************************************************************************/
void setRemoteWorklist(uint32_t hostID, BSPWorklist * wl) {
   if (hostID >= remote_wl.size())
      remote_wl.resize(hostID + 1);
   remote_wl[hostID] = wl;
}
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
/*********************************************************************************
 *
 **********************************************************************************/

void loadGraphNonCPU(PGraph &g) {
   MarshalGraph m;
   assert(personality != CPU);
   switch (personality) {
   case GPU_CUDA:
      m = pGraph2MGraph(g);
      load_graph_CUDA(cuda_ctx, m);
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
void sendChangeFlag(Galois::Runtime::NetworkInterface & net) {
   for (uint32_t x = 0; x < Galois::Runtime::NetworkInterface::Num; ++x)
      net.sendAlt(x, recvChangeFlag, hasChanged);
}
/*********************************************************************************
 *
 **********************************************************************************/
void inner_main() {
   auto& net = Galois::Runtime::getSystemNetworkInterface();
   Galois::StatManager statManager;
   auto& barrier = Galois::Runtime::getSystemBarrier();
   const unsigned my_host_id = Galois::Runtime::NetworkInterface::ID;
   //Parse arg string when running on multiple hosts and update/override personality
   //with corresponding value.
   if (personality_set.length() == Galois::Runtime::NetworkInterface::Num) {
      switch (personality_set.c_str()[Galois::Runtime::NetworkInterface::ID]) {
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
   PGraph g;
   g.loadGraph(inputFile);

   if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(Galois::Runtime::NetworkInterface::ID);
      if (!init_CUDA_context(cuda_ctx, gpudevice))
         return;
   } else if (personality == GPU_OPENCL) {
      Galois::OpenCL::cl_env.init();
   }
   if (personality != CPU)
      loadGraphNonCPU(g);
#if _HETERO_DEBUG_
   std::cout << g.id << " graph loaded\n";
#endif
   //local initialization
   if (personality == CPU) {
      InitializeGraph::go(g.g, g.numNodes);
      if (g.getHost((src_node.Value)) == Galois::Runtime::NetworkInterface::ID) {
         fprintf(stderr, "Initialized src %d to zero at part %d\n", src_node.Value, Galois::Runtime::NetworkInterface::ID);
         NodeData & ndata = g.g.getData(src_node.Value);
         ndata.dist[ndata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = 0;
         ndata.dist[ndata.current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = 0;
         worklist.bags[worklist.current_version(BSP_FIELD_NAMES::WORKLIST)].push(src_node.Value);
      }
   } else if (personality == GPU_CUDA) {
      initialize_graph_cuda(cuda_ctx);
   } else if (personality == GPU_OPENCL) {
      dOp.init(g.numOwned, g.numNodes);
      if (g.getHost((src_node.Value)) == Galois::Runtime::NetworkInterface::ID) {
         fprintf(stderr, "Initialized src %d to zero at part %d\n", src_node.Value, Galois::Runtime::NetworkInterface::ID);
         dOp.getData(src_node).dist[dOp.getData(src_node).current_version(BSP_FIELD_NAMES::SSSP_DIST_FIELD)] = 0;
      }
   }
#if _HETERO_DEBUG_
   std::cout << g.id << " initialized\n";
#endif
   barrier.wait();

   //send pGraph pointers
   for (uint32_t x = 0; x < Galois::Runtime::NetworkInterface::Num; ++x) {
      net.sendAlt(x, setRemotePtr, Galois::Runtime::NetworkInterface::ID, &g);
      net.sendAlt(x, setRemoteWorklist, Galois::Runtime::NetworkInterface::ID, &worklist);
   }
   //Ask for cells
   for (auto GID : g.L2G)
      net.sendAlt(g.getHost(GID), recvNodeStatic, GID, Galois::Runtime::NetworkInterface::ID);

#if _HETERO_DEBUG_
   std::cout << "["<<my_host_id<< "]:ask for remote replicas\n";
#endif
   barrier.wait();
   if (false) {   //DEBUG
      {
         {
            auto x = my_host_id;
            fprintf(stderr, "Host:[%d], Offset::[%d], \n", x, g.g_offset);
            for (auto id = 0; id != g.numNodes; ++id) {
               fprintf(stderr, "%d [%d->%d]\n", x, id, g.local2Global(g.begin() + id));
            }
         }
      }
      barrier.wait();
   }

   // send final nout values to remote replicas
#if _HETERO_DEBUG_
   std::cout << "["<<my_host_id<< "]:ask for ghost cell attrs\n";
#endif

   for (int i = 0; i < maxIterations; ++i) {
//      fprintf(stderr, "=========================%d[%d]======================\n", my_host_id, i);
      switch (personality) {
      case CPU:
         SSSP::go(g, g.numOwned);
         break;
      case GPU_OPENCL:
         dOp(g.numOwned, hasChanged);
         break;
      case GPU_CUDA:
         pagerank_cuda(cuda_ctx);
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
   if (hasChanged)
      fprintf(stderr, "Terminating after max=%d steps\n", maxIterations.Value);
   //Final synchronization to ensure that all the nodes are updated.
   sync_distances(net, g);
   barrier.wait();
////////////////////////////////////////////////////////////////////////////////////////////////////////////
   if (verify) {
      std::stringstream ss;
      ss << personality_str(personality) << "_" << my_host_id << "_of_" << Galois::Runtime::NetworkInterface::Num << "_distances.csv";
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
         for (int n = 0; n < g.numOwned; n++) {
            out_file << n + g.g_offset << ", " << getNodeValue_CUDA(cuda_ctx, n) << ", " << getNodeAttr_CUDA(cuda_ctx, n) << "\n";
         }
         break;
      }
      out_file.close();
   }
   std::cout.flush();

}
/********************************************************************************************
 *
 ********************************************************************************************/
int main(int argc, char** argv) {
   LonestarStart(argc, argv, name, desc, url);
   auto& net = Galois::Runtime::getSystemNetworkInterface();
   inner_main();
   return 0;
}
