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

#include "cuda/hsssp_cuda.h"
#include "cuda/cuda_mtypes.h"
#include "opencl/CLWrapper.h"

#include <iostream>
#include <typeinfo>
#include <algorithm>

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
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));

typedef unsigned NodeDataType;
typedef Galois::Graph::LC_CSR_Graph<NodeDataType, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;
std::map<GNode, NodeDataType> buffered_updates;

//////////////////////////////////////////////////////////////////////////////////////
struct CUDA_Context *cuda_ctx;
/*********************************************************************************
 * Partitioned graph structure.
 **********************************************************************************/
struct pGraph {
   Graph& g;
   unsigned g_offset; // LID + g_offset = GID
   unsigned numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes) are replicas
   unsigned numNodes; // number of nodes (may differ from g.size() to simplify loading)
   unsigned numEdges;

   // [numNodes, g.size()) should be ignored
   std::vector<unsigned> L2G; // GID = L2G[LID - numOwned]
   unsigned id; // my hostid
   std::vector<unsigned> lastNodes; //[ [i - 1], [i]) -> Node owned by host i
   unsigned getHost(unsigned node) { // node is GID
      return std::distance(lastNodes.begin(), std::upper_bound(lastNodes.begin(), lastNodes.end(), node));
   }
   unsigned G2L(unsigned GID) {
      auto ii = std::find(L2G.begin(), L2G.end(), GID);
      assert(ii != L2G.end());
      return std::distance(L2G.begin(), ii) + numOwned;
   }

   pGraph(Graph& _g) :
         g(_g), g_offset(0), numOwned(0), numNodes(0), id(0), numEdges(0) {
   }
};
/*********************************************************************************
 * Given a partitioned graph  .
 * lastNodes maintains indices of nodes for each co-host. This is computed by
 * determining the number of nodes for each partition in 'pernum', and going over
 * all the nodes assigning the next 'pernum' nodes to the ith partition.
 * The lastNodes is used to find the host by a binary search.
 **********************************************************************************/

void loadLastNodes(pGraph &g, size_t size, unsigned numHosts) {
   if (numHosts == 1)
      return;

   auto p = Galois::block_range(0UL, size, 0, numHosts);
   unsigned pernum = p.second - p.first;
   unsigned pos = pernum;

   while (pos < size) {
      g.lastNodes.push_back(pos);
      pos += pernum;
   }
#if _HETERO_DEBUG_
   for (int i = 0; size < 10 && i < size; i++) {
      printf("node %d owned by %d\n", i, g.getHost(i));
   }
#endif
}
/*********************************************************************************
 * Load a partitioned graph from a file.
 * @param file the graph filename to be loaded.
 * @param hostID the identifier of the current host.
 * @param numHosts the total number of hosts.
 * @param out A graph instance that stores the actual graph.
 * @return a partitioned graph backed by the #out instance.
 **********************************************************************************/
pGraph loadGraph(std::string file, unsigned hostID, unsigned numHosts, Graph& out) {
   pGraph retval { out };
   Galois::Graph::FileGraph fg;
   fg.fromFile(file);
   auto p = Galois::block_range(0UL, fg.size(), hostID, numHosts);
   retval.g_offset = p.first;
   retval.numOwned = p.second - p.first;
   retval.id = hostID;
   std::vector<unsigned> perm(fg.size(), ~0); //[i (orig)] -> j (final)
   unsigned nextSlot = 0;
//   std::cout << fg.size() << " " << p.first << " " << p.second << "\n";
   //Fill our partition
   for (unsigned i = p.first; i < p.second; ++i) {
      //printf("%d: owned: %d local: %d\n", hostID, i, nextSlot);
      perm[i] = nextSlot++;
   }
   //find ghost cells
   for (auto ii = fg.begin() + p.first; ii != fg.begin() + p.second; ++ii) {
      for (auto jj = fg.edge_begin(*ii); jj != fg.edge_end(*ii); ++jj) {
         //std::cout << *ii << " " << *jj << " " << nextSlot << " " << perm.size() << "\n";
         //      assert(*jj < perm.size());
         auto dst = fg.getEdgeDst(jj);
         if (perm.at(dst) == ~0) {
            //printf("%d: ghost: %d local: %d\n", hostID, dst, nextSlot);
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
//   std::cout << nextSlot << " " << fg.size() << "\n";
   assert(nextSlot == fg.size());
   //permute graph
   Galois::Graph::FileGraph fg2;
   Galois::Graph::permute<unsigned>(fg, perm, fg2);
   Galois::Graph::readGraph(retval.g, fg2);

   loadLastNodes(retval, fg.size(), numHosts);

   /* TODO: This still counts edges from ghosts to remote nodes,
    ideally we only want edges from ghosts to local nodes.
    See pGraphToMarshalGraph for one implementation.
    */

   retval.numEdges = std::distance(retval.g.edge_begin(*retval.g.begin()), retval.g.edge_end(*(retval.g.begin() + retval.numNodes - 1)));

   return retval;
}
/*********************************************************************************
 *
 **********************************************************************************/
struct InitializeGraph {
   Graph* g;
   void static go(Graph& _g, unsigned num) {
      Galois::do_all(_g.begin(), _g.begin() + num, InitializeGraph { &_g }, Galois::loopname("init"));
   }
   void operator()(GNode src) const {
      unsigned & sdata = g->getData(src);
      sdata = std::numeric_limits<unsigned>::max() / 2;
      buffered_updates[src] = std::numeric_limits<unsigned>::max() / 2;
   }
};
/************************************************************************************
 * OpenCL PageRank operator implementation.
 * Uses two kernels for the updates and one kernel for initialization.
 * In order to support BSP semantics, an aux_array is created which will
 * be used to buffer the writes in 'kernel'. These updates will be
 * written to the node-data in the 'writeback' kernel.
 *************************************************************************************/
typedef Galois::OpenCL::LC_LinearArray_Graph<Galois::OpenCL::Array, unsigned int, unsigned int> DeviceGraph;
DeviceGraph dGraph;
struct dSSSP {
   Galois::OpenCL::CL_Kernel kernel;
   Galois::OpenCL::CL_Kernel wb_kernel;
   Galois::OpenCL::Array<NodeDataType> *aux_array;
   dSSSP() :
         aux_array(nullptr) {
   }
   void init(int num_items, int num_inits) {
      Galois::OpenCL::CL_Kernel init_nodes;
      init_nodes.init("sssp_kernel.cl", "initialize_nodes");
      aux_array = new Galois::OpenCL::Array<NodeDataType>(dGraph.num_nodes());
      kernel.init("sssp_kernel.cl", "sssp");
      wb_kernel.init("sssp_kernel.cl", "writeback");
      dGraph.copy_to_device();

      wb_kernel.set_work_size(num_items);
      kernel.set_work_size(num_items);
      init_nodes.set_work_size(num_items);

      kernel.set_arg_list(&dGraph, aux_array);
      wb_kernel.set_arg_list(&dGraph, aux_array);
      init_nodes.set_arg_list(&dGraph, aux_array);
      int num_nodes = dGraph.num_nodes();

      wb_kernel.set_arg(2, sizeof(cl_int), &num_items);
      kernel.set_arg(2, sizeof(cl_int), &num_items);
      init_nodes.set_arg(2, sizeof(cl_int), &num_items);
      init_nodes();

      dGraph.copy_to_host();

   }
   template<typename GraphType>
   void operator()(GraphType & graph, int num_items) {
      graph.copy_to_device();
      kernel();
      wb_kernel();
      graph.copy_to_host();
   }
};
/*********************************************************************************
 * CPU PageRank operator implementation.
 **********************************************************************************/
struct WriteBack {
   Graph * g;
   void static go(Graph& _g, unsigned num) {
      Galois::do_all(_g.begin(), _g.begin() + num, WriteBack { &_g }, Galois::loopname("Writeback"));
   }
   void operator()(GNode src) const {
      unsigned & sdata = g->getData(src);
      sdata = buffered_updates[src];
   }
};
struct SSSP {
   Graph* g;

   void static go(Graph& _g, unsigned num) {
      Galois::do_all(_g.begin(), _g.begin() + num, SSSP { &_g }, Galois::loopname("SSSP"));
   }

   void operator()(GNode src) const {
      NodeDataType& sdata = g->getData(src);
      NodeDataType min_dist = sdata;
      for (auto jj = g->edge_begin(src), ej = g->edge_end(src); jj != ej; ++jj) {
         GNode dst = g->getEdgeDst(jj);
         NodeDataType& ddata = g->getData(dst);
         unsigned int ewt = g->getEdgeData(jj);
         min_dist = std::min(min_dist, ewt + ddata);
      }
      buffered_updates[src] = min_dist;
   }
};
/*********************************************************************************
 *
 **********************************************************************************/

// [hostid] -> vector of GID that host has replicas of
std::vector<std::vector<unsigned> > remoteReplicas;
// [hostid] -> remote pGraph Structure (locally invalid)
std::vector<pGraph*> magicPointer;
/*********************************************************************************
 *
 **********************************************************************************/

void setRemotePtr(uint32_t hostID, pGraph* p) {
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

void setNodeValue(pGraph* p, unsigned GID, NodeDataType v) {
   if (p->g.getData(p->G2L(GID)) > v) {
//      fprintf(stderr, "Lowered-ghost[%d, %d->%d]", p->G2L(GID), p->g.getData(p->G2L(GID)),v);
   }
//   else{
//      return;
//   }
   switch (personality) {
   case CPU:
      p->g.getData(p->G2L(GID)) = v;
      break;
   case GPU_CUDA:
      setNodeValue_CUDA(cuda_ctx, p->G2L(GID), v);
      break;
   case GPU_OPENCL:
      dGraph.getData(p->G2L(GID)) = v;
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

void sendGhostCells(Galois::Runtime::NetworkInterface& net, pGraph& g) {
   for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
      for (auto n : remoteReplicas[x]) {
         switch (personality) {
         case CPU:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, g.g.getData(n - g.g_offset));
            break;
         case GPU_CUDA:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, getNodeValue_CUDA(cuda_ctx, n - g.g_offset));
            break;
         case GPU_OPENCL:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, dGraph.getData(n - g.g_offset));
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

MarshalGraph pGraph2MGraph(pGraph &g) {
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
//   printf("dropped %u edges (g->r, g->l, g->g)\n", g.numEdges - m.nedges);
   // for(int i = 0; i < node_counter; i++) {
   //   printf("%u ", m.row_start[i]);
   // }

   // for(int i = 0; i < edge_counter; i++) {
   //   printf("%u ", m.edge_dst[i]);
   // }

   // printf("nc: %zu ec: %zu\n", node_counter, edge_counter);
   return m;
}
/*********************************************************************************
 *
 **********************************************************************************/

void loadGraphNonCPU(pGraph &g) {
   MarshalGraph m;
   assert(personality != CPU);
   switch (personality) {
   case GPU_CUDA:
      m = pGraph2MGraph(g);
      load_graph_CUDA(cuda_ctx, m);
      break;
   case GPU_OPENCL:
      dGraph.load_from_galois(g.g, g.numOwned, g.numEdges, g.numNodes - g.numOwned);
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
   Graph rg;
   pGraph g = loadGraph(inputFile, Galois::Runtime::NetworkInterface::ID, Galois::Runtime::NetworkInterface::Num, rg);

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
   dSSSP dOp;

   //local initialization
   if (personality == CPU) {
      InitializeGraph::go(g.g, g.numNodes);
      if (g.getHost((src_node.Value)) == Galois::Runtime::NetworkInterface::ID) {
         fprintf(stderr, "Initialized src %d to zero at part %d\n", src_node.Value, Galois::Runtime::NetworkInterface::ID);
         g.g.getData(src_node) = 0;
      }
   } else if (personality == GPU_CUDA) {
      initialize_graph_cuda(cuda_ctx);
   } else if (personality == GPU_OPENCL) {
      dOp.init(g.numOwned, g.numNodes);
      if (g.getHost((src_node.Value)) == Galois::Runtime::NetworkInterface::ID) {
         fprintf(stderr, "Initialized src %d to zero at part %d\n", src_node.Value, Galois::Runtime::NetworkInterface::ID);
         dGraph.getData(src_node) = 0;
      }
   }
#if _HETERO_DEBUG_
   std::cout << g.id << " initialized\n";
#endif
   barrier.wait();

   //send pGraph pointers
   for (uint32_t x = 0; x < Galois::Runtime::NetworkInterface::Num; ++x)
      net.sendAlt(x, setRemotePtr, Galois::Runtime::NetworkInterface::ID, &g);

   //Ask for cells
   for (auto GID : g.L2G)
      net.sendAlt(g.getHost(GID), recvNodeStatic, GID, Galois::Runtime::NetworkInterface::ID);
#if _HETERO_DEBUG_
   std::cout << "["<<my_host_id<< "]:ask for remote replicas\n";
#endif
   barrier.wait();
   // send final nout values to remote replicas
#if _HETERO_DEBUG_
   std::cout << "["<<my_host_id<< "]:ask for ghost cell attrs\n";
#endif

   for (int i = 0; i < maxIterations; ++i) {
      //communicate ghost cells
      sendGhostCells(net, g);
      barrier.wait();
      //Do pagerank
      switch (personality) {
      case CPU:
         SSSP::go(g.g, g.numOwned);
         WriteBack::go(g.g, g.numOwned);
         break;
      case GPU_OPENCL:
         dOp(dGraph, g.numOwned);
         break;
      case GPU_CUDA:
         pagerank_cuda(cuda_ctx);
         break;
      default:
         break;
      }
   }
   //Final synchronization to ensure that all the nodes are updated.
   sendGhostCells(net, g);
   barrier.wait();

   if (verify) {
      std::stringstream ss;
      ss << personality_str(personality) << "_" << my_host_id << "_of_" << Galois::Runtime::NetworkInterface::Num << "_distances.csv";
      std::ofstream out_file(ss.str());
      switch (personality) {
      case CPU: {
         unsigned max_dist = 0;
         for (auto n = g.g.begin(); n != g.g.begin() + g.numOwned; ++n) {
            max_dist = std::max(max_dist, g.g.getData(*n));
            out_file << *n + g.g_offset << ", " << g.g.getData(*n) << "\n";
         }
         fprintf(stderr, "Maxdist - %d\n", max_dist);
         break;
      }
      case GPU_OPENCL: {
         unsigned max_dist = 0;
         for (int n = 0; n < g.numOwned; ++n) {
            max_dist = std::max(max_dist, dGraph.getData(n));
            out_file << n + g.g_offset << ", " << dGraph.getData(n) << "\n";
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
