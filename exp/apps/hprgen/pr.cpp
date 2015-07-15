
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/LC_CSR_Graph.h"
#include "Galois/Graph/Util.h"
#include "Lonestar/BoilerPlate.h"

#include "pr_opencl.h"
#include "pr_cuda.h"

#define _HETERO_DEBUG_ 0

static const char* const name = "Sample";
static const char* const desc = "Sample DH Application";
static const char* const url = 0;

enum Personality {
GPU_OPENCL,GPU_CUDA,CPU
};

namespace cll = llvm::cl;

static cll::opt<Personality> personality("personality", cll::desc("Personality"), cll::values(clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OPENCL"),clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"),clEnumValN(CPU, "cpu/galois", "CPU"), clEnumValEnd), cll::init(GPU_OPENCL));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host: 'o'=GPU/OPENCL, 'g'=GPU/CUDA, 'c'=CPU"), cll::init(""));
static cll::opt<float> cldevice("cldevice", cll::desc("Select OpenCL device (platformid.deviceid) to run on , default is to choose automatically (OpenCL backend)"), cll::init(-1));
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file (transpose)>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(4));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));

struct PRNode { float pr; unsigned int nout; };
typedef struct PRNode LNode;
typedef Galois::OpenCL::LC_LinearArray_Graph<Galois::OpenCL::Array, LNode, void> DeviceGraph;
struct OPENCL_Context<DeviceGraph> pr_cl_ctx;
struct pr_CUDA_Context *pr_cuda_ctx;
typedef Galois::Graph::LC_CSR_Graph<LNode, void> Graph;
typedef typename Graph::GraphNode GNode;
std::map<GNode, float> buffered_updates;

std::string personality_str (Personality p) {
  switch (p) {
    case GPU_OPENCL:
      return "GPU_OPENCL";
      break;
    case GPU_CUDA:
      return "GPU_CUDA";
      break;
    case CPU:
      return "CPU";
      break;
    default:
      assert(false);
      break;
  }
}
/* -*- mode: c++ -*- */

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
         g(_g), g_offset(0), numOwned(0), numNodes(0), id(0),numEdges(0) {
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
   Galois::Graph::permute<void>(fg, perm, fg2);
   Galois::Graph::readGraph(retval.g, fg2);

   loadLastNodes(retval, fg.size(), numHosts);

   /* TODO: This still counts edges from ghosts to remote nodes,
      ideally we only want edges from ghosts to local nodes.
   
      See pGraphToMarshalGraph for one implementation.
   */

   retval.numEdges = std::distance(retval.g.edge_begin(*retval.g.begin()), 
				   retval.g.edge_end(*(retval.g.begin() + 
						       retval.numNodes - 1)));

   return retval;
}

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
      if(*n < g.numOwned) {
	for (auto e = g.g.edge_begin(*n); e != g.g.edge_end(*n); e++) {
	  if(g.g.getEdgeDst(e) < g.numNodes)
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
void update_distr_state(pGraph &g,  Galois::Runtime::Barrier& barrier, Galois::Runtime::NetworkInterface& net) {
   //send pGraph pointers
   for (uint32_t x = 0; x < Galois::Runtime::NetworkInterface::Num; ++x)
      net.sendAlt(x, setRemotePtr, Galois::Runtime::NetworkInterface::ID, &g);

   //Ask for cells
   for (auto GID : g.L2G)
      net.sendAlt(g.getHost(GID), recvNodeStatic, GID, Galois::Runtime::NetworkInterface::ID);

#if _HETERO_DEBUG_
   std::cout << "["<<Galois::Runtime::NetworkInterface::ID<< "]:ask for remote replicas\n";
#endif

   barrier.wait();
}

#include "pr_galois.inc"

void recv_PRNode_nout(pGraph *p, unsigned GID, unsigned int nout) {
  auto LID = GID - p->g_offset;
  if(LID >= p->numOwned) LID = p->G2L(GID);
  switch (personality) {
    case GPU_OPENCL:
      pr_cl_ctx.getData(LID).nout = nout;
      break;
    case GPU_CUDA:
      set_PRNode_nout_CUDA(pr_cuda_ctx, LID, nout);
      break;
    case CPU:
      p->g.getData(LID).nout = nout;
      break;
    default:
      assert(false);
      break;
  }
}
void send_nout(Galois::Runtime::NetworkInterface& net, pGraph& p) {
  for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
    for (auto n : remoteReplicas[x]) {
      
      switch (personality) {
        case GPU_OPENCL:
          net.sendAlt(x, recv_PRNode_nout, magicPointer[x], n, pr_cl_ctx.getData(n - p.g_offset).nout);
          break;
        case GPU_CUDA:
          net.sendAlt(x, recv_PRNode_nout, magicPointer[x], n, get_PRNode_nout_CUDA(pr_cuda_ctx, n - p.g_offset));
          break;
        case CPU:
          net.sendAlt(x, recv_PRNode_nout, magicPointer[x], n, p.g.getData(n - p.g_offset).nout);
          break;
        default:
          assert(false);
          break;
      }
    }
  }
}
void recv_PRNode_nout_plus(pGraph *p, unsigned GID, unsigned int nout) {
  auto LID = GID - p->g_offset;
  if(LID >= p->numOwned) LID = p->G2L(GID);
  switch (personality) {
    case GPU_OPENCL:
      pr_cl_ctx.getData(LID).nout += nout;
      break;
    case GPU_CUDA:
      set_PRNode_nout_plus_CUDA(pr_cuda_ctx, LID, nout);
      break;
    case CPU:
      p->g.getData(LID).nout += nout;
      break;
    default:
      assert(false);
      break;
  }
}
void send_nout_shared(Galois::Runtime::NetworkInterface& net, pGraph& p) {
  for (auto n = p.g.begin() + p.numOwned; n != p.g.begin() + p.numNodes; ++n) {
    auto l2g_ndx = std::distance(p.g.begin(), n) - p.numOwned;
    auto x = p.getHost(p.L2G[l2g_ndx]);
    
    switch (personality) {
      case GPU_OPENCL:
        net.sendAlt(x, recv_PRNode_nout_plus, magicPointer[x], p.L2G[l2g_ndx], pr_cl_ctx.getData(*n).nout);
        break;
      case GPU_CUDA:
        net.sendAlt(x, recv_PRNode_nout_plus, magicPointer[x], p.L2G[l2g_ndx], get_PRNode_nout_CUDA(pr_cuda_ctx, *n));
        break;
      case CPU:
        net.sendAlt(x, recv_PRNode_nout_plus, magicPointer[x], p.L2G[l2g_ndx], p.g.getData(*n).nout);
        break;
      default:
        assert(false);
        break;
    }
  }
}
void recv_PRNode_pr(pGraph *p, unsigned GID, float pr) {
  auto LID = GID - p->g_offset;
  if(LID >= p->numOwned) LID = p->G2L(GID);
  switch (personality) {
    case GPU_OPENCL:
      pr_cl_ctx.getData(LID).pr = pr;
      break;
    case GPU_CUDA:
      set_PRNode_pr_CUDA(pr_cuda_ctx, LID, pr);
      break;
    case CPU:
      p->g.getData(LID).pr = pr;
      break;
    default:
      assert(false);
      break;
  }
}
void send_pr(Galois::Runtime::NetworkInterface& net, pGraph& p) {
  for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
    for (auto n : remoteReplicas[x]) {
      
      switch (personality) {
        case GPU_OPENCL:
          net.sendAlt(x, recv_PRNode_pr, magicPointer[x], n, pr_cl_ctx.getData(n - p.g_offset).pr);
          break;
        case GPU_CUDA:
          net.sendAlt(x, recv_PRNode_pr, magicPointer[x], n, get_PRNode_pr_CUDA(pr_cuda_ctx, n - p.g_offset));
          break;
        case CPU:
          net.sendAlt(x, recv_PRNode_pr, magicPointer[x], n, p.g.getData(n - p.g_offset).pr);
          break;
        default:
          assert(false);
          break;
      }
    }
  }
}
void loadGraphNonCPU(pGraph &g) {
  MarshalGraph m;
  assert(personality != CPU);
  switch(personality) {
    case GPU_OPENCL:
      pr_cl_ctx.loadGraphNonCPU(g);
      break;
    case GPU_CUDA:
      m = pGraph2MGraph(g);
      load_graph_CUDA(pr_cuda_ctx, m);
      break;
    default:
      assert(false);
      break;
  }
}
void parse_pset() {
  if(personality_set.length()==Galois::Runtime::NetworkInterface::Num) {
    switch (personality_set.c_str()[Galois::Runtime::NetworkInterface::ID]) {
      case 'o':
        personality = GPU_OPENCL;
        break;
      case 'g':
        personality = GPU_CUDA;
        break;
      case 'c':
        personality = CPU;
        break;
      default:
        assert(false);
        break;
    }
  }
}
void dev_init() {
  switch(personality) {
    case GPU_OPENCL:
      Galois::OpenCL::cl_env.init(cldevice);
      break;
    case GPU_CUDA:
      pr_cuda_ctx = get_CUDA_context(Galois::Runtime::NetworkInterface::ID);
      if(!init_CUDA_context(pr_cuda_ctx, gpudevice)) exit(1);
      break;
    case CPU:
      break;
    default:
      assert(false);
      break;
  }
}
int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;
  auto& net = Galois::Runtime::getSystemNetworkInterface();
  auto& barrier = Galois::Runtime::getSystemBarrier();
  const unsigned my_host_id = Galois::Runtime::NetworkInterface::ID;
  
  parse_pset();
  std::cout << "Host:"<<Galois::Runtime::NetworkInterface::ID << " personality is " << personality_str(personality) << std::endl;

  Graph rg;
  pGraph g = loadGraph(inputFile, Galois::Runtime::NetworkInterface::ID, Galois::Runtime::NetworkInterface::Num, rg);
  dev_init();
  loadGraphNonCPU(g);
  update_distr_state(g, barrier, net);
  #include "main.inc"
}
