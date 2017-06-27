/** Betweeness Centrality -*- C++ -*-
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
 * @section Description
 *
 * Compute Betweeness-Centrality on distributed Galois 
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

////////////////////////////////////////////////////////////////////////////////
// IMPORTANT: THIS CODE ONLY WORKS FOR OUTGOING EDGE CUTS
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "Galois/Runtime/dGraph_edgeCut.h"
#include "Galois/Runtime/dGraph_cartesianCut.h"
#include "Galois/Runtime/dGraph_hybridCut.h"

#include "Galois/DistAccumulator.h"
#include "Galois/Runtime/Tracer.h"

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Runtime/Cuda/cuda_device.h"
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;

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
   assert(false && "Invalid personality");
   return "";
}
#endif

enum VertexCut {
  PL_VCUT, CART_VCUT
};

static const char* const name = "Betweeness Centrality (OEC) - "
                                "Distributed Heterogeneous.";
static const char* const desc = "Betweeness Centrality on Distributed Galois.";
static const char* const url = 0;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional,
                                       cll::desc("<input file>"),
                                       cll::Required);
static cll::opt<std::string> partFolder("partFolder",
                                        cll::desc("path to partitionFolder"),
                                        cll::init(""));
static cll::opt<unsigned int> maxIterations("maxIterations", 
                               cll::desc("Maximum iterations: Default 1000000"), 
                               cll::init(1000000));
static cll::opt<bool> transpose("transpose", 
                                cll::desc("transpose the graph in memory after "
                                          "partitioning"),
                                cll::init(false));
static cll::opt<bool> verify("verify", 
                             cll::desc("Verify ranks by printing to "
                                       "'page_ranks.#hid.csv' file"),
                             cll::init(false));
static cll::opt<bool> enableVCut("enableVertexCut", 
                                 cll::desc("Use vertex cut for graph " 
                                           "partitioning."), 
                                 cll::init(false));
static cll::opt<unsigned int> VCutThreshold("VCutThreshold", 
                                            cll::desc("Threshold for high "
                                                      "degree edges."),
                                            cll::init(100));
static cll::opt<VertexCut> vertexcut("vertexcut", 
                                     cll::desc("Type of vertex cut."),
                                     cll::values(clEnumValN(PL_VCUT, "pl_vcut",
                                                        "Powerlyra Vertex Cut"),
                                                 clEnumValN(CART_VCUT, 
                                                   "cart_vcut", 
                                                   "Cartesian Vertex Cut"),
                                                 clEnumValEnd),
                                     cll::init(PL_VCUT));
static cll::opt<bool> singleSourceBC("singleSource", 
                                cll::desc("Use for single source BC"),
                                cll::init(false));
static cll::opt<unsigned int> singleSourceNode("srcNodeId", 
                                cll::desc("Source node used for single source "
                                          "betweeness-centraility"),
                                cll::init(0));

#ifdef __GALOIS_HET_CUDA__
// If running on both CPUs and GPUs, below is included
static cll::opt<int> gpudevice("gpu", 
                      cll::desc("Select GPU to run on, default is "
                                "to choose automatically"), cll::init(-1));
static cll::opt<Personality> personality("personality", 
                 cll::desc("Personality"),
                 cll::values(clEnumValN(CPU, "cpu", "Galois CPU"),
                             clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"),
                             clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"),
                             clEnumValEnd),
                 cll::init(CPU));
static cll::opt<std::string> personality_set("pset", 
                              cll::desc("String specifying personality for "
                                        "each host. 'c'=CPU,'g'=GPU/CUDA and "
                                        "'o'=GPU/OpenCL"),
                              cll::init(""));
static cll::opt<unsigned> scalegpu("scalegpu", 
                           cll::desc("Scale GPU workload w.r.t. CPU, default "
                                     "is proportionally equal workload to CPU "
                                     "and GPU (1)"), 
                           cll::init(1));
static cll::opt<unsigned> scalecpu("scalecpu", 
                           cll::desc("Scale CPU workload w.r.t. GPU, "
                                     "default is proportionally equal "
                                     "workload to CPU and GPU (1)"), 
                           cll::init(1));
static cll::opt<int> num_nodes("num_nodes", 
                      cll::desc("Num of physical nodes with devices (default "
                                "= num of hosts): detect GPU to use for each "
                                "host automatically"), 
                      cll::init(-1));
#endif

const unsigned int infinity = std::numeric_limits<unsigned int>::max() / 4;
static unsigned int current_src_node = 0;

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

struct NodeData {
  // SSSP vars
  std::atomic<unsigned int> current_length;

  unsigned int old_length;

  // Betweeness centrality vars
  std::atomic<unsigned int> num_shortest_paths;
  unsigned int num_successors;
  std::atomic<unsigned int> num_predecessors;
  std::atomic<unsigned int> trim;
  float dependency;
  float betweeness_centrality;

  // used to determine if data has been propogated yet
  bool propogation_flag;
};

struct WriteStatus {
  bool src_write;
  bool dst_write;
};

/* set all flags to false */
static void resetFlags(WriteStatus& flag_object) {
  flag_object.src_write = false;
  flag_object.dst_write = false;
}

// Flags for keeping track of reads/writes on src/dest
WriteStatus current_length_flags;
WriteStatus num_successors_flags;
WriteStatus num_predecessors_flags;
WriteStatus num_shortest_paths_flags;
WriteStatus trim_flags;
WriteStatus propogation_flag_flags;
WriteStatus dependency_flags;

// second type (unsigned int) is for edge weights
typedef hGraph<NodeData, unsigned int> Graph;
typedef hGraph_edgeCut<NodeData, unsigned int> Graph_edgeCut;
typedef hGraph_vertexCut<NodeData, unsigned int> Graph_vertexCut;
typedef hGraph_cartesianCut<NodeData, unsigned int> Graph_cartesianCut;

typedef typename Graph::GraphNode GNode;

// bitset for tracking updates
Galois::DynamicBitSet bitset_update;
Galois::DynamicBitSet bitset_update_succ;
Galois::DynamicBitSet bitset_update_flag;

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/
struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
    // sync unnecessary if I loop through all nodes (using ghost_end)
    Galois::do_all(_graph.begin(), _graph.ghost_end(), InitializeGraph{&_graph}, 
                   Galois::loopname("InitializeGraph"), 
                   Galois::numrun(_graph.get_run_identifier()));
    // technically I would set the betweeness-cent flag here, but assuming
    // OEC only it will never matter as we never read/write from dst's bet-cet
    // measure
    // TODO technically a reset should be broadcast as it's a non reduction
    // write
  }

  /* Functor passed into the Galois operator to carry out initialization;
   * Reset bc measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    src_data.betweeness_centrality = 0;
  }
};

/* This is used to reset node data when switching to a difference source */
struct InitializeIteration {
  const unsigned int &local_infinity;
  const unsigned int &local_current_src_node;
  Graph *graph;

  InitializeIteration(const unsigned int &_local_infinity,
                      const unsigned int &_local_current_src_node,
                      Graph* _graph) : 
                       local_infinity(_local_infinity),
                       local_current_src_node(_local_current_src_node),
                       graph(_graph){}

  /* Reset graph metadata node-by-node */
  void static go(Graph& _graph) {
    ////////////////////////////////////////////////////////////////////////////
    // Trim
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushTrim {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
        return node.trim;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
        return false;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      }
    };

    struct SyncPullTrim {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_trim_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.trim;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_trim_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_trim_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_trim_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.trim, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // # short paths
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushPaths {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
        return 0;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
        return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
        Galois::set(node.num_shortest_paths, (unsigned int)0);
      }
    };

    struct SyncPullPaths {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_shortest_paths_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif

        return node.num_shortest_paths;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_shortest_paths_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif

        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_num_shortest_paths_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.num_shortest_paths, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Succ
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushSucc {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
        return 0;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
        return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      }
    };

    struct SyncPullSucc {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_successors_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_successors;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_successors_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_num_successors_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.num_successors, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Pred
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushPred {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
        return node.num_predecessors;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
        return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
        Galois::set(node.num_predecessors, (unsigned int)0);
      }
    };

    struct SyncPullPred {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_predecessors_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_predecessors;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_predecessors_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_num_predecessors_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.num_predecessors, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Lengths
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushLength {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
        return 0;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
        return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      }
    };

    struct SyncPullLength {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_current_length_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.current_length;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_current_length_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_current_length_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.current_length, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Flag
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushFlag {
      typedef unsigned int ValTy;
      //typedef bool ValTy; TODO for some reason this causes issues

      static bool extract(uint32_t node_id, const struct NodeData & node) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         bool y) {
        return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
        node.propogation_flag = false;
      }
    };

    struct SyncPullFlag {
      typedef unsigned int ValTy;
      //typedef bool ValTy; TODO for some reason this causes issues

      static bool extract(uint32_t node_id, const struct NodeData & node) {
        return node.propogation_flag;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, bool y) {
        Galois::set(node.propogation_flag, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

    };


    ////////////////////////////////////////////////////////////////////////////
    // Dependency
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushDependency {
      typedef unsigned int ValTy;
      //typedef float ValTy; TODO for some reason this causes issues

      static float extract(uint32_t node_id, const struct NodeData & node) {
        return 0.0;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, float y) {
        return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
        Galois::set(node.dependency, (float)0);
      }
    };

    struct SyncPullDependency {
      typedef unsigned int ValTy;
      //typedef float ValTy; TODO for some reason this causes issues

      static float extract(uint32_t node_id, const struct NodeData & node) {
        return node.dependency;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         float y) {
        Galois::set(node.dependency, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }
    };


    ////////////////////////////////////////////////////////////////////////////

    resetFlags(propogation_flag_flags);
    resetFlags(current_length_flags);
    resetFlags(num_successors_flags);
    resetFlags(num_predecessors_flags);
    resetFlags(trim_flags);
    resetFlags(num_shortest_paths_flags);
    resetFlags(dependency_flags);

    Galois::do_all(_graph.begin(), _graph.end(), 
                   InitializeIteration{infinity, current_src_node, &_graph},
                   Galois::loopname("InitializeIteration"), 
                   Galois::numrun(_graph.get_run_identifier()));

    // broadcast ALL reset values (inefficient, but this is initialization +
    // it's to guarantee a hard reset on slave nodes because otherwise in this
    // scheme you now have to track non-reduce writes, which these are)

    // note the pushes structures' reduce has been set to ignore (i.e.
    // push reduce does nothing as I'm only interested in the broadcast)
    _graph.sync_backward<SyncPushLength, SyncPullLength>("InitializeIteration");
    _graph.sync_backward<SyncPushFlag, SyncPullFlag>("InitializeIteration");
    _graph.sync_backward<SyncPushSucc, SyncPullSucc>("InitializeIteration");
    _graph.sync_backward<SyncPushPred, SyncPullPred>("InitializeIteration");
    _graph.sync_backward<SyncPushTrim, SyncPullTrim>("InitializeIteration");
    _graph.sync_backward<SyncPushPaths, SyncPullPaths>("InitializeIteration");
    _graph.sync_backward<SyncPushDependency, 
                         SyncPullDependency>("InitializeIteration");
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.current_length = (graph->getGID(src) == local_current_src_node) ?  0 : local_infinity;
    src_data.old_length = (graph->getGID(src) == local_current_src_node) ?  0 : local_infinity;

    src_data.trim = 0;

    // set to true = "I have propogated" i.e. don't propogate anymore
    // set to false = "I have not propogated and/or I cannot propogate"
    src_data.propogation_flag = false; 

    // set num to 1 on source so that it can propogate it across nodes later
    // note source will not have sigma accessed anyways (at least it shouldn't)
    src_data.num_shortest_paths = (graph->getGID(src) == local_current_src_node) ?  1 : 0;

    src_data.num_successors = 0;
    src_data.num_predecessors = 0;
    src_data.dependency = 0;
  }
};

/* Need a separate call for the first iteration as the condition check is 
 * different */
struct FirstIterationSSSP {
  Graph* graph;
  FirstIterationSSSP(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    struct SyncPushLength {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_current_length_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.current_length;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_reset_node_current_length_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          min_node_current_length_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        return y < Galois::min(node.current_length, y);
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_min_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                             data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      }
    };

    struct SyncPullLength {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_current_length_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.current_length;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_current_length_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_current_length_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.current_length, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    unsigned int __begin, __end;
    if (_graph.isLocal(current_src_node)) {
      __begin = _graph.getLID(current_src_node);
      __end = __begin + 1;
    } else {
      __begin = 0;
      __end = 0;
    }

    bitset_update.clear();

    Galois::do_all(boost::make_counting_iterator(__begin), 
                   boost::make_counting_iterator(__end), 
                   FirstIterationSSSP(&_graph),
                   Galois::loopname("FirstIterationSSSP"));

    current_length_flags.dst_write = true;
  }

  /* Does SSSP, push/filter based */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);

      //unsigned int new_dist = graph->getEdgeData(current_edge) + 
      //                        src_data.current_length;
      // TODO change this back later 
      // BFS simulation
      unsigned int new_dist = 1 + src_data.current_length;

      Galois::atomicMin(dst_data.current_length, new_dist);

      bitset_update.set(dst);
    }
  }
};

/* Sub struct for running SSSP (beyond 1st iteration) */
struct SSSP {
  Graph* graph;
  static Galois::DGAccumulator<unsigned int> DGAccumulator_accum;
  SSSP(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    struct SyncPushLength {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_current_length_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.current_length;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_reset_node_current_length_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          min_node_current_length_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        return y < Galois::min(node.current_length, y);
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_min_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                             data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      }
    };

    struct SyncPullLength {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_current_length_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.current_length;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_current_length_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_current_length_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.current_length, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    FirstIterationSSSP::go(_graph);

    // starts at 1 since FirstSSSP takes care of the first one
    unsigned int iterations = 1;

    do {
      _graph.set_num_iter(iterations);
      DGAccumulator_accum.reset();

      // READ SRC current length; only care if dst was written
      if (current_length_flags.dst_write) {
        // reduce to master
        _graph.sync_forward<SyncPushLength, SyncPullLength>("SSSP");
        //_graph.sync_forward<SyncPushLength, SyncPullLength>("SSSP", 
        //                                                   bitset_update);
        resetFlags(current_length_flags);
      }

      // READ DST OLD LENGTH (for optimization purposes, you could only set
      // bitset where old length > new length... TODO?

      // READ SRC old length; unnecessary as you will never write to dest
      // old_length
      //if (old_length_flags.dst_write) {
      //  _graph....
      //}

      bitset_update.clear();

      Galois::do_all(_graph.begin(), _graph.end(), SSSP(&_graph), 
                     Galois::loopname("SSSP"));

      // WRITE SRC old length (but we don't need to sync it as old length
      // isn't written on dst, only src)
      //old_length_flags.dst_write = true; // note this isn't declared yet

      current_length_flags.dst_write = true;

      iterations++;

    } while (DGAccumulator_accum.reduce());
  }

  /* Does SSSP, push/filter based */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    if (src_data.old_length > src_data.current_length) {
      src_data.old_length = src_data.current_length;

      for (auto current_edge = graph->edge_begin(src), 
                end_edge = graph->edge_end(src); 
           current_edge != end_edge; 
           ++current_edge) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        //unsigned int new_dist = graph->getEdgeData(current_edge) + 
        //                        src_data.current_length;
        // TODO change this back later 
        // BFS simulation
        unsigned int new_dist = 1 + src_data.current_length;

        Galois::atomicMin(dst_data.current_length, new_dist);
        // TODO could optimize this bitset to only set if old length
        // is greater than new length, but extra sync may be required...
        bitset_update.set(dst);
      }

      DGAccumulator_accum += 1;
    }
  }
};
Galois::DGAccumulator<unsigned int> SSSP::DGAccumulator_accum;

/* Struct to get pred and succon the SSSP DAG */
struct PredAndSucc {
  Graph* graph;

  PredAndSucc(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    struct SyncPushLength {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_current_length_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.current_length;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_reset_node_current_length_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          min_node_current_length_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        return y < Galois::min(node.current_length, y);
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_min_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                             data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      }
    };

    struct SyncPullLength {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_current_length_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.current_length;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_current_length_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_current_length_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.current_length, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };
    ////////////////////////////////////////////////////////////////////////////

    // READ SRC current length and READ DST current length
    if (current_length_flags.src_write && current_length_flags.dst_write) {
      _graph.sync_exchange<SyncPushLength, SyncPullLength>("PredAndSucc");
      resetFlags(current_length_flags);
    } else if (current_length_flags.src_write) {
      _graph.sync_backward<SyncPushLength, SyncPullLength>("PredAndSucc");
      resetFlags(current_length_flags);
    } else if (current_length_flags.dst_write) {
      _graph.sync_exchange<SyncPushLength, SyncPullLength>("PredAndSucc");
      resetFlags(current_length_flags);
    } 

    //else {
    //  // TODO verify this
    //  // no new write has occured
    //  // HOWEVER, it's possible that dst may not have the most up to date
    //  // value: do a broadcast (e.g. in 1 op, src is read after dst is
    //  // written, so flags are reset, but in that op no writes occur:
    //  // then, once you get to a new op, you will never update dst unless
    //  // the flag is set so dst holds the incorrect value)
    //  // TODO: also, find a way to make it more efficient instead of a
    //  // broadcast every round like what it would be doing if this code
    //  // was active
    //  _graph.sync_backward<SyncPushLength, SyncPullLength>("PredAndSucc");
    //}

    bitset_update.clear();
    // Loop over all nodes in graph iteratively
    Galois::do_all(_graph.begin(), _graph.end(), PredAndSucc(&_graph), 
                   Galois::loopname("PredAndSucc"));

    num_successors_flags.src_write = true;
    num_predecessors_flags.dst_write = true;
  }

  /* Summary:
   * Look at outgoing edges; see if dest is on a shortest path from src node.
   * If it is, increment the number of successors on src by 1 and
   * increment # of pred on dest by 1 
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);

      // TODO change BFS back when done testing
      //unsigned int edge_weight = graph->getEdgeData(current_edge);
      unsigned int edge_weight = 1;

      if ((src_data.current_length + edge_weight) == dst_data.current_length) {
        // dest on shortest path with this node as predecessor
        Galois::add(src_data.num_successors, (unsigned int)1);
        Galois::atomicAdd(dst_data.num_predecessors, (unsigned int)1);

        bitset_update_succ.set(src);
      }
    }
  }
};

/* Uses an incremented trim value to decrement the predecessor: the trim value
 * has to be synchronized across ALL nodes (including slaves) */
struct PredecessorDecrement {
  Graph* graph;

  PredecessorDecrement(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    struct SyncPushTrim {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_trim_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.trim;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__

        if (personality == GPU_CUDA) {
          batch_get_reset_node_trim_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__

        if (personality == GPU_CUDA) {
          add_node_trim_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        { Galois::add(node.trim, y); return true; }
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {

          batch_add_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          set_node_trim_cuda(cuda_ctx, node_id, 0);
        }
        else if (personality == CPU)
      #endif
        Galois::set(node.trim, (unsigned int)0);
      }
    };

    struct SyncPullTrim {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_trim_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.trim;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_trim_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_trim_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_trim_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.trim, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    // READ SRC trim
    if (trim_flags.dst_write) {
      // reduce to master
      //_graph.sync_forward<SyncPushTrim, SyncPullTrim>("PredecessorDecrement", 
      //                                                bitset_update);
      _graph.sync_forward<SyncPushTrim, SyncPullTrim>("PredecessorDecrement");

      resetFlags(trim_flags);
    }

    // DO NOT DO A BITSET RESET HERE BECAUSE IT WILL BE REUSED BY THE NEXT STEP
    // (updates to trim and pred are on the same nodes)

    Galois::do_all(_graph.begin(), _graph.end(), PredecessorDecrement{&_graph}, 
                   Galois::loopname("PredecessorDecrement"), 
                   Galois::numrun(_graph.get_run_identifier()));
    
    trim_flags.src_write = true;
    num_predecessors_flags.src_write = true;
    propogation_flag_flags.src_write = true;
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // decrement predecessor by trim then reset
    if (src_data.trim > 0) {
      if (src_data.trim > src_data.num_predecessors) {
        std::cerr << "ISSUE P: src " << src << " " << src_data.trim << " " << 
                                     src_data.num_predecessors << "\n";
        abort();                                    
      }

      src_data.num_predecessors -= src_data.trim;
      src_data.trim = 0;

      // if I hit 0 predecessors, set the flag to false (i.e. says
      // I need to propogate my value)
      // TODO: actually, at the moment, it's set to false by default (otherwise
      // nodes that already have 0 will be ignored in the first iteration),
      // so this isn't doing much
      if (src_data.num_predecessors == 0) {
        src_data.propogation_flag = false;
      }
    }
  }
};


/* Calculate the number of shortest paths for each node */
struct NumShortestPaths {
  Graph* graph;
  static Galois::DGAccumulator<unsigned int> DGAccumulator_accum;

  NumShortestPaths(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    ////////////////////////////////////////////////////////////////////////////
    // # short paths
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushPaths {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_shortest_paths_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_shortest_paths;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {

      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o,
                                                       y, s, data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_num_shortest_paths_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          add_node_num_shortest_paths_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        Galois::add(node.num_shortest_paths, y); return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_add_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s, 
                                                 data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif

        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          set_node_num_shortest_paths_cuda(cuda_ctx, node_id, 0);
        }
        else if (personality == CPU)
      #endif
        Galois::set(node.num_shortest_paths, (unsigned int)0);
      }
    };

    struct SyncPullPaths {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_shortest_paths_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif

        return node.num_shortest_paths;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_shortest_paths_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif

        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_num_shortest_paths_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.num_shortest_paths, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };
    ////////////////////////////////////////////////////////////////////////////
    // Succ
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushSucc {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_successors_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_successors;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, 
                                      unsigned int *y, 
                                      size_t *s, 
                                      DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_reset_node_num_successors_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          add_node_num_successors_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        { return Galois::add(node.num_successors, y); return true; }
      }

      static bool reduce_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {

          batch_add_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          set_node_num_predecessors_cuda(cuda_ctx, node_id, 0);
        }
        else if (personality == CPU)
      #endif
        Galois::set(node.num_successors, (unsigned int)0);
      }
    };

    struct SyncPullSucc {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_successors_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_successors;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_successors_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_num_successors_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.num_successors, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };
    ////////////////////////////////////////////////////////////////////////////
    // Pred
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushPred {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_predecessors_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_predecessors;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, 
                                      unsigned int *y, 
                                      size_t *s, 
                                      DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_reset_node_num_predecessors_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          add_node_num_predecessors_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        { return Galois::add(node.num_predecessors, y); return true; }
      }

      static bool reduce_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {

          batch_add_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          set_node_num_predecessors_cuda(cuda_ctx, node_id, 0);
        }
        else if (personality == CPU)
      #endif
        Galois::set(node.num_predecessors, (unsigned int)0);
      }
    };

    struct SyncPullPred {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_predecessors_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_predecessors;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_predecessors_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_num_predecessors_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.num_predecessors, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
    };
    ////////////////////////////////////////////////////////////////////////////

    unsigned int iterations = 0;

    do {
      _graph.set_num_iter(iterations);
      DGAccumulator_accum.reset();

      // READ SRC # shortest paths
      if (num_shortest_paths_flags.dst_write) {
        _graph.sync_forward<SyncPushPaths, SyncPullPaths>("NumShortestPaths");
        resetFlags(num_shortest_paths_flags);
      }

      // READ SRC pred
      if (num_predecessors_flags.dst_write) {
        _graph.sync_forward<SyncPushPred, SyncPullPred>("NumShortestPaths");
        resetFlags(num_predecessors_flags);
      }

      // READ SRC current length and READ DST current length
      // NOTE: It should never get in here since current length at this point 
      // will no longer be updated
      //if (current_length_flags.src_write && current_length_flags.dst_write) {
      //  _graph.sync_exchange<SyncPushLength, 
      //                       SyncPullLength>("NumShortestPaths");
      //  resetFlags(current_length_flags);
      //} else if (current_length_flags.src_write) {
      //  _graph.sync_backward<SyncPushLength, 
      //                       SyncPullLength>("NumShortestPaths");
      //  resetFlags(current_length_flags);
      //} else if (current_length_flags.dst_write) {
      //  _graph.sync_exchange<SyncPushLength, 
      //                       SyncPullLength>("NumShortestPaths");
      //  resetFlags(current_length_flags);
      //} 

      // READ SRC succ (only if you want optimization; not activated here)
      //if (num_successors_flags.dst_write) {
      //  _graph.sync_forward<SyncPushSucc, SyncPullSucc>("NumShortestPaths");
      //  resetFlags(num_successors_flags);
      //}
      // READ SRC prop flag; should never happen since we never write to 
      // dst flag before this
      //if (propogation_flag_flags.dst_write) {
      //  _graph.sync_forward<SyncPushFlag, SyncPullFlag>("NumShortestPaths");
      //  resetFlags(propogation_flag_flags);
      //}
      
      bitset_update.clear();

      Galois::do_all(_graph.begin(), _graph.end(), 
                     NumShortestPaths(&_graph), 
                     Galois::loopname("NumShortestPaths"));

      trim_flags.dst_write = true;
      num_shortest_paths_flags.dst_write = true;
      propogation_flag_flags.src_write = true;

      // do predecessor decrementing using trim
      PredecessorDecrement::go(_graph);

      iterations++;
    } while (DGAccumulator_accum.reduce());

  }

  /* If a source has no more predecessors, then its shortest path value is
   * complete.
   *
   * Propogate the shortest path value through all outgoing edges where this
   * source is a predecessor in the DAG, then
   * set a flag saying that we should not propogate it any more (otherwise you
   * send extra).
   *
   * Additionally, decrement the predecessor field on the destination nodes of
   * the outgoing edges.
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.num_predecessors == 0 && !src_data.propogation_flag) {
      for (auto current_edge = graph->edge_begin(src), 
                end_edge = graph->edge_end(src); 
           current_edge != end_edge; 
           ++current_edge) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        // TODO change back from BFS
        //unsigned int edge_weight = graph->getEdgeData(current_edge);
        unsigned int edge_weight = 1;

        unsigned int to_add = src_data.num_shortest_paths;

        if ((src_data.current_length + edge_weight) == dst_data.current_length) {
          // add my num shortest paths to dest's num shortest paths
          Galois::atomicAdd(dst_data.num_shortest_paths, to_add);
          
          // increment dst trim so it can decrement predecessor
          Galois::atomicAdd(dst_data.trim, (unsigned int)1);
          bitset_update.set(dst);

          DGAccumulator_accum += 1;
        }
      }

      // set flag so that it doesn't propogate its info more than once
      src_data.propogation_flag = true;
    }
  }
};
Galois::DGAccumulator<unsigned int> NumShortestPaths::DGAccumulator_accum;


/* Loop over all nodes and set the prop flag to false in prep for more
 * propogation later */
struct PropFlagReset {
  Graph* graph;

  PropFlagReset(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    Galois::do_all(_graph.begin(), _graph.end(), PropFlagReset{&_graph}, 
                   Galois::loopname("PropFlagReset"), 
                   Galois::numrun(_graph.get_run_identifier()));

    propogation_flag_flags.src_write = true;
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    src_data.propogation_flag = false;
    bitset_update_flag.set(src);
  }
};

/* Uses an incremented trim value to decrement the successor: the trim value
 * has to be synchronized across ALL nodes (including slaves) */
struct SuccessorDecrement {
  Graph* graph;

  SuccessorDecrement(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    // READ SRC trim; note at this stage we don't increment dst trim anymore
    // so this shouldn't matter
    if (trim_flags.dst_write) {
      std::cerr << "BIG ISSUE, shouldn't be incrementing trim on dst\n";
      abort();
      // SHOULD NEVER GET IN HERE AT THIS POINT
      //_graph.sync_forward<SyncPushTrim, SyncPullTrim>("SuccessorDecrement");
      //resetFlags(trim_flags);
    }

    // READ SRC successors; shouldn't need to sync at this point as dst
    // successors are never written at this point in the program

    // READ SRC flag; dst flag will never be set so sync doesn't matter

    Galois::do_all(_graph.begin(), _graph.end(), SuccessorDecrement{&_graph}, 
                   Galois::loopname("SuccessorDecrement"), 
                   Galois::numrun(_graph.get_run_identifier()));

    trim_flags.src_write = true;
    propogation_flag_flags.src_write = true;
    num_successors_flags.src_write = true;
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.num_successors == 0 && !src_data.propogation_flag) {
      assert(src_data.trim == 0);
      src_data.propogation_flag = true;
    } else if (src_data.trim > 0) {
    // decrement successor by trim then reset
      if (src_data.trim > src_data.num_successors) {
        std::cerr << "ISSUEsucc: src " << src << " " << src_data.trim << " " << 
                                  src_data.num_successors << "\n";
        abort();                                    
      }

      src_data.num_successors -= src_data.trim;
      src_data.trim = 0;
      bitset_update_succ.set(src);

      // multiply dependency by # of shortest paths to finalize if no more 
      // successors, then set prop flag to false so it can propogate the value
      if (src_data.num_successors == 0) {
        // TODO revert back to this if necessary
        //src_data.dependency = src_data.dependency * src_data.num_shortest_paths;
        src_data.propogation_flag = false;
        bitset_update_flag.set(src);
      }
    }
  }
};

/* Do dependency propogation which is required for betweeness centraility
 * calculation */
struct DependencyPropogation {
  const unsigned int &local_current_src_node;
  Graph* graph;
  static Galois::DGAccumulator<unsigned int> DGAccumulator_accum;

  DependencyPropogation(const unsigned int &_local_current_src_node,
                        Graph* _graph) : 
                          local_current_src_node(_local_current_src_node),
                          graph(_graph){}

  /* Look at all nodes to do propogation until no more work is done */
  void static go(Graph& _graph) {
    ////////////////////////////////////////////////////////////////////////////
    // # short paths
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushPaths {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_shortest_paths_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_shortest_paths;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {

      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o,
                                                       y, s, data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_num_shortest_paths_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          add_node_num_shortest_paths_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        Galois::add(node.num_shortest_paths, y); return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_add_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s, 
                                                 data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif

        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          set_node_num_shortest_paths_cuda(cuda_ctx, node_id, 0);
        }
        else if (personality == CPU)
      #endif
        Galois::set(node.num_shortest_paths, (unsigned int)0);
      }
    };

    struct SyncPullPaths {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_shortest_paths_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif

        return node.num_shortest_paths;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_shortest_paths_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif

        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_num_shortest_paths_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.num_shortest_paths, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Succ
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushSucc {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_successors_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_successors;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, 
                                      unsigned int *y, 
                                      size_t *s, 
                                      DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          batch_get_reset_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_reset_node_num_successors_cuda(cuda_ctx, from_id, y, 0);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          add_node_num_successors_cuda(cuda_ctx, node_id, y);
          return true;
        } 
        //else if (personality == CPU)
        assert(personality == CPU);
      #endif
        { return Galois::add(node.num_successors, y); return true; }
      }

      static bool reduce_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {

          batch_add_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          set_node_num_predecessors_cuda(cuda_ctx, node_id, 0);
        }
        else if (personality == CPU)
      #endif
        Galois::set(node.num_successors, (unsigned int)0);
      }
    };

    struct SyncPullSucc {
      typedef unsigned int ValTy;

      static unsigned int extract(uint32_t node_id, 
                                  const struct NodeData & node) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) 
          return get_node_num_successors_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
      #endif
        return node.num_successors;
      }

      static bool extract_batch(unsigned from_id,
                                unsigned long long int *b,
                                unsigned int *o,
                                unsigned int *y,
                                size_t *s, 
                                DataCommMode *data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s,
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_get_node_num_successors_cuda(cuda_ctx, from_id, y);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         unsigned int y) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_num_successors_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
      #endif
        Galois::set(node.num_successors, y);
      }

      static bool setVal_batch(unsigned from_id, 
                               unsigned long long int *b, 
                               unsigned int *o, 
                               unsigned int *y, 
                               size_t s, 
                               DataCommMode data_mode) {
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          batch_set_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                   data_mode);
          return true;
        }
        assert (personality == CPU);
      #endif
        return false;
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Flag
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushFlag {
      typedef unsigned int ValTy;

      static bool extract(uint32_t node_id, const struct NodeData & node) {
        return node.propogation_flag;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, 
                         bool y) {
        // shouldn't even get here in the first place as you shouldn't be 
        // reducing flags in this algorithm)
        std::cerr << "reducing a flag, shouldn't do this!\n";
        abort();
        return false;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
        node.propogation_flag = false;
      }
    };

    struct SyncPullFlag {
      typedef unsigned int ValTy;

      static bool extract(uint32_t node_id, const struct NodeData & node) {
        return node.propogation_flag;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, bool y) {
        Galois::set(node.propogation_flag, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

    };


    ////////////////////////////////////////////////////////////////////////////
    // Dependency
    ////////////////////////////////////////////////////////////////////////////
    struct SyncPushDependency {
      typedef unsigned int ValTy;

      static float extract(uint32_t node_id, const struct NodeData & node) {
        return node.dependency;
      }

      static bool extract_reset_batch(unsigned from_id, 
                                      unsigned long long int *b, 
                                      unsigned int *o, unsigned int *y, 
                                      size_t *s, DataCommMode *data_mode) {
        return false;
      }

      static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static bool reduce(uint32_t node_id, struct NodeData & node, float y) {
        Galois::add(node.dependency, y); return true;
      }

      static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }

      static void reset (uint32_t node_id, struct NodeData & node) {
        Galois::set(node.dependency, (float)0);
      }
    };

    struct SyncPullDependency {
      typedef unsigned int ValTy;

      static float extract(uint32_t node_id, const struct NodeData & node) {
        return node.dependency;
      }

      static bool extract_batch(unsigned from_id, unsigned long long int *b,
                                unsigned int *o, unsigned int *y, size_t *s, 
                                DataCommMode *data_mode) {
        return false;
      }

      static bool extract_batch(unsigned from_id, unsigned int *y) {
        return false;
      }

      static void setVal(uint32_t node_id, struct NodeData & node, 
                         float y) {
        Galois::set(node.dependency, y);
      }

      static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                               unsigned int *o, unsigned int *y, size_t s, 
                               DataCommMode data_mode) {
        return false;
      }
    };


    ////////////////////////////////////////////////////////////////////////////

    unsigned int iterations = 0;
    
    do {
      _graph.set_num_iter(iterations);
      DGAccumulator_accum.reset();

      // READ SRC/DST succ
      if (num_successors_flags.src_write && num_successors_flags.dst_write) {
        // big problem: shouldn't be writing to dst succ anymore
        std::cerr << "num successors dst flag set\n";
        abort();
      } else if (num_successors_flags.src_write) {
        _graph.sync_backward<SyncPushSucc, 
                             SyncPullSucc>("DependencyPropogation");
        //_graph.sync_backward<SyncPushSucc, 
        //                     SyncPullSucc>("DependencyPropogation", 
        //                                   bitset_update_succ);
        resetFlags(num_successors_flags);
        bitset_update_succ.clear();
      } else if (num_successors_flags.dst_write) {
        // big problem: shouldn't be writing to dst succ anymore
        std::cerr << "num successors dst flag set\n";
        abort();
      }

      // READ SRC/DST current length; 
      // NOTE at this point it shouldn't matter
      // anymore as they should have been sync'd already
      //if (current_length_flags.src_write && current_length_flags.dst_write) {
      //  std::cerr << "in current length src/dst\n";
      //  abort();
      //  _graph.sync_exchange<SyncPushLength, 
      //                       SyncPullLength>("DependencyPropogation");
      //  resetFlags(current_length_flags);
      //} else if (current_length_flags.src_write) {
      //  std::cerr << "in current length src\n";
      //  abort();
      //  _graph.sync_backward<SyncPushLength, 
      //                       SyncPullLength>("DependencyPropogation");
      //  resetFlags(current_length_flags);
      //} else if (current_length_flags.dst_write) {
      //  std::cerr << "in current length dst\n";
      //  abort();
      //  _graph.sync_exchange<SyncPushLength, 
      //                       SyncPullLength>("DependencyPropogation");
      //  resetFlags(current_length_flags);
      //} 

      // READ DST prop flag
      if (propogation_flag_flags.src_write && 
          propogation_flag_flags.dst_write) {
        // big problem: shouldn't be writing to dst flag EVER
        std::cerr << "propogation flag dst flag set\n";
        abort();
      } else if (propogation_flag_flags.src_write) {
        _graph.sync_backward<SyncPushFlag, 
                             SyncPullFlag>("DependencyPropogation");
        //_graph.sync_backward<SyncPushFlag, 
        //                     SyncPullFlag>("DependencyPropogation",
        //                                   bitset_update_flag);
        bitset_update_flag.clear();
        resetFlags(propogation_flag_flags);
      } else if (propogation_flag_flags.dst_write) {
        // big problem: shouldn't be writing to dst flag EVER
        std::cerr << "propogation flag dst flag set\n";
        abort();
      }

      // READ SRC/DST # shortest paths
      if (num_shortest_paths_flags.src_write && 
          num_shortest_paths_flags.dst_write) {
        // should never get in here
        std::cerr << "num short paths src\n";
        abort();
        _graph.sync_exchange<SyncPushPaths, 
                             SyncPullPaths>("DependencyPropogation");
        resetFlags(num_shortest_paths_flags);
      } else if (num_shortest_paths_flags.src_write) {
        // should never get in here
        std::cerr << "num short paths src\n";
        abort();
        _graph.sync_backward<SyncPushPaths, 
                             SyncPullPaths>("DependencyPropogation");
        resetFlags(num_shortest_paths_flags);
      } else if (num_shortest_paths_flags.dst_write) {
        _graph.sync_exchange<SyncPushPaths, 
                             SyncPullPaths>("DependencyPropogation");
        //_graph.sync_exchange<SyncPushPaths, 
        //                     SyncPullPaths>("DependencyPropogation",
        //                                    bitset_update);
        resetFlags(num_shortest_paths_flags);
      }

      // READ DST dependency
      if (dependency_flags.src_write && dependency_flags.dst_write) {
        // shouldn't get here
        std::cerr << "dep dst\n";
        abort();
        _graph.sync_exchange<SyncPushDependency, 
                             SyncPullDependency>("DependencyPropogation");
        resetFlags(dependency_flags);
      } else if (dependency_flags.src_write) {
        _graph.sync_backward<SyncPushDependency, 
                             SyncPullDependency>("DependencyPropogation");
        //_graph.sync_backward<SyncPushDependency, 
        //                     SyncPullDependency>("DependencyPropogation",
        //                                         bitset_update);

        resetFlags(dependency_flags);
      } else if (dependency_flags.dst_write) {
        // shouldn't get here
        std::cerr << "dep dst\n";
        abort();
        _graph.sync_exchange<SyncPushDependency, 
                             SyncPullDependency>("DependencyPropogation");
        resetFlags(dependency_flags);
      }

      bitset_update.clear();

      Galois::do_all(_graph.begin(), _graph.end(), 
                     DependencyPropogation(current_src_node, &_graph), 
                     Galois::loopname("DependencyPropogation"));

      dependency_flags.src_write = true;
      trim_flags.src_write = true;

      // do successor decrementing using trim
      SuccessorDecrement::go(_graph);

      iterations++;
    } while (DGAccumulator_accum.reduce());
  }

  /* Summary:
   * TOP based, but can filter if successors = 0; can do trim based decrement
   * like kcore
   * if we have outgoing edges...
   * for each node, check if dest of edge has no successors + check if on 
   * shortest path with src as predeccesor
   *
   * if yes, then decrement src successors by 1 + grab dest delta + dest num 
   * shortest * paths and use it to increment src own delta (1 / dest short 
   * paths * (1 + delta of dest)
   *
   * sync details: push src delta changes, src sucessor changes (via trim) 
   * to ALL COPIES (not just master)
   *
   * dest to src flow for successors
   * dest to src flow for delta
   **/
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // IGNORE THE SOURCE NODE OF THIS CURRENT ITERATION OF SSSP
    // + do not redo computation if src has no successors left
    if (graph->getGID(src) == local_current_src_node || src_data.num_successors == 0) {
      return;
    }

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);
      // TODO change back from BFS
      //unsigned int edge_weight = graph->getEdgeData(current_edge);
      unsigned int edge_weight = 1;
      //std::cout << "dest " << graph->getGID(dst) << " " 
      //          << dst_data.propogation_flag << "\n";

      // only operate if a dst has no more successors (i.e. delta finalized
      // for this round) + if it hasn't propogated the value yet (i.e. because
      // successors become 0 in the last do_all round)
      if (dst_data.num_successors == 0 && !dst_data.propogation_flag) {
        // dest on shortest path with this node as predecessor
        if ((src_data.current_length + edge_weight) == dst_data.current_length) {
          // increment my trim for later use to decrement successor
          Galois::atomicAdd(src_data.trim, (unsigned int)1);

          // update my dependency
          // TODO revert to saving multiplication till the end?
          // cuda gen makes me have all of this in one line; quite annoying..
          src_data.dependency = src_data.dependency + (((float)src_data.num_shortest_paths / (float)dst_data.num_shortest_paths) * (float)(1.0 + dst_data.dependency));

          bitset_update.set(src);

          DGAccumulator_accum += 1;
        }
      }
    }
  }
};
Galois::DGAccumulator<unsigned int> DependencyPropogation::DGAccumulator_accum;

struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    uint64_t start_i;
    uint64_t end_i;

    if (singleSourceBC) {
      start_i = singleSourceNode;
      end_i = singleSourceNode + 1;
    } else {
      start_i = 0;
      end_i = _graph.totalNodes;
    }

    for (uint64_t i = start_i; i < end_i; i++) {
      // change source nodes for this iteration of SSSP
      //if (i % 5000 == 0) {
      //  std::cout << "SSSP source node " << i << "\n";
      //}

      current_src_node = i;
      std::cout << "SSSP source node " << i << "\n";

      // reset the graph aside from the between-cent measure
      InitializeIteration::go(_graph);

      // get SSSP on the current graph
      SSSP::go(_graph);

      // calculate the succ/pred for all nodes in the SSSP DAG
      PredAndSucc::go(_graph);

      // calculate the number of shortest paths for each node
      NumShortestPaths::go(_graph);

      // RESET PROP FLAG
      PropFlagReset::go(_graph);

      // do between-cent calculations for this iteration 
      DependencyPropogation::go(_graph);

      // READ SRC dependencies
      if (dependency_flags.dst_write) {
        std::cerr << "dependency shouldn't be written on dst\n";
        abort();
      }

      // finally, since dependencies are finalized for this round at this 
      // point, add them to the betweeness centrality measure on each node
      Galois::do_all(_graph.begin(), _graph.end(), BC(&_graph), 
                     Galois::loopname("BC"));

      // WRITE SRC bc; unnecessary to set since you will never write
      // bc dst + it's over
    }
  }

  /* adds dependency measure to BC measure (dependencies should be finalized,
   * i.e. no unprocessed successors on the node) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.betweeness_centrality = src_data.betweeness_centrality + src_data.dependency;
  }
};
 
/******************************************************************************/
/* Main method for running */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    Galois::StatManager statManager;

    auto& net = Galois::Runtime::getSystemNetworkInterface();
    if (net.ID == 0) {
      Galois::Runtime::reportStat("(NULL)", "Max Iterations", 
                                  (unsigned long)maxIterations, 0);
    }

    Galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT"),
                      StatTimer_total("TIMER_TOTAL"),
                      StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;
  #ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = Galois::Runtime::getHostID();
    int gpu_device = gpudevice;

    if (num_nodes == -1) num_nodes = net.Num;
    assert((net.Num % num_nodes) == 0);

    // Parse arg string when running on multiple hosts and update/override 
    // personality with corresponding value.
    if (personality_set.length() == Galois::Runtime::NetworkInterface::Num) {
      switch (personality_set.c_str()[my_host_id]) {
        case 'g':
          personality = GPU_CUDA;
          break;
        case 'o':
          assert(0); // o currently not supported
          personality = GPU_OPENCL;
          break;
        case 'c':
        default:
          personality = CPU;
          break;
      }

      if ((personality == GPU_CUDA) && (gpu_device == -1)) {
        gpu_device = get_gpu_device_id(personality_set, num_nodes);
      }

      if ((scalecpu > 1) || (scalegpu > 1)) {
        for (unsigned i = 0; i < net.Num; ++i) {
          if (personality_set.c_str()[i % num_nodes] == 'c') 
            scalefactor.push_back(scalecpu);
          else
            scalefactor.push_back(scalegpu);
        }
      }
    }
  #endif

    StatTimer_hg_init.start();

    Graph* h_graph;
    if (enableVCut) {
      if (vertexcut == CART_VCUT)
        h_graph = new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num,
                                         scalefactor, transpose);
      else if (vertexcut == PL_VCUT)
        h_graph = new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                      scalefactor, transpose, VCutThreshold);
    } else {
      h_graph = new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scalefactor);
    }

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = (*h_graph).getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      //Galois::OpenCL::cl_env.init(cldevice.Value);
    }
  #endif

    bitset_update.resize(h_graph->get_local_total_nodes());
    bitset_update_succ.resize(h_graph->get_local_total_nodes());
    bitset_update_flag.resize(h_graph->get_local_total_nodes());

    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";

    StatTimer_graph_init.start();
      InitializeGraph::go((*h_graph));
    StatTimer_graph_init.stop();

    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] BC::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      Galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        BC::go((*h_graph));
      StatTimer_main.stop();

      // re-init graph for next run
      if ((run + 1) != numRuns) {
        Galois::Runtime::getHostBarrier().wait();
        (*h_graph).reset_num_iter(run+1);
        InitializeGraph::go((*h_graph));
      }
    }

    StatTimer_total.stop();

    // Verify, i.e. print out graph data for examination
    if (verify) {
      char test[100];
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) {
            //sprintf(test, "%lu %.9f %u\n", (*h_graph).getGID(*ii),
            //        (*h_graph).getData(*ii).betweeness_centrality,
            //        (*h_graph).getData(*ii).current_length.load());
            sprintf(test, "%lu %.9f\n", (*h_graph).getGID(*ii),
                    (*h_graph).getData(*ii).betweeness_centrality);
            Galois::Runtime::printOutput(test);
            // outputs betweenness centrality
            //Galois::Runtime::printOutput("% %\n", (*h_graph).getGID(*ii), 
            //                     (*h_graph).getData(*ii).betweeness_centrality);
          }
        }
#ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA) {
        for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
            sprintf(test, "%lu %.9f\n", (*h_graph).getGID(*ii),
                    get_node_betweeness_centrality_cuda(cuda_ctx, *ii));

            Galois::Runtime::printOutput(test);
        }
      }
#endif
    }
    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
