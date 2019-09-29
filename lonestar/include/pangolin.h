#ifndef PANGOLIN
#define PANGOLIN
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Profile.h"
#include <boost/iterator/transform_iterator.hpp>

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: unsymmetrized graph>"), cll::Required);
#ifdef USE_QUERY_GRAPH
static cll::opt<std::string> query_graph_filename(cll::Positional, cll::desc("<filename: unsymmetrized graph>"), cll::Required);
static cll::opt<std::string> preset_filename("pf", cll::desc("<filename: preset matching order>"), cll::init("default"));
#endif
#ifndef TRIANGLE
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-clique (default value 3)"), cll::init(3));
#endif
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
static cll::opt<unsigned> debug("d", cll::desc("print out the frequent patterns for debugging"), cll::init(0));
#ifdef EDGE_INDUCED
static cll::opt<unsigned> minsup("ms", cll::desc("minimum support (default value 0)"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#else
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#endif
typedef Graph::GraphNode GNode;

#ifdef USE_DFS
#ifdef EDGE_INDUCED
#ifdef USE_DFSCODE
#include "Dfscode/miner.h"
#else
#include "DfsMining/edge_miner.h"
#endif
#else
#include "DfsMining/vertex_miner.h"
#endif
#else
#ifdef EDGE_INDUCED
#include "BfsMining/edge_miner.h"
#else
#include "BfsMining/vertex_miner.h"
#endif
#endif

#include "util.h"

#endif
