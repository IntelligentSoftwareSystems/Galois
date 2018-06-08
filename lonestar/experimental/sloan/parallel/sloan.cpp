/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
#include "galois/Bag.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "Lonestar/BoilerPlate.h"

//kik 
#include "galois/Atomic.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <deque>
#include <cmath>
#include <functional>
#include <numeric>

#include <sys/time.h>

#include <omp.h>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/sloan_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>
#include <boost/graph/profile.hpp>
#include <boost/graph/wavefront.hpp>

#include <queue>
#include <algorithm>
#include <boost/pending/queue.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/properties.hpp>
#include <boost/pending/indirect_cmp.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>

#include <boost/heap/priority_queue.hpp>

#define FINE_GRAIN_TIMING 
//#define GALOIS_JUNE
//#define PRINT_DEGREE_DISTR

#define W1 1               //default weight for the distance in the Sloan algorithm
#define W2 2               //default weight for the degree in the Sloan algorithm

static const char* name = "Sloan's reordering algorithm";
static const char* desc = "Computes a permutation of a matrix according to Sloan's algorithm";
static const char* url = 0;

//****** Command Line Options ******
enum BFSAlgo {
	serialSloan,
	//barrierSloan,
};

enum ExecPhase {
	INIT,
	RUN,
	TOTAL,
};

enum Statuses {
	INACTIVE,
	PREACTIVE,
	ACTIVE,
	NUMBERED,
};

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from"),
    cll::init(0));
static cll::opt<unsigned int> terminalNode("terminalnode",
    cll::desc("Terminal Node to find distance to"),
    cll::init(0));
static cll::opt<bool> scaling("scaling", 
		llvm::cl::desc("Scale to the number of threads with a given step starting from"), 
		llvm::cl::init(false));
static cll::opt<unsigned int> scalingStep("step",
    cll::desc("Scaling step"),
    cll::init(2));
static cll::opt<unsigned int> niter("iter",
    cll::desc("Number of benchmarking iterations"),
    cll::init(5));
static cll::opt<BFSAlgo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serialSloan, "Serial Sloan"),
      //clEnumVal(barrierSloan, "Barrier-based Parallel Sloan"),
      clEnumValEnd), cll::init(serialSloan));
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

//****** Work Item and Node Data Defintions ******
struct SNode {
	unsigned int id;
  unsigned int dist;
	unsigned int degree;
	unsigned int status;
	int prio;
	galois::substrate::SimpleLock mutex;
	//galois::GAtomic<unsigned int>& status;
	//unsigned int sum;
  //bool flag;
	//std::vector<galois::graphs::LC_CSR_Graph<SNode, void>::GraphNode> bucket;
};

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out <<  "(dist: " << n.dist << ")";
  return out;
}

typedef galois::graphs::LC_CSR_Graph<SNode, void>::_with_no_lockable<true>::_with_numa_alloc<true> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

static size_t degree(const GNode& node) { 
    return std::distance(graph.edge_begin(node), graph.edge_end(node));
}

struct UpdateRequest {
	GNode node;
	int prio;

	UpdateRequest(): prio(0) { }
	UpdateRequest(const GNode& N, int P): node(N), prio(P) { }

	//bool operator==(const UpdateRequest& o) const { return (node == o.node) && (prio == o.prio); }
	//bool operator!=(const UpdateRequest& o) const { return !operator==(o); }
	//inline bool operator<=(const X& lhs, const X& rhs){return !operator> (lhs,rhs);} 
	//inline bool operator>=(const X& lhs, const X& rhs){return !operator< (lhs,rhs);}
	//bool operator<(const UpdateRequest& o) const { return prio < o.prio; }
	//bool operator>(const UpdateRequest& o) const { return prio > o.prio; }
	//unsigned getID() const { return /* graph.getData(n).id; */ 0; }
};

struct UpdateRequestIndexer: public std::unary_function<UpdateRequest,int> {
	int operator()(const UpdateRequest& val) const {
		int p = val.prio;
		return -p;
	}
};

struct UpdateRequestLess {
  bool operator()(const UpdateRequest& a, const UpdateRequest& b) const {
    return a.prio <= b.prio;
		/*
		if(a.prio == b.prio)
			return graph.getData(a.node, galois::MethodFlag::UNPROTECTED).degree <= graph.getData(b.node, galois::MethodFlag::UNPROTECTED).degree;
    return a.prio < b.prio;
		*/
  }
};

struct GNodeIndexer: public std::unary_function<GNode,int> {
  int operator()(const GNode& node) const {
    return graph.getData(node, galois::MethodFlag::UNPROTECTED).prio;
  }
};

struct GNodeLess {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, galois::MethodFlag::UNPROTECTED).prio < graph.getData(b, galois::MethodFlag::UNPROTECTED).prio;
  }
};

struct GNodeGreater {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, galois::MethodFlag::UNPROTECTED).prio > graph.getData(b, galois::MethodFlag::UNPROTECTED).prio;
  }
};

struct GNodeBefore {
  bool operator()(const GNode& a, const GNode& b) const {
    return (degree(a) < degree(b));
  }
};

unsigned int maxDist;
std::vector<GNode> perm;
std::priority_queue<UpdateRequest, std::vector<UpdateRequest>, UpdateRequestLess> pq;
//std::set<UpdateRequest, std::greater<UpdateRequest> > pq;
//std::multiset<UpdateRequest, UpdateRequestGreater> pq;

static void printSloan(){
	std::cerr << "Sloan Permutation:\n";

	for(std::vector<GNode>::iterator nit = perm.begin(); nit != perm.end(); nit++){
		SNode& data = graph.getData(*nit);
		//std::cerr << "[" << data.id << "] level: " << data.dist << " degree: " << data.degree << "\n";
		//std::cerr << data.id + 1 << " (" << data.degree << ") level: " << data.dist << "\n";
		//std::cerr << data.id + 1 << "\n";
		std::cerr << data.id << "\n";
	}
	std::cerr << "\n";
}

static void printRSloan(){
	std::cerr << "Reverse Sloan Permutation:\n";
	for(std::vector<GNode>::reverse_iterator nit = perm.rbegin(); nit != perm.rend(); nit++){
		SNode& data = graph.getData(*nit);
		//std::cerr << "[" << data.id << "] level: " << data.dist << " degree: " << data.degree << "\n";
		//std::cerr << data.id + 1 << " (" << data.degree << ") level: " << data.dist << "\n";

		std::cerr << data.id << " (" << degree(*nit) << ") level: " << data.dist << "\n";
	}
	std::cerr << "\n";
}

static void permute() {

	std::vector<GNode> nodemap;
	nodemap.reserve(graph.size());;

	for (Graph::iterator src = graph.begin(), ei =
			graph.end(); src != ei; ++src) {

		nodemap[graph.getData(*src).id] = *src;
	}

	unsigned int N = perm.size() - 1;

	for(int i = N; i >= 0; --i) {
		//std::cerr << perm[i] << " " << graph.getData(nodemap[permid[i]]).id << " changes to: " << N - i << "\n";
		graph.getData(perm[i]).id = N - i;
	}
}

//debugging 
static void printAccess(std::string msg){
	std::cerr << msg << " Access Pattern:\n";

	std::vector<unsigned int> temp;

	for (Graph::iterator src = graph.begin(), ei =
			graph.end(); src != ei; ++src) {

		SNode& sdata = graph.getData(*src);

		std::cerr << sdata.id << " connected with (" << degree(*src) << "): ";

		for (Graph::edge_iterator ii = graph.edge_begin(*src, galois::MethodFlag::UNPROTECTED), 
				ei = graph.edge_end(*src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
			GNode dst = graph.getEdgeDst(ii);
			SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

			unsigned int diff = abs(sdata.id - ddata.id);

			std::cerr << ddata.id << " (" << diff << "), ";
		}

		std::cerr << "\n";
		//std::cerr << data.id << " (" << degree(*src) << ") level: " << data.dist << " reads: " << data.read << " writes: " << data.write << "\n";
		//std::cerr << data.id << " (" << degree(*src) << ") level: " << data.dist << "\n";

		temp.push_back(sdata.id);
	}

	//for(std::vector<unsigned int>::reverse_iterator nit = temp.rbegin(); nit != temp.rend(); nit++)
	for(std::vector<unsigned int>::iterator nit = temp.begin(); nit != temp.end(); nit++){
		std::cerr << *nit + 1 << "\n";
	}
	std::cerr << "\n";
}

static void findStartingNode(GNode& starting) {
	unsigned int mindegree = DIST_INFINITY; 

	for (Graph::iterator src = graph.begin(), ei =
			graph.end(); src != ei; ++src) {
		unsigned int nodedegree = degree(*src);

		if(nodedegree < mindegree){
			mindegree = nodedegree;
			starting = *src;
		}
	}

	SNode& data = graph.getData(starting);
	std::cerr << "Starting Node: " << data.id << " degree: " << degree(starting) << "\n";
}

//Compute variance around mean distance from the source
static void variance(unsigned long int mean) {
	unsigned long int n = 0;
	long double M2 = 0.0;
	long double var = 0.0;

	for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei; ++src) {
		SNode& data = graph.getData(*src);
		if(data.dist < DIST_INFINITY){
			M2 += (data.dist - mean)*(data.dist - mean);
			++n;
		}
	}

	var = M2/(n-1);
	std::cout << "var: " << var << " mean: " << mean << "\n";
}

struct not_consistent {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(dst).dist;
      if (ddist > dist + 1) {
        std::cerr << "bad level value for " << graph.getData(dst).id << ": " << ddist << " > " << (dist + 1) << "\n";
	return true;
      }
    }
    return false;
  }
};

struct not_visited {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr << "unvisited node " << graph.getData(n).id << ": " << dist << " >= INFINITY\n";
      return true;
    }
		//std::cerr << "visited node " << graph.getData(n).id << ": " << dist << "\n";
    return false;
  }
};


//! Simple verifier
static bool verify(GNode& source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  
  size_t id = 0;
  
#ifdef GALOIS_JUNE
  bool okay = galois::find_if(graph.begin(), graph.end(), not_consistent()) == graph.end()
    && galois::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();
#else
  bool okay = galois::ParallelSTL::find_if(graph.begin(), graph.end(), not_consistent()) == graph.end()
    && galois::ParallelSTL::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();
#endif

  //if (okay) {
    galois::GReduceMax<unsigned int> maxDist;
    galois::GAccumulator<unsigned> sum;
    galois::GAccumulator<unsigned> count;
    
    galois::do_all(galois::iterate(graph),
        [&] (const GNode& n) {
          auto d = graph.getData(n, galois::MethodFlag::UNPROTECTED).dist;
          if (d < INFINITY) {
            maxDist.update(d);
            sum += d;
            count += 1;
          }
        });
    std::cout << "max dist: " << maxDist.reduce() << "\n";
    unsigned mean = sum.reduce() / count.reduce() ;
    std::cout << "avg dist: " << mean << "\n";

		variance(mean.reduce());
  //}
  
  return okay;
}

// Compute maximum bandwidth for a given graph
struct banddiff {

	galois::GAtomic<unsigned int>& maxband;
  banddiff(galois::GAtomic<unsigned int>& _mb): maxband(_mb) { }

  void operator()(const GNode& source) const {

		SNode& sdata = graph.getData(source, galois::MethodFlag::UNPROTECTED);
		for (Graph::edge_iterator ii = graph.edge_begin(source, galois::MethodFlag::UNPROTECTED), 
				 ei = graph.edge_end(source, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

			unsigned int diff = abs(sdata.id - ddata.id);
			unsigned int max = maxband;

			if(diff > max){
				while(!maxband.cas(max, diff)){
					max = maxband;
					if(!diff > max)
						break;
				}
			}
    }
  }
};

// Compute maximum bandwidth for a given graph
struct profileFn {

	galois::GAtomic<unsigned int>& sum;
  profileFn(galois::GAtomic<unsigned int>& _s): sum(_s) { }

	void operator()(const GNode& source) const {

		unsigned int max = 0;
		SNode& sdata = graph.getData(source, galois::MethodFlag::UNPROTECTED);

		for (Graph::edge_iterator ii = graph.edge_begin(source, galois::MethodFlag::UNPROTECTED), 
				ei = graph.edge_end(source, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

			GNode dst = graph.getEdgeDst(ii);
			SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

			unsigned int diff = abs(sdata.id - ddata.id);

			max = (diff > max) ? diff : max;
		}

		sum += (max + 1);
	}
};

// Parallel loop for maximum bandwidth computation
static void bandwidth(std::string msg) {
		galois::GAtomic<unsigned int> maxband = galois::GAtomic<unsigned int>(0);
    galois::do_all(graph.begin(), graph.end(), banddiff(maxband));
    std::cout << msg << " Bandwidth: " << maxband << "\n";
}

// Parallel loop for maximum bandwidth computation
static void profile(std::string msg) {
		galois::GAtomic<unsigned int> prof = galois::GAtomic<unsigned int>(0);
    galois::do_all(graph.begin(), graph.end(), profileFn(prof));
    std::cout << msg << " Profile: " << prof << "\n";
}
//Clear node data to re-execute on specific graph
struct resetNode {
	void operator()(const GNode& n) const {
		graph.getData(n).dist = DIST_INFINITY;
		//graph.getData(n).flag = false;
		//graph.getData(n).bucket.clear();
	}
};

static void resetGraph() {
	galois::do_all(graph.begin(), graph.end(), resetNode());
	perm.clear();
}

static void printDegreeDistribution() {
	std::map<unsigned int, unsigned int> distr;

	for (Graph::iterator n = graph.begin(), ei = graph.end(); n != ei; ++n) {
			distr[degree(*n)]++;
			//std::cerr << graph.getData(*n, galois::MethodFlag::UNPROTECTED).id << "	" << graph.getData(*n, galois::MethodFlag::UNPROTECTED).dist << "\n";
	}

	std::cerr << "Degree	Count\n";
	for (std::map<unsigned int, unsigned int>::iterator slot = distr.begin(), ei = distr.end(); slot != ei; ++slot) {
		std::cerr << slot->first << "	" << slot->second << "\n";
	}
}

// Read graph from a binary .gr as dirived from a Matrix Market .mtx using graph-convert
static void readGraph(GNode& source, GNode& terminal) {
  galois::graphs::readGraph(graph, filename);

  source = *graph.begin();
  terminal = *graph.begin();

  size_t nnodes = graph.size();
  std::cout << "Read " << nnodes << " nodes\n";
  
  size_t id = 0;
  bool foundTerminal = false;
  bool foundSource = false;

	perm.reserve(nnodes);

  for (Graph::iterator src = graph.begin(), ei =
      graph.end(); src != ei; ++src) {

    SNode& node = graph.getData(*src, galois::MethodFlag::UNPROTECTED);
    node.dist = DIST_INFINITY;
    node.id = id;
    //node.status = INACTIVE;
    //node.flag = false;;
		//node.read = galois::GAtomic<unsigned int>(0);
		//node.write = galois::GAtomic<unsigned int>(0);

    //std::cout << "Terminal node: " << terminalNode << " (dist: " << distances[terminalNode] << ")\n";

    if (id == startNode) {
      source = *src;
      foundSource = true;
    } 
    if (id == terminalNode) {
      foundTerminal = true;
      terminal = *src;
    }
    ++id;
  }

/*
	if(startNode == DIST_INFINITY){
		findStartingNode(source);
		foundSource = true;
	}
	*/

  if (!foundTerminal || !foundSource) {
    std::cerr 
      << "failed to set terminal: " << terminalNode 
      << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
}

//! Serial BFS using Galois graph
//Boost-based
struct SerialSloan {
	std::string name() const { return "Serial Sloan"; }

	struct bfsFn {
		typedef int tt_does_not_need_aborts;

		void operator()(GNode& n, galois::UserContext<GNode>& ctx) const {
			SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
			unsigned int old_max;
			while ((old_max = maxDist) < data.dist)
				__sync_bool_compare_and_swap(&maxDist, old_max, data.dist);

			unsigned int newDist = data.dist + 1;

			for (Graph::edge_iterator ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
					ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
				GNode dst = graph.getEdgeDst(ii);
				SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

				unsigned int oldDist;
				while (true) {
					oldDist = ddata.dist;
					if (oldDist <= newDist)
						break;
					if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
						ctx.push(dst);
						break;
					}
				}
			}
		}

		static void go(GNode source) {
			using namespace galois::worklists;
			typedef PerSocketChunkFIFO<64> PSchunk;
			typedef ChunkFIFO<64> Chunk;
			typedef OrderedByIntegerMetric<GNodeIndexer,PSchunk> OBIM;

			graph.getData(source).dist = 0;
			//std::cout << "max dist is " << maxDist << "\n";
			galois::for_each(source, bfsFn(), galois::loopname("BFS"), galois::wl<OBIM>());
			//std::cout << "max dist is " << maxDist << "\n";
		}
	};
	
	struct initFn {
		void operator()(const GNode& n) const {
			SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
			data.status = INACTIVE;
			//data.status = galois::GAtomic<unsigned int>(INACTIVE);
			data.degree = degree(n);
			data.prio = W1 * data.dist - W2 * (data.degree+1);
			//std::cerr << "[" << data.id << "] prio: " << data.prio << " dist: " << data.dist << " deg: " << data.degree << " skata: " << W1 * data.dist + W2 * (data.degree+1) << "\n";
			//graph.getData(n).flag = false;
			//graph.getData(n).bucket.clear();
		}

		static void go() {
			galois::do_all(graph.begin(), graph.end(), initFn());
		}
	};


	struct sloanFn {

		typedef int tt_does_not_need_aborts;
		typedef int tt_needs_parallel_break;

		void operator()(UpdateRequest& next, galois::UserContext<UpdateRequest>& ctx) const {

			unsigned int status;
			GNode parent = next.node;
			SNode& pdata = graph.getData(parent, galois::MethodFlag::UNPROTECTED);

			pdata.mutex.lock();
			//std::cerr << "[" << perm.size() << "] Extracting node: " << pdata.id << " with status: " << pdata.status << " and priority: " << pdata.prio << "\n";
			status = pdata.status;
			pdata.status = NUMBERED;
			pdata.mutex.unlock();

			if(status != NUMBERED){
				perm.push_back(parent);

				if(perm.size() == graph.size())
					ctx.breakLoop();

				//std::cerr << "[" << perm.size() << "] Will process node: " << pdata.id << " with status: " << pdata.status << " and priority: " << pdata.prio << "\n";

				if(status == PREACTIVE){
					for (Graph::edge_iterator ii = graph.edge_begin(parent, galois::MethodFlag::UNPROTECTED), ei = graph.edge_end(parent, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

						GNode child = graph.getEdgeDst(ii);
						SNode& cdata = graph.getData(child, galois::MethodFlag::UNPROTECTED);

						cdata.mutex.lock();
						if(cdata.status != NUMBERED){
							if(cdata.status == INACTIVE){
								cdata.status = PREACTIVE;
							}
							cdata.prio += W2;
							ctx.push(UpdateRequest(child, cdata.prio));
							//std::cerr << "[" << perm.size() << "] Adding: " << cdata.id << " with status: " << cdata.status << " and priority: " << cdata.prio << "\n";
						}
						cdata.mutex.unlock();

						/*
							 cdata.mutex.lock();
							 if(cdata.status == INACTIVE){
							 cdata.status = PREACTIVE;
							 cdata.prio += W2;
							 ctx.push(UpdateRequest(child, cdata.prio));
							 }
							 else if(cdata.status == PREACTIVE || cdata.status == ACTIVE){
							 cdata.prio += W2;
							 ctx.push(UpdateRequest(child, cdata.prio));
							 }
							 cdata.mutex.unlock();
							 */

						//ctx.push(UpdateRequest(graph.getEdgeDst(ii), cdata.prio));
					}
				}

				for (Graph::edge_iterator ii = graph.edge_begin(parent, galois::MethodFlag::UNPROTECTED), ei = graph.edge_end(parent, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

					GNode child = graph.getEdgeDst(ii);
					SNode& cdata = graph.getData(child, galois::MethodFlag::UNPROTECTED);

					if(cdata.status == PREACTIVE){
						cdata.mutex.lock();
						cdata.status = ACTIVE;
						cdata.prio += W2;
						cdata.mutex.unlock();

						ctx.push(UpdateRequest(child, cdata.prio));
						//std::cerr << "[" << perm.size() << "] Adding: " << cdata.id << " with status: " << cdata.status << " and priority: " << cdata.prio << "\n";

						for (Graph::edge_iterator ij = graph.edge_begin(child, galois::MethodFlag::UNPROTECTED), ej = graph.edge_end(child, galois::MethodFlag::UNPROTECTED); ij != ej; ++ij) {
							GNode grandchild = graph.getEdgeDst(ij);
							SNode& gdata = graph.getData(grandchild, galois::MethodFlag::UNPROTECTED);

							gdata.mutex.lock();
							if(gdata.status != NUMBERED){
								if(gdata.status == INACTIVE){
									gdata.status = PREACTIVE;
								}
								gdata.prio += W2;
								ctx.push(UpdateRequest(grandchild, gdata.prio));
								//std::cerr << "[" << perm.size() << "] Adding: " << gdata.id << " with status: " << gdata.status << " and priority: " << gdata.prio << "\n";
							}
							gdata.mutex.unlock();
						}
					}
				}
			}
			//else
				//std::cerr << "[" << perm.size() << "] Will throw node: " << pdata.id << " with status: " << pdata.status << " and priority: " << pdata.prio << "\n";
		}

		/*
		static void printpq() {
				std::cerr << "Queue: ";
				for(std::priority_queue<UpdateRequest>::iterator ii = pq.begin(), ei = pq.end(); ii != ei; ++ii){
						std::cerr << graph.getData(ii->node, galois::MethodFlag::UNPROTECTED).id << "(" <<  ii->prio << ") ";
				}
				std::cerr << "\n";
		}
		*/

		static void go(GNode source) {
			using namespace galois::worklists;
			typedef PerSocketChunkLIFO<256> PSchunk;
			typedef ChunkLIFO<256> Chunk;
			typedef OrderedByIntegerMetric<UpdateRequestIndexer,PSchunk> OBIM;

			graph.getData(source).status = PREACTIVE;
			//graph.getData(source).status = galois::GAtomic<unsigned int>(PREACTIVE);

			//std::cout << "max dist is " << maxDist << "\n";
			galois::for_each(UpdateRequest(source, graph.getData(source).prio), sloanFn(), galois::loopname("Sloan"), galois::wl<OBIM>());
			//std::cout << "max dist is " << maxDist << "\n";
		}
	};

	static void go(GNode source, GNode terminal) {

#ifdef FINE_GRAIN_TIMING
		galois::TimeAccumulator vTmain[6]; 
		vTmain[0] = galois::TimeAccumulator();
		vTmain[1] = galois::TimeAccumulator();
		vTmain[2] = galois::TimeAccumulator();
		vTmain[3] = galois::TimeAccumulator();

		vTmain[0].start();
#endif

		bfsFn::go(terminal);
    //verify(terminal);
#ifdef FINE_GRAIN_TIMING
		vTmain[0].stop();
		vTmain[1].start();
#endif
		initFn::go();
#ifdef FINE_GRAIN_TIMING
		vTmain[1].stop();
		vTmain[2].start();
#endif
		sloanFn::go(source);
#ifdef FINE_GRAIN_TIMING
		vTmain[2].stop();
#endif

#ifdef FINE_GRAIN_TIMING
		std::cerr << "bfsFn: " << vTmain[0].get() << "\n";
		std::cerr << "initFn: " << vTmain[1].get() << "\n";
		std::cerr << "sloanFn: " << vTmain[2].get() << "\n";
		//std::cout << "& " << vTmain[0].get() << " & \\multicolumn{2} {c|} {" << vTmain[1].get() << "} & " << vTmain[2].get() << " & " << vTmain[0].get() + vTmain[1].get()  + vTmain[2].get() << "\n";
#endif

		//printSloan();
	}
};

template<typename AlgoTy>
void run(const AlgoTy& algo) {
  GNode source, terminal;

	int maxThreads = numThreads; 
	galois::TimeAccumulator vT[maxThreads+2]; 

	//Measure time to read graph
	vT[INIT] = galois::TimeAccumulator();
	vT[INIT].start();

  readGraph(source, terminal);

	bandwidth("Initial");
	profile("Initial");

  //std::cout << "original bandwidth: " << boost::bandwidth(*bgraph) << std::endl;
  //std::cout << "original profile: " << boost::profile(*bgraph) << std::endl;
  //std::cout << "original max_wavefront: " << boost::max_wavefront(*bgraph) << std::endl;
  //std::cout << "original aver_wavefront: " << boost::aver_wavefront(*bgraph) << std::endl;
  //std::cout << "original rms_wavefront: " << boost::rms_wavefront(*bgraph) << std::endl;

	vT[INIT].stop();

	std::cout << "Init: " << vT[INIT].get() << " ( " << (double) vT[INIT].get() / 1000 << " seconds )\n";

	//Measure total computation time to read graph
	vT[TOTAL].start();

	//galois::setActiveThreads(1);

	// Execution with the specified number of threads
	vT[RUN] = galois::TimeAccumulator();

	std::cout << "Running " << algo.name() << " version with " << numThreads << " threads for " << niter << " iterations\n";

	// I've observed cold start. First run takes a few millis more. 
	//algo.go(source, terminal);

	for(int i = 0; i < niter; i++){
		vT[RUN].start();

		//algo.go(graph.getData(source).id);
		algo.go(source, terminal);

		vT[RUN].stop();

		permute();
		bandwidth("Permuted");
		profile("Permuted");

		std::cout << "Iteration " << i << " numthreads: " << numThreads << " " << vT[RUN].get() << "\n";

		if(i != niter-1)
			resetGraph();
	}

	std::cout << "Final time numthreads: " << numThreads << " " << vT[RUN].get() << "\n";
	std::cout << "Avg time numthreads: " << numThreads << " " << vT[RUN].get() / niter << "\n";

#ifdef PRINT_DEGREE_DISTR
	printDegreeDistribution();
#endif

	vT[TOTAL].stop();

	std::cout << "Total with threads: " << numThreads << " " << vT[TOTAL].get() << " ( " << (double) vT[TOTAL].get() / 1000 << " seconds )\n";

  if (!skipVerify) {
    if (verify(source)) {
      std::cout << "Verification successful.\n";
    } else {
      std::cerr << "Verification failed.\n";
      assert(0 && "Verification failed");
      abort();
    }
  }
}

int main(int argc, char **argv) {
  //galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  using namespace galois::worklists;
  typedef BulkSynchronous<PerSocketChunkLIFO<256> > BSWL;

  //#ifdef GALOIS_USE_EXP
  //  typedef BulkSynchronousInline<> BSInline;
  //#else
  typedef BSWL BSInline;
  //#endif

  switch (algo) {
		case serialSloan: run(SerialSloan()); break;
		//case barrierSloan: run(BarrierRegular()); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }

  return 0;
}
