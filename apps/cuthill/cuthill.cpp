/** Cuthull-McCee -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/CheckedObject.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/Graph2.h"
#ifdef GALOIS_EXP
#include "Galois/PriorityScheduling.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "Galois/Atomic.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <deque>
#include <queue>
#include <numeric>

static const char* name = "Cuthill Mcee";
static const char* desc = "";
static const char* url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int dist;
  unsigned int degree;
  bool done;
};

typedef Galois::Graph::LC_Linear_Graph<SNode, void> Graph;
//typedef Galois::Graph::FirstGraph<SNode, double, false> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

struct GNodeIndexer {
  unsigned int operator()(const GNode& val) const {
    return graph.getData(val, Galois::NONE).dist;// >> 2;
  }
};

unsigned int max_dist;
std::deque<unsigned int> level_count;
std::deque<unsigned int> read_offset;
std::deque<GaloisRuntime::LL::CacheLineStorage<unsigned int> > write_offset;
//std::deque<unsigned int> write_offset;
std::vector<GNode> perm;

struct CutHillUnordered {
  std::string name() const { return "Cuthill unordered"; }

  struct bfsFn {
    typedef int tt_does_not_need_aborts;

    void operator()(GNode& n, Galois::UserContext<GNode>& ctx) const {
      SNode& data = graph.getData(n, Galois::NONE);
      unsigned int old_max;
      while ((old_max = max_dist) < data.dist)
	__sync_bool_compare_and_swap(&max_dist, old_max, data.dist);

      unsigned int newDist = data.dist + 1;
      
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
	     ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
	GNode dst = graph.getEdgeDst(ii);
	SNode& ddata = graph.getData(dst, Galois::NONE);
	
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

  struct bfsFnGalois {
    void operator()(GNode& n, Galois::UserContext<GNode>& ctx) const {
      SNode& data = graph.getData(n);
      unsigned int old_max;
      while ((old_max = max_dist) < data.dist)
	__sync_bool_compare_and_swap(&max_dist, old_max, data.dist);

      unsigned int newDist = data.dist + 1;
      
      for (Graph::edge_iterator ii = graph.edge_begin(n),
	     ei = graph.edge_end(n); ii != ei; ++ii) {
	GNode dst = graph.getEdgeDst(ii);
	SNode& ddata = graph.getData(dst);
	if (ddata.dist > newDist) {
	  ddata.dist = newDist;
	  ctx.push(dst);
	}
      }
    }
  };

    static void go(GNode source) {
      using namespace GaloisRuntime::WorkList;
      typedef dChunkedFIFO<64> dChunk;
      typedef ChunkedFIFO<64> Chunk;
      typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;
      
      graph.getData(source).dist = 0;
      Galois::for_each<OBIM>(source, bfsFn(), "BFS");
      std::cout << "max dist is " << max_dist << "\n";
    }
  };

  struct default_reduce {
    template<typename T>
    void operator()(T& dest, T& src) {
      T::reduce(dest,src);
    }
  };
  
  struct count_levels {
    std::deque<unsigned> counts;
    void operator()(GNode& n) {
      //Delete the Galois::NONE to use conflict detection
      SNode& data = graph.getData(n, Galois::NONE);
      if (counts.size() <= data.dist)
	counts.resize(data.dist + 1);
      ++counts[data.dist];
    }
    static void reduce(count_levels& dest, count_levels& src) {
      if (dest.counts.size() < src.counts.size())
	dest.counts.resize(src.counts.size());
      std::transform(src.counts.begin(), src.counts.end(), dest.counts.begin(), dest.counts.begin(), std::plus<unsigned>());
    }

    static void go(GNode source) {
      level_count = GaloisRuntime::do_all_impl(graph.begin(), graph.end(), count_levels(), default_reduce(), true).counts;
      read_offset.push_back(0);
      std::partial_sum(level_count.begin(), level_count.end(), back_inserter(read_offset));
      //write_offset = read_offset;
      write_offset.resize(read_offset.size());
      for(int x = 0; x < read_offset.size(); ++x)
	write_offset[x].data = read_offset[x];
    }
  };


  struct UnsignedIndexer {
    unsigned operator()(unsigned x) const { return x;}
  };

  struct placeFn {
    typedef int tt_does_not_need_aborts;

    void operator()(unsigned me, unsigned int tot) const {
      unsigned n = me;
      while (n < max_dist + 1) {
	unsigned start = read_offset[n];
	volatile unsigned* endp = (volatile unsigned*)&write_offset[n].data;
	unsigned todo = level_count[n];
	while (todo) {
	  //spin
	  while (start == *endp) {}
	  GNode next = perm[start];
	  for (Graph::edge_iterator ii = graph.edge_begin(next, Galois::NONE),
		 ei = graph.edge_end(next, Galois::NONE); ii != ei; ++ii) {
	    GNode dst = graph.getEdgeDst(ii);
	    SNode& ddata = graph.getData(dst, Galois::NONE);
	    if (!ddata.done && (ddata.dist == n + 1)) {
	      ddata.done = true;
	      perm[write_offset[ddata.dist].data] = dst;
	      ++write_offset[ddata.dist].data;
	    }
	  }
	  //	++read_offset[n];
	  //	--level_count[n];
	  ++start;
	  --todo;
	}
	n += tot;
      }
    }

    static void go(GNode source) {
      using namespace GaloisRuntime::WorkList;
      typedef dChunkedFIFO<8> dChunk;
      typedef ChunkedFIFO<8> Chunk;
      typedef OrderedByIntegerMetric<UnsignedIndexer,FIFO<> > OBIM;
      perm[0] = source;
      write_offset[0].data = 1;
      Galois::on_each(placeFn(), "place");
      //Galois::for_each<OBIM >(0U, placeFn(), "place");
      //std::cout << "max dist is " << max_dist << "\n";
    }
  };

  /*
  void operator()(GNode& n, Galois::UserContext<GNode>& lwl) const {
    SNode& data = graph.getData(n);
    //std::cout << "visiting " << data.dist << " path " << data.path << "\n";

    for (Graph::edge_iterator ii = graph.edge_begin(n),
	   ei = graph.edge_end(n); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);
      if (ddata.dist > data.dist + 1) {
	//First time at this level, we are trivially the best parent
	ddata.dist = data.dist + 1;
	ddata.parent = &data;
	lwl.push(dst);
      } else if (ddata.dist == data.dist + 1 && 
		 firstIsMin(&data, ddata.parent)) {
	//someone else already got here, but we are the minpath
	ddata.parent = &data;
	lwl.push(dst);
      }
    }
  }
  static void go(const GNode& source) {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;
    
    SNode& s = graph.getData(source);
    s.dist = 0;
    Galois::for_each<OBIM>(source, CutHillUnordered());
  }
  */

  static void go(GNode source) {
    bfsFn::go(source);
    count_levels::go(source);
    placeFn::go(source);
  }
};

void initNode(const GNode& n) {
  SNode& data = graph.getData(n, Galois::NONE);
  data.dist = DIST_INFINITY;
  data.done = false;
  data.degree = std::distance(graph.edge_begin(n, Galois::NONE), graph.edge_end(n,Galois::NONE));
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  Galois::StatTimer itimer("Init", "");
  itimer.start();
  LonestarStart(argc, argv,name, desc, url);
  
  graph.structureFromFile(filename);
  Galois::do_all(graph.begin(), graph.end(), initNode);
  itimer.stop();
  std::cout << "read " << std::distance(graph.begin(), graph.end()) << " nodes\n";
  perm.resize(std::distance(graph.begin(), graph.end()));

  {
  Galois::StatTimer T;
  T.start();
  CutHillUnordered::go(*graph.begin());
  T.stop();
  }
  std::cout << "done!\n";
  return 0;
}
