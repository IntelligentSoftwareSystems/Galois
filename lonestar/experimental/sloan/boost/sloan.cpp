/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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
#include "galois/Bag.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "Lonestar/BoilerPlate.h"

// kik
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

#define W1 1 // default weight for the distance in the Sloan algorithm
#define W2 2 // default weight for the degree in the Sloan algorithm

static const char* name = "Sloan's reordering algorithm";
static const char* desc =
    "Computes a permutation of a matrix according to Sloan's algorithm";
static const char* url = 0;

//****** Command Line Options ******
enum BFSAlgo {
  serialSloan,
  // barrierSloan,
};

enum ExecPhase {
  INIT,
  RUN,
  TOTAL,
};

static const unsigned int DIST_INFINITY =
    std::numeric_limits<unsigned int>::max() - 1;

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
                                        cll::desc("Node to start search from"),
                                        cll::init(DIST_INFINITY));
static cll::opt<unsigned int>
    reportNode("reportnode", cll::desc("Node to report distance to"),
               cll::init(1));
static cll::opt<bool> scaling(
    "scaling",
    llvm::cl::desc(
        "Scale to the number of threads with a given step starting from"),
    llvm::cl::init(false));
static cll::opt<unsigned int> scalingStep("step", cll::desc("Scaling step"),
                                          cll::init(2));
static cll::opt<unsigned int>
    niter("iter", cll::desc("Number of benchmarking iterations"), cll::init(5));
static cll::opt<BFSAlgo> algo(
    cll::desc("Choose an algorithm:"),
    cll::values(clEnumVal(serialSloan, "Boost Serial Sloan"), clEnumValEnd),
    cll::init(serialSloan));
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int dist;
  unsigned int id;
  // unsigned int sum;
  // bool flag;
  // std::vector<galois::graphs::LC_CSR_Graph<SNode, void>::GraphNode> bucket;
};

struct Prefix {
  unsigned int id;
  unsigned int val;
  Prefix(unsigned int _id, unsigned _val) : id(_id), val(_val) {}
};

// typedef galois::graphs::LC_Linear_Graph<SNode, void> Graph;
typedef galois::graphs::LC_CSR_Graph<SNode, void>::_with_no_lockable<
    true>::_with_numa_alloc<true>
    Graph;
// typedef galois::graphs::LC_CSRInline_Graph<SNode, char> Graph;
// typedef galois::graphs::MorphGraph<SNode, void, false> Graph;
typedef Graph::GraphNode GNode;

typedef boost::adjacency_list<
    boost::setS, boost::vecS, boost::undirectedS,
    boost::property<
        boost::vertex_color_t, boost::default_color_type,
        boost::property<boost::vertex_degree_t, int,
                        boost::property<boost::vertex_priority_t, signed int>>>>
    BGraph;

typedef boost::graph_traits<BGraph>::vertex_descriptor Vertex;
typedef boost::graph_traits<BGraph>::vertices_size_type size_type;

Graph graph;
BGraph* bgraph = NULL;

struct PSum {
  unsigned int sum;
  PSum(unsigned int _s) : sum(_s) {}
};

static size_t degree(const GNode& node) {
  return std::distance(graph.edge_begin(node), graph.edge_end(node));
}

struct UpdateRequest {
  GNode n;
  unsigned int w;

  UpdateRequest() : w(0) {}
  UpdateRequest(const GNode& N, unsigned int W) : n(N), w(W) {}
  bool operator<(const UpdateRequest& o) const { return w < o.w; }
  bool operator>(const UpdateRequest& o) const { return w > o.w; }
  unsigned getID() const { return /* graph.getData(n).id; */ 0; }
};

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out << "(dist: " << n.dist << ")";
  return out;
}

struct UpdateRequestIndexer {
  unsigned int operator()(const UpdateRequest& val) const {
    unsigned int t = val.w;
    return t;
  }
};

struct GNodeIndexer {
  unsigned int operator()(const GNode& val) const {
    return graph.getData(val, galois::MethodFlag::UNPROTECTED).dist;
  }
};

struct GNodeLess {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, galois::MethodFlag::UNPROTECTED).dist <
           graph.getData(b, galois::MethodFlag::UNPROTECTED).dist;
  }
};

struct GNodeGreater {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, galois::MethodFlag::UNPROTECTED).dist >
           graph.getData(b, galois::MethodFlag::UNPROTECTED).dist;
  }
};

struct GNodeBefore {
  bool operator()(const GNode& a, const GNode& b) const {
    return (degree(a) < degree(b));
  }
};

std::vector<GNode> perm;
// std::vector<GNode> tperm;
// std::vector< std::vector<GNode> > bucket;
// debug
galois::GAtomic<unsigned int> loops     = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> sorts     = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> maxbucket = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> minbucket =
    galois::GAtomic<unsigned int>(DIST_INFINITY);
galois::GAtomic<unsigned int> avgbucket   = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> numbucket   = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> smallbucket = galois::GAtomic<unsigned int>(0);

static void printSloan() {
  std::cerr << "Sloan Permutation:\n";

  for (std::vector<GNode>::iterator nit = perm.begin(); nit != perm.end();
       nit++) {
    SNode& data = graph.getData(*nit);
    // std::cerr << "[" << data.id << "] level: " << data.dist << " degree: " <<
    // data.degree << "\n"; std::cerr << data.id + 1 << " (" << data.degree << ")
    // level: " << data.dist << "\n";
    std::cerr << data.id + 1 << "\n";
  }
  std::cerr << "\n";
}

static void permute() {

  // std::vector<GNode> perm;
  // perm.reserve(graph.size());;

  std::vector<GNode> nodemap;
  nodemap.reserve(graph.size());
  ;

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {

    nodemap[graph.getData(*src).id] = *src;
  }

  unsigned int N = perm.size() - 1;

  for (int i = N; i >= 0; --i) {
    // std::cerr << perm[i] << " " << graph.getData(nodemap[permid[i]]).id << "
    // changes to: " << N - i << "\n";
    graph.getData(perm[i]).id = N - i;
  }
}

// debugging
static void printAccess(std::string msg) {
  std::cerr << msg << " Access Pattern:\n";

  std::vector<unsigned int> temp;

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {

    SNode& sdata = graph.getData(*src);

    std::cerr << sdata.id << " connected with (" << degree(*src) << "): ";

    for (Graph::edge_iterator
             ii = graph.edge_begin(*src, galois::MethodFlag::UNPROTECTED),
             ei = graph.edge_end(*src, galois::MethodFlag::UNPROTECTED);
         ii != ei; ++ii) {
      GNode dst    = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      unsigned int diff = abs(sdata.id - ddata.id);

      std::cerr << ddata.id << " (" << diff << "), ";
    }

    std::cerr << "\n";
    // std::cerr << data.id << " (" << degree(*src) << ") level: " << data.dist
    // << " reads: " << data.read << " writes: " << data.write << "\n"; std::cerr
    // << data.id << " (" << degree(*src) << ") level: " << data.dist << "\n";

    temp.push_back(sdata.id);
  }

  // for(std::vector<unsigned int>::reverse_iterator nit = temp.rbegin(); nit !=
  // temp.rend(); nit++)
  for (std::vector<unsigned int>::iterator nit = temp.begin();
       nit != temp.end(); nit++) {
    std::cerr << *nit + 1 << "\n";
  }
  std::cerr << "\n";
}

static void findStartingNode(GNode& starting) {
  unsigned int mindegree = DIST_INFINITY;

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {
    unsigned int nodedegree = degree(*src);

    if (nodedegree < mindegree) {
      mindegree = nodedegree;
      starting  = *src;
    }
  }

  SNode& data = graph.getData(starting);
  std::cerr << "Starting Node: " << data.id << " degree: " << degree(starting)
            << "\n";
}

// Compute variance around mean distance from the source
static void variance(unsigned long int mean) {
  unsigned long int n = 0;
  long double M2      = 0.0;
  long double var     = 0.0;

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {
    SNode& data = graph.getData(*src);
    if (data.dist < DIST_INFINITY) {
      M2 += (data.dist - mean) * (data.dist - mean);
      ++n;
    }
  }

  var = M2 / (n - 1);
  std::cout << "var: " << var << " mean: " << mean << "\n";
}

struct not_consistent {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n);
         ii != ei; ++ii) {
      GNode dst          = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(dst).dist;
      if (ddist > dist + 1) {
        std::cerr << "bad level value for " << graph.getData(dst).id << ": "
                  << ddist << " > " << (dist + 1) << "\n";
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
      std::cerr << "unvisited node " << graph.getData(n).id << ": " << dist
                << " >= INFINITY\n";
      return true;
    }
    // std::cerr << "visited node " << graph.getData(n).id << ": " << dist <<
    // "\n";
    return false;
  }
};

//! Simple verifier
static bool verify(GNode& source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }

  // size_t id = 0;

#ifdef GALOIS_JUNE
  bool okay =
      galois::find_if(graph.begin(), graph.end(), not_consistent()) ==
          graph.end() &&
      galois::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();
#else
  bool okay = galois::ParallelSTL::find_if(graph.begin(), graph.end(),
                                           not_consistent()) == graph.end() &&
              galois::ParallelSTL::find_if(graph.begin(), graph.end(),
                                           not_visited()) == graph.end();
#endif

  // if (okay) {
  galois::GReduceMax<unsigned int> maxDist;
  galois::GAccumulator<unsigned> sum;
  galois::GAccumulator<unsigned> count;

  galois::do_all(galois::iterate(graph), [&](const GNode& n) {
    auto d = graph.getData(n, galois::MethodFlag::UNPROTECTED).dist;
    if (d < INFINITY) {
      maxDist.update(d);
      sum += d;
      count += 1;
    }
  });
  std::cout << "max dist: " << maxDist.reduce() << "\n";
  unsigned mean = sum.reduce() / count.reduce();
  std::cout << "avg dist: " << mean << "\n";

  variance(mean.reduce());
  //}

  return okay;
}

// Clear node data to re-execute on specific graph
struct resetNode {
  void operator()(const GNode& n) const {
    graph.getData(n).dist = DIST_INFINITY;
    // graph.getData(n).flag = false;
    // graph.getData(n).bucket.clear();
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
    // std::cerr << graph.getData(*n, galois::MethodFlag::UNPROTECTED).id << "	"
    // << graph.getData(*n, galois::MethodFlag::UNPROTECTED).dist << "\n";
  }

  std::cerr << "Degree	Count\n";
  for (std::map<unsigned int, unsigned int>::iterator slot = distr.begin(),
                                                      ei   = distr.end();
       slot != ei; ++slot) {
    std::cerr << slot->first << "	" << slot->second << "\n";
  }
}

// Read graph from a binary .gr as dirived from a Matrix Market .mtx using
// graph-convert
static void readGraph(GNode& source, GNode& report) {
  galois::graphs::readGraph(graph, filename);
  source = *graph.begin();

  size_t nnodes = graph.size();
  std::cout << "Read " << nnodes << " nodes\n";

  size_t id = 0;

  bgraph = new BGraph(nnodes);

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {
    SNode& node = graph.getData(*src, galois::MethodFlag::UNPROTECTED);
    node.id     = id++;
  }

  std::cout << "Read binary graph\n";

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {
    SNode& dsrc = graph.getData(*src, galois::MethodFlag::UNPROTECTED);
    for (Graph::edge_iterator
             ii = graph.edge_begin(*src, galois::MethodFlag::UNPROTECTED),
             ei = graph.edge_end(*src, galois::MethodFlag::UNPROTECTED);
         ii != ei; ++ii) {
      GNode dst   = graph.getEdgeDst(ii);
      SNode& ddst = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      boost::add_edge(dsrc.id, ddst.id, *bgraph);
    }
  }
}

//! Serial BFS using Galois graph
struct BoostSloan {
  std::string name() const { return "Boost Serial Sloan"; }

  static void sloan_ordering(
      BGraph& g, Vertex s, Vertex e, std::vector<Vertex>::iterator permutation,
      boost::property_map<BGraph, boost::vertex_color_t>::type color,
      boost::property_map<BGraph, boost::vertex_degree_t>::type degree,
      boost::property_map<BGraph, boost::vertex_priority_t>::type priority) {
    // typedef typename boost::property_traits<boost::property_map<BGraph,
    // boost::vertex_degree_t>::type >::value_type Degree;
    typedef boost::property_traits<
        boost::property_map<BGraph, boost::vertex_priority_t>::type>::value_type
        Degree;
    typedef boost::property_traits<
        boost::property_map<BGraph, boost::vertex_color_t>::type>::value_type
        ColorValue;
    // typedef typename boost::property_traits<boost::property<
    // boost::vertex_color_t, boost::default_color_type> >::value_type
    // ColorValue;
    typedef boost::color_traits<ColorValue> Color;
    typedef boost::graph_traits<BGraph>::vertex_descriptor Vertex;
    typedef std::vector<size_type>::iterator vec_iter;

    typedef boost::property_map<BGraph, boost::vertex_index_t>::const_type
        VertexID;

    std::vector<Vertex>::iterator skato = permutation;

#ifdef FINE_GRAIN_TIMING
    galois::TimeAccumulator vTmain[5];
    vTmain[0] = galois::TimeAccumulator();
    vTmain[1] = galois::TimeAccumulator();
    vTmain[2] = galois::TimeAccumulator();
    vTmain[3] = galois::TimeAccumulator();
    vTmain[4] = galois::TimeAccumulator();

    vTmain[0].start();
#endif

    // Creating a std-vector for storing the distance from the end vertex in it
    std::vector<size_type> dist(num_vertices(g), 0);

    // Wrap a property_map_iterator around the std::iterator
    boost::iterator_property_map<vec_iter, VertexID, size_type, size_type&>
        dist_pmap(dist.begin(), get(boost::vertex_index, g));

    boost::breadth_first_search(
        g, e,
        boost::visitor(boost::make_bfs_visitor(
            boost::record_distances(dist_pmap, boost::on_tree_edge()))));

    // Creating a property_map for the indices of a vertex
    boost::property_map<BGraph, boost::vertex_index_t>::type index_map =
        get(boost::vertex_index, g);

    // Sets the color and priority to their initial status
    unsigned cdeg;
    boost::graph_traits<BGraph>::vertex_iterator ui, ui_end;
    for (boost::tie(ui, ui_end) = boost::vertices(g); ui != ui_end; ++ui) {
      put(color, *ui, Color::white());
      cdeg = get(degree, *ui) + 1;
      put(priority, *ui, W1 * dist[index_map[*ui]] - W2 * cdeg);
      // int skata = W1*dist[index_map[*ui]]-W2*cdeg;
      // std::cerr << "[" << index_map[*ui] << "]: " << get(priority, *ui) << "
      // cdge: " << cdeg << " dist: " << dist[index_map[*ui]] << " prio: " <<
      // W1*dist[index_map[*ui]]+W2*cdeg<< " real: " <<
      // W1*dist[index_map[*ui]]-W2*cdeg << " skata: " << skata << "\n";
    }

    // Priority list
    typedef boost::indirect_cmp<
        boost::property_map<BGraph, boost::vertex_priority_t>::type,
        std::greater<Degree>>
        Compare;

    Compare comp(priority);
    std::list<Vertex> priority_list;

    // Some more declarations
    boost::graph_traits<BGraph>::out_edge_iterator ei, ei_end, ei2, ei2_end;
    Vertex u, v, w;

    put(color, s,
        Color::green()); // Sets the color of the starting vertex to gray
    priority_list.push_front(s); // Puts s into the priority_list

#ifdef FINE_GRAIN_TIMING
    vTmain[0].stop();
    vTmain[4].start();
#endif

    while (!priority_list.empty()) {
#ifdef FINE_GRAIN_TIMING
      vTmain[3].start();
#endif

      // std::cerr << "Sorting " << priority_list.size() << "\n";
      priority_list.sort(comp); // Orders the elements in the priority list in
                                // an ascending manner

      /*
      std::cerr << "Queue: ";
      for(std::list<Vertex>::iterator ii = priority_list.begin(), ee =
      priority_list.end(); ii != ee; ++ii){ std::cerr << index_map[*ii] << "("
      << get(priority, *ii) << ") ";
      }
      std::cerr << "\n";
      */

#ifdef FINE_GRAIN_TIMING
      vTmain[3].stop();
      vTmain[1].start();
#endif

      u = priority_list
              .front(); // Accesses the last element in the priority list
      priority_list.pop_front(); // Removes the last element in the priority
                                 // list

      if (get(color, u) == Color::green()) {
        // for-loop over all out-edges of vertex u
        for (boost::tie(ei, ei_end) = boost::out_edges(u, g); ei != ei_end;
             ++ei) {
          v = boost::target(*ei, g);

          put(priority, v, get(priority, v) + W2); // updates the priority

          if (get(color, v) == Color::white()) // test if the vertex is inactive
          {
            put(color, v,
                Color::green()); // giving the vertex a preactive status
            priority_list.push_front(
                v); // writing the vertex in the priority_queue
          }
        }
      }

      // Here starts step 8
      *permutation++ =
          u; // Puts u to the first position in the permutation-vector
      put(color, u, Color::black()); // Gives u an inactive status

      /*
      std::cerr << "Permutation: ";
      for(std::vector<Vertex>::iterator ii = skato, ee = permutation; ii != ee;
      ++ii){
              //std::cerr << index_map[*ii] << " ";
              std::cerr << index_map[*ii] << "(" << get(priority, *ii) << ") ";
      }
      std::cerr << "\n";
      */

#ifdef FINE_GRAIN_TIMING
      vTmain[1].stop();
      vTmain[2].start();
#endif

      // for loop over all the adjacent vertices of u
      for (boost::tie(ei, ei_end) = out_edges(u, g); ei != ei_end; ++ei) {

        v = target(*ei, g);

        if (get(color, v) == Color::green()) { // tests if the vertex is
                                               // inactive

          put(color, v, Color::red()); // giving the vertex an active status
          put(priority, v, get(priority, v) + W2); // updates the priority

          // for loop over alll adjacent vertices of v
          for (boost::tie(ei2, ei2_end) = out_edges(v, g); ei2 != ei2_end;
               ++ei2) {
            w = target(*ei2, g);

            if (get(color, w) !=
                Color::black()) { // tests if vertex is postactive

              put(priority, w, get(priority, w) + W2); // updates the priority

              if (get(color, w) == Color::white()) {

                put(color, w,
                    Color::green()); // gives the vertex a preactive status
                priority_list.push_front(
                    w); // puts the vertex into the priority queue

              } // end if

            } // end if

          } // end for

        } // end if

      } // end for
#ifdef FINE_GRAIN_TIMING
      vTmain[2].stop();
#endif

    } // end while
#ifdef FINE_GRAIN_TIMING
    vTmain[4].stop();
#endif

#ifdef FINE_GRAIN_TIMING
    std::cout << "Init: " << vTmain[0].get() << "\n";
    std::cout << "First phase: " << vTmain[1].get() << "\n";
    std::cout << "Second phase for loop: " << vTmain[2].get() << "\n";
    std::cout << "Sort: " << vTmain[3].get() << "\n";
    std::cout << "Total comp: " << vTmain[4].get() << "\n";
#endif

    // return permutation;
  }

  void operator()(const unsigned int& source) const {
    boost::graph_traits<BGraph>::vertex_iterator ui, ui_end;

    boost::property_map<BGraph, boost::vertex_degree_t>::type deg =
        get(boost::vertex_degree, *bgraph);
    for (boost::tie(ui, ui_end) = boost::vertices(*bgraph); ui != ui_end; ++ui)
      deg[*ui] = boost::degree(*ui, *bgraph);

    boost::property_map<BGraph, boost::vertex_index_t>::type index_map =
        boost::get(boost::vertex_index, *bgraph);

    // Creating a vector of vertices
    std::vector<Vertex> sloan_order(num_vertices(*bgraph));
    // Creating a vector of size_type
    std::vector<size_type> perm(num_vertices(*bgraph));

    Vertex s = boost::vertex(source, *bgraph);
    int ecc; // defining a variable for the pseudoperipheral radius

    // Calculating the pseudoeperipheral node and radius
    Vertex e = boost::pseudo_peripheral_pair(
        *bgraph, s, ecc, get(boost::vertex_color, *bgraph),
        get(boost::vertex_degree, *bgraph));

    std::cout << std::endl;
    std::cout << "Starting vertex: " << s << std::endl;
    std::cout << "Pseudoperipheral vertex: " << e << std::endl;
    std::cout << "Pseudoperipheral radius: " << ecc << std::endl << std::endl;

    // Sloan ordering
    // boost::sloan_ordering(*bgraph, s, e, sloan_order.begin(),
    // get(boost::vertex_color, *bgraph), get(boost::vertex_degree, *bgraph),
    // get(boost::vertex_priority, *bgraph));
    sloan_ordering(*bgraph, s, e, sloan_order.begin(),
                   get(boost::vertex_color, *bgraph),
                   get(boost::vertex_degree, *bgraph),
                   get(boost::vertex_priority, *bgraph));

    /*
    std::cout << "Sloan ordering starting at: " << s << std::endl;

    for (std::vector<Vertex>::const_iterator i = sloan_order.begin(); i !=
    sloan_order.end(); ++i) std::cout << index_map[*i] << "\n"; std::cout <<
    std::endl;
    */

    for (size_type c = 0; c != sloan_order.size(); ++c)
      perm[index_map[sloan_order[c]]] = c;

    std::cout << "bandwidth: "
              << bandwidth(*bgraph, make_iterator_property_map(
                                        &perm[0], index_map, perm[0]))
              << std::endl;
    std::cout << "profile: "
              << profile(*bgraph, make_iterator_property_map(
                                      &perm[0], index_map, perm[0]))
              << std::endl;
    /*
std::cout << "max_wavefront: "
    << max_wavefront(*bgraph, make_iterator_property_map(&perm[0], index_map,
perm[0]))
    << std::endl;
std::cout << "aver_wavefront: "
    << aver_wavefront(*bgraph, make_iterator_property_map(&perm[0], index_map,
perm[0]))
    << std::endl;
std::cout << "rms_wavefront: "
    << rms_wavefront(*bgraph, make_iterator_property_map(&perm[0], index_map,
perm[0]))
    << std::endl;
    */
  }
};

template <typename AlgoTy>
void run(const AlgoTy& algo) {
  GNode source, report;

  int maxThreads = numThreads;
  galois::TimeAccumulator vT[maxThreads + 2];

  // Measure time to read graph
  vT[INIT] = galois::TimeAccumulator();
  vT[INIT].start();

  readGraph(source, report);

  std::cout << "original bandwidth: " << boost::bandwidth(*bgraph) << std::endl;
  std::cout << "original profile: " << boost::profile(*bgraph) << std::endl;
  // std::cout << "original max_wavefront: " << boost::max_wavefront(*bgraph) <<
  // std::endl; std::cout << "original aver_wavefront: " <<
  // boost::aver_wavefront(*bgraph) << std::endl; std::cout << "original
  // rms_wavefront: " << boost::rms_wavefront(*bgraph) << std::endl;

  vT[INIT].stop();

  std::cout << "Init: " << vT[INIT].get() << " ( "
            << (double)vT[INIT].get() / 1000 << " seconds )\n";

  // Measure total computation time to read graph
  vT[TOTAL].start();

  // galois::setActiveThreads(1);

  // Execution with the specified number of threads
  vT[RUN] = galois::TimeAccumulator();

  std::cout << "Running " << algo.name() << " version with " << numThreads
            << " threads for " << niter << " iterations\n";

  // I've observed cold start. First run takes a few millis more.
  // algo(source);

  for (int i = 0; i < niter; i++) {
    vT[RUN].start();

    algo(graph.getData(source).id);

    vT[RUN].stop();

    std::cout << "Iteration " << i << " numthreads: " << numThreads << " "
              << vT[RUN].get() << "\n";
  }

  std::cout << "Final time numthreads: " << numThreads << " " << vT[RUN].get()
            << "\n";
  std::cout << "Avg time numthreads: " << numThreads << " "
            << vT[RUN].get() / niter << "\n";

#ifdef PRINT_DEGREE_DISTR
  printDegreeDistribution();
#endif

  vT[TOTAL].stop();

  std::cout << "Total with threads: " << numThreads << " " << vT[TOTAL].get()
            << " ( " << (double)vT[TOTAL].get() / 1000 << " seconds )\n";

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

int main(int argc, char** argv) {
  // galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  using namespace galois::worklists;
  typedef BulkSynchronous<PerSocketChunkLIFO<256>> BSWL;

  //#ifdef GALOIS_USE_EXP
  //  typedef BulkSynchronousInline<> BSInline;
  //#else
  typedef BSWL BSInline;
  //#endif

  switch (algo) {
  case serialSloan:
    run(BoostSloan());
    break;
    // case barrierSloan: run(BarrierRegular()); break;
  default:
    std::cerr << "Unknown algorithm" << algo << "\n";
    abort();
  }

  return 0;
}
