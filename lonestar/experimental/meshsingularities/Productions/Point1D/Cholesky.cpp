/** Elimination game -*- C++ -*-
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
 * @section Description
 *
 * compute Cholesky factorization of a graph.
 *
 * @author Noah Anderson <noah@ices.utexas.edu>
 */

// A bunch of this is copied from SpanningTree

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/UnionFind.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/Graph/Graph.h" // FirstGraph
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdio>               // For certain debugging output
#include "1DSingularityMatrixGenerator.hxx"

namespace cll = llvm::cl;

const char* name = "Cholesky Factorization";
const char* desc = "Compute the Cholesky factorization of a graph";
const char* url = NULL;

enum Ordering {
  sequential,
  leastdegree,
  pointless
};

static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<graph file>"), cll::Required);
static cll::opt<Ordering> ordering("ordering",
                                   cll::desc("Graph traversal order:"),
    cll::values(
      clEnumVal(sequential, "Sequential ordering"),
      clEnumVal(leastdegree, "Least-degree ordering"),
      clEnumVal(pointless, "1,6,4,5,0,3,7,2;9,... ordering"),
      clEnumValEnd), cll::init(leastdegree));

struct Node {
  unsigned id;
  bool seen;
  Node(): seen(false) { };
};

// WARNING: Will silently behave oddly when given wrong data type
typedef double edgedata;
//typedef float edgedata;

// LC_Linear_Graph cannot have structure modified; not suitable for
// symbolic factorization.
//typedef galois::Graph::LC_Linear_Graph<Node,edgedata>::with_numa_alloc<true>::type Graph;
typedef galois::Graph::FirstGraph<Node,edgedata,true> Graph;
typedef galois::Graph::FirstGraph<Node,edgedata,false> SymbolicGraph;

typedef Graph::GraphNode GNode;
typedef SymbolicGraph::GraphNode SGNode;

// The dependency list is stored as a total ordering
typedef unsigned int DepItem;
DepItem *depgraph;
unsigned int nodecount = 0;

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << &n << "]";
  return os;
}

// Adapted from preflowpush/Preflowpush.cpp
// Find the edge between src and dst. Sets hasEdge if such an edge was found.
template<typename GraphType, typename NodeType>
typename GraphType::edge_iterator
findEdge(GraphType& g, NodeType src, NodeType dst, bool *hasEdge) {
  typename GraphType::edge_iterator
    ii = g.edge_begin(src, galois::MethodFlag::UNPROTECTED),
    ei = g.edge_end(src, galois::MethodFlag::UNPROTECTED);
  *hasEdge = false;
  for (; ii != ei; ++ii) {
    if (g.getEdgeDst(ii) == dst) {
      *hasEdge = true;
      break;
    }
  }
  return ii;
}

// include/Galois/Graphs/Serialize.h
// Output a graph to a file as an edgelist
template<typename GraphType>
bool outputTextEdgeData(const char* ofile, GraphType& G) {
  std::ofstream file(ofile);
  for (typename GraphType::iterator ii = G.begin(),
         ee = G.end(); ii != ee; ++ii) {
    unsigned src = G.getData(*ii).id;
    // FIXME: Version in include/Galois/Graphs/Serialize.h is wrong.
    for (typename GraphType::edge_iterator jj = G.edge_begin(*ii),
           ej = G.edge_end(*ii); jj != ej; ++jj) {
      unsigned dst = G.getData(G.getEdgeDst(jj)).id;
      file << src << ' ' << dst << ' ' << G.getEdgeData(jj) << '\n';
    }
  }
  return true;
}

// Find the unseen node in the graph of least degree
unsigned int ordering_leastdegree(SymbolicGraph &graph, unsigned int i) {
  unsigned int nseen = 0, bestid = 0, bestdegree = graph.size()+1;
  // Iterate over nodes
  for ( SymbolicGraph::iterator ii = graph.begin(), ei = graph.end();
        ii != ei; ++ii ) {
    SGNode node = *ii;
    Node &noded = graph.getData(node);
    if ( noded.seen ) {
      nseen++;
      continue;
    }
    // Measure degree of the node
    unsigned int degree = 0;
    for (SymbolicGraph::edge_iterator
           iis = graph.edge_begin(node, galois::MethodFlag::WRITE),
           eis = graph.edge_end(node, galois::MethodFlag::WRITE);
	 iis != eis; ++iis) {
      // Only include unseen (not yet eliminated) neighbors in the degree
      if ( graph.getData(graph.getEdgeDst(iis)).seen ) continue;
      degree++;
      // Maybe this isn't going to work out; abort if degree is too high
      if ( degree >= bestdegree ) break;
    }
    // Keep track of the current least-degree node
    if ( degree < bestdegree ) {
      bestid = noded.id;
      bestdegree = degree;
      // We can't do better than 0
      if ( degree == 0 ) break;
    }
  }
  assert(nseen == i || bestdegree == 0);
  //std::cout << "Least degree: Found node " << bestid << " with degree " << bestdegree << "\n";
  return bestid;
}

// For the given ordering, return the ID of the next node that should
// be eliminated.
unsigned int ordering_next_node(SymbolicGraph &graph, unsigned int i) {
  const unsigned int pointless_len = 8,
    pointless_data[] = {1, 6, 4, 5, 0, 3, 7, 2}; // For "pointless" ordering
  unsigned int n = graph.size();
  assert(i<n);

  switch (ordering) {
  case sequential:
    return i;
  case leastdegree:
    return ordering_leastdegree(graph, i);
  case pointless:
    for ( unsigned int offset = i % pointless_len, base = i-offset, j = 0;
          j < pointless_len; j++ ) {
      unsigned int pointless_result = base + pointless_data[j];
      if ( pointless_result >= n ) continue;
      if ( offset == 0 ) return pointless_result;
      offset--;
    }
    assert(false && "Pointless overflow");
  default:
    std::cerr << "Unknown ordering: " << ordering << "\n";
    assert(false && "Unknown ordering");
  }
}

/**
 * Perform the symbolic factorization. Modifies the graph structure.
 * Produces as output a (directed) graph to use with NumericAlgo, the
 * numeric factorization.
 */
template<typename GraphType, typename OutGraphType>
struct SymbolicAlgo {
  GraphType &graph;
  OutGraphType &outgraph;
  SymbolicAlgo(GraphType &graph, OutGraphType &outgraph):
    graph(graph), outgraph(outgraph) { };

  template<typename C>
  void operator()(SGNode node, C& ctx) {
    // Update seen flag on node
    Node &noded = graph.getData(node);
    assert(!noded.seen);
    noded.seen = true;

    // Make sure remaining neighbors form a clique
    // It should be safe to add edges between neighbors here.
    for (typename GraphType::edge_iterator
           iis = graph.edge_begin(node, galois::MethodFlag::WRITE),
           eis = graph.edge_end(node, galois::MethodFlag::WRITE);
	 iis != eis; ++iis) {
      SGNode src = graph.getEdgeDst(iis);
      Node &srcd = graph.getData(src);
      if ( srcd.seen ) continue;

      // Enumerate all other neighbors
      for (typename GraphType::edge_iterator
             iid = graph.edge_begin(node, galois::MethodFlag::WRITE),
             eid = graph.edge_end(node, galois::MethodFlag::WRITE);
	   iid != eid; ++iid) {
        SGNode dst = graph.getEdgeDst(iid);
        Node &dstd = graph.getData(dst);
        if ( dstd.seen ) continue;

	// Find the edge that bridges these two neighbors
	bool hasEdge = false;
        typename GraphType::edge_iterator
          bridge = findEdge(graph, src, dst, &hasEdge);
	if ( hasEdge ) continue;

        // The edge doesn't exist, so add an undirected edge between
	// these two nodes
        bridge = graph.addEdge(src, dst, galois::MethodFlag::WRITE);
        edgedata &ed = graph.getEdgeData(bridge, galois::MethodFlag::UNPROTECTED);
        ed = 0;
      }
    }

    // Determine dependencies; build elimination graph from these edges.
    for (typename GraphType::edge_iterator
           iis = graph.edge_begin(node, galois::MethodFlag::WRITE),
           eis = graph.edge_end(node, galois::MethodFlag::WRITE);
	 iis != eis; ++iis) {
      SGNode src = graph.getEdgeDst(iis);
      Node &srcd = graph.getData(src);
      if ( !srcd.seen ) continue;
      // Add a directed edge from src to node (copying weight)
      typename OutGraphType::edge_iterator
        edge = outgraph.addEdge(outnodes[srcd.id],outnodes[noded.id]);
      edgedata &ed = outgraph.getEdgeData(edge);
      ed = graph.getEdgeData(iis);
    }
  }

  std::vector<GNode> outnodes;
  std::vector<SGNode> innodes;

  void operator()() {
    // Initialize the output (directed) graph: create nodes
    unsigned int nodeID = 0, n = graph.size();
    outnodes.resize(n);
    innodes.resize(n);
    for ( typename GraphType::iterator ii = graph.begin(), ei = graph.end();
          ii != ei; ++ii ) {
      innodes[nodeID] = *ii;
      GNode node = outgraph.createNode(graph.getData(*ii));
      outgraph.addNode(node);
      outnodes[nodeID] = node;
      assert(outgraph.getData(node).id == nodeID);
      nodeID++;
    }

    // Eliminate each node in given traversal order.
    // FIXME: parallelize? See paper.
    for (unsigned int i = 0; i < n; i++ ) {
      nodeID = ordering_next_node(graph, i);
      SGNode node = innodes[nodeID];
      void *emptyctx = NULL;
      (*this)(node, emptyctx);
      // Append to execution order
      depgraph[i] = nodeID;
    }
  }
};

/**
 * Comparison function. The symbolic factorization produces a total
 * ordering of the nodes. In conjunction with the neighborhood
 * function, this defines the traversal order for the numeric
 * factorization.
 */
template<typename GraphType>
struct Cmp {
  GraphType &graph;
  Cmp(GraphType &graph): graph(graph) { };

  bool operator()(const GNode& node1, const GNode& node2) const {
    Node &node1d = graph.getData(node1, galois::MethodFlag::UNPROTECTED);
    Node &node2d = graph.getData(node2, galois::MethodFlag::UNPROTECTED);
    int pos1 = -1, pos2 = -1;

    // Check the total ordering to determine if item1 <= item2
    for ( unsigned int i = 0; i < nodecount; i++ ) {
      if ( depgraph[i] == node1d.id ) pos1 = i;
      if ( depgraph[i] == node2d.id ) pos2 = i; // FIXME: make else if
      if ( pos1 >= 0 && pos2 >= 0 ) break;      // FIXME: eliminate
    }
    assert(pos1 >= 0 && pos2 >= 0);
    bool result = pos1 <= pos2;
    /*
    std::cout << "Cmp: " << node1d.id << " <= " << node2d.id << ": " <<
      (result ? "true" : "false") << "\n";
    */
    return result;
  }
};

/**
 * Defining the neighborhood of the operator. The operator touches all
 * of the edges to and between neighbors. Nodes with overlapping
 * neighborhoods won't be executed in parallel.
 */
template<typename GraphType>
struct NhFunc {
  /*
  // Affect for_each_ordered's choice of executor. This has certain issues.
  typedef int tt_has_fixed_neighborhood;
  static_assert(galois::has_fixed_neighborhood<NhFunc>::value, "Oops!");
  */

  GraphType &graph;
  NhFunc(GraphType &graph): graph(graph) { };

  template<typename C>
  void operator()(GNode& node, C& ctx) {
    (*this)(node);
  }
  void operator()(GNode& node) {
    // Touch all neighbors (this seems to be good enough)
    Graph::edge_iterator ii = graph.edge_begin(node, galois::MethodFlag::WRITE);
  }
};

/**
 * Perform the numeric factorization. Assumes the graph is a directed
 * graph as produced by the symbolic factorization.
 */
template<typename GraphType>
struct NumericAlgo {
  /*
  // Affect for_each_ordered's choice of executor. This has certain issues.
  typedef int tt_does_not_need_push;
  static_assert(galois::does_not_need_push<NumericAlgo>::value, "Oops!");
  */

  GraphType &graph;
  NumericAlgo(GraphType &graph): graph(graph) { };

  void operator()(GNode node, galois::UserContext<GNode>& ctx) {
    // Find self-edge for this node, update it
    bool hasEdge = false;
    edgedata& factor = graph.getEdgeData(findEdge(graph, node, node, &hasEdge),
                                         galois::MethodFlag::UNPROTECTED);
    assert(hasEdge);
    assert(factor > 0);
    factor = sqrt(factor);
    assert(factor != 0 && !isnan(factor));

    // Update seen flag on node
    Node &noded = graph.getData(node);
    assert(!noded.seen);
    noded.seen = true;

    //std::cout << "STARTING " << noded.id << " " << factor << "\n";
    //printf("STARTING %4d %10.5f\n", noded.id, factor);

    // Update all edges (except self-edge)
    for (Graph::edge_iterator
           ii = graph.edge_begin(node, galois::MethodFlag::WRITE),
           ei = graph.edge_end(node, galois::MethodFlag::WRITE);
         ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node &dstd = graph.getData(dst);
      if ( !dstd.seen ) {
        edgedata &ed = graph.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
        ed /= factor;
        //printf("N-EDGE %4d %4d %10.5f\n", noded.id, graph.getData(dst).id, ed);
        //std::cout << noded.id << " " << dstd.id << " " << ed << "\n";
      }
    }

    // Update all edges between neighbors (we're operating on the filled graph,
    // so we they form a (directed) clique)
    for (Graph::edge_iterator
           iis = graph.edge_begin(node, galois::MethodFlag::WRITE),
           eis = graph.edge_end(node, galois::MethodFlag::WRITE);
         iis != eis; ++iis) {
      GNode src = graph.getEdgeDst(iis);
      Node &srcd = graph.getData(src);
      if ( srcd.seen ) continue;
      edgedata& eds = graph.getEdgeData(iis, galois::MethodFlag::UNPROTECTED);

      // Enumerate all other neighbors
      for (Graph::edge_iterator
             iid = graph.edge_begin(node, galois::MethodFlag::WRITE),
             eid = graph.edge_end(node, galois::MethodFlag::WRITE);
           iid != eid; ++iid) {
        GNode dst = graph.getEdgeDst(iid);
        Node &dstd = graph.getData(dst);
        if ( dstd.seen ) continue;

        // Find the edge that bridges these two neighbors
        hasEdge = false;
        Graph::edge_iterator bridge = findEdge(graph, src, dst, &hasEdge);
        if ( !hasEdge ) continue;

        // Update the weight of the bridge edge
        edgedata &edd = graph.getEdgeData(iid, galois::MethodFlag::UNPROTECTED),
          &edb = graph.getEdgeData(bridge, galois::MethodFlag::UNPROTECTED);
        edb -= eds*edd;

        //printf("I-EDGE %4d %4d %10.5f\n", srcd.id, dstd.id, edb);
        //std::cout << srcd.id << " " << dstd.id << " " << edb << "\n";
      }
    }
    //std::cout << "OPERATED ON " << noded.id << "\n";
    //sleep(1); // Maybe use this to help debug parallelism
  }

  void operator()() {
    Graph::iterator ii = graph.begin(), ei = graph.end();
    if (ii != ei) { // Ensure there is at least one node in the graph.
      galois::for_each_ordered(ii, ei, Cmp<GraphType>(graph),
                               NhFunc<GraphType>(graph), *this);
      // galois::for_each(ii, ei, *this);
    }
  }
};

// Load a graph into a FirstGraph. Based on makeGraph from Boruvka.
template <typename GraphType>
static void makeGraph(GraphType &graph, const char* input) {
   std::vector<SGNode> nodes;
   //Create local computation graph.
   typedef galois::Graph::LC_CSR_Graph<Node, edgedata> InGraph;
   typedef InGraph::GraphNode InGNode;
   InGraph in_graph;
   //Read graph from file.
   galois::Graph::readGraph(in_graph, input);
   std::cout << "Read " << in_graph.size() << " nodes\n";
   //A node and a int is an element.
   typedef std::pair<InGNode, edgedata> Element;
   //A vector of element is 'Elements'
   typedef std::vector<Element> Elements;
   //A vector of 'Elements' is a 'Map'
   typedef std::vector<Elements> Map;
   //'in_edges' is a vector of vector of pairs of nodes and int.
   Map edges(in_graph.size());
   //
   int numEdges = 0;
   // Extract edges from input graph
   for (InGraph::iterator src = in_graph.begin(), esrc = in_graph.end();
        src != esrc; ++src) {
      for (InGraph::edge_iterator
             dst = in_graph.edge_begin(*src, galois::MethodFlag::UNPROTECTED),
             edst = in_graph.edge_end(*src, galois::MethodFlag::UNPROTECTED);
           dst != edst; ++dst) {
         edgedata w = in_graph.getEdgeData(dst);
         Element e(*src, w);
         edges[in_graph.getEdgeDst(dst)].push_back(e);
         numEdges++;
      }
   }
   //#if BORUVKA_DEBUG
   std::cout<<"Number of edges "<<numEdges<<std::endl;
   //#endif
   // Create nodes in output graph
   nodes.resize(in_graph.size());
   int nodeID = 0;
   for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
      Node n;
      n.id = nodeID;
      assert(!n.seen);
      SGNode node = graph.createNode(n);
      graph.addNode(node);
      nodes[nodeID] = node;
      nodeID++;
   }

   int id = 0;
   numEdges = 0;
   for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
      SGNode src = nodes[id];
      for (Elements::iterator j = i->begin(), ej = i->end(); j != ej; ++j) {
         typename GraphType::edge_iterator
           it = graph.findEdge(src, nodes[j->first], galois::MethodFlag::UNPROTECTED);
         if ( it != graph.edge_end(src, galois::MethodFlag::UNPROTECTED) ) {
           assert(graph.getEdgeData(it) == j->second);
           continue;
         }
         it = graph.addEdge(src, nodes[j->first], galois::MethodFlag::UNPROTECTED);
         graph.getEdgeData(it) = j->second;
         numEdges++;
      }
      id++;
   }
   
   //#if BORUVKA_DEBUG
   std::cout << "Final num edges " << numEdges << std::endl;
   //#endif
}

// FIXME: implement verify, etc. See SpanningTree.


// Load a double[][] matrix into a FirstGraph. 
template <typename GraphType>
static void makeGraph(GraphType &graph, double** input, int matrix_size) {
     
    std::vector<SGNode> nodes;
    nodes.resize(matrix_size); 
    for(int i = 0; i<matrix_size; i++){
		Node n; 
		n.id = i;
		SGNode node = graph.createNode(n);
		graph.addNode(node); 
		nodes[i] = node; 
	} 
	
	
	int numEdges = 0; 
	for(int i = 0; i<matrix_size; i++){
		SGNode src = nodes[i];
		for(int j = i; j<matrix_size; j++){
			if(input[i][j] != 0){
				typename GraphType::edge_iterator it = graph.addEdge(src, nodes[j], galois::MethodFlag::UNPROTECTED);
				graph.getEdgeData(it) = input[i][j];
				numEdges++;
			}
			
		}
	}
   
   //#if BORUVKA_DEBUG
   std::cout << "Final num edges " << numEdges << std::endl;
   //#endif
}

template <typename GraphType>
bool verify(GraphType &graph) {
  outputTextEdgeData("choleskyedges.txt", graph);
  std::cout << "\n\n\nPlease verify by comparing ./choleskyedges.txt against expected contents.\n\n\n\n"; 
  // FIXME: Try multiplying to double-check result
  return true;
  /*
  if (galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad_graph()) == graph.end()) {
    if (galois::ParallelSTL::find_if(mst.begin(), mst.end(), is_bad_mst()) == mst.end()) {
      CheckAcyclic c;
      return c();
    }
  }
  return false;
  */
}

template<typename Algo>
void run(Algo algo, const char *algoname) {
  galois::StatTimer T, U(algoname);
  T.start(); U.start();
  algo();
  T.stop(); U.stop();
}

template <typename GraphType>
int get_solution_from_graph(GraphType &outgraph, double* rhs){
	clock_t t_start = clock();
	int matrix_size = outgraph.size();
	double* mid_strip = new double[matrix_size]();
	double* lower_strip = new double[matrix_size](); 
	double* upper_strip = new double[matrix_size]();
	
  
	int i = 0; 
  
	for (Graph::GraphNode src : outgraph) {

		for(Graph::edge_iterator edge : outgraph.out_edges(src)) {
			Graph::GraphNode dst = outgraph.getEdgeDst(edge);
			
			double edgeData = outgraph.getEdgeData(edge);
			Node &noded = outgraph.getData(dst);
			if(noded.id == i){
				mid_strip[i] = edgeData;
			}
			else{
				upper_strip[i] = edgeData;
				lower_strip[noded.id] = edgeData;
			}

		}
		i++; 
	}
	/*
	for(int i = 0; i< matrix_size; i++){
		for(int j = 0; j<matrix_size; j++){
			printf("%lf ",matrix[i][j]); 
		}
		printf("\n"); 
	}
	*/
	//forward
	
	
	rhs[0] /=mid_strip[0];
	for(int i = 1; i<matrix_size; i++){
		
		rhs[i] -= lower_strip[i]*rhs[i-1];
		rhs[i] /= mid_strip[i]; 
	}
  
	//backward
	
	rhs[matrix_size - 1] /= mid_strip[matrix_size - 1];
	
	for(int i = matrix_size - 2; i>-1; i--){
		
		rhs[i] -= upper_strip[i]*rhs[i+1];
		rhs[i] /= mid_strip[i]; 
	}
	
	/*
	for(int i = 0; i<matrix_size; i++){
		printf("%lf\n",rhs[i]); 
	}
	*/
	
	printf("\nBS and FS time: %.4fs\n", (float)(clock() - t_start)/CLOCKS_PER_SEC);
}

int main(int argc, char** argv) {

  

  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();

  SymbolicGraph graph;
  Graph outgraph;

  // Load input graph. Read to an LC_Graph and then convert to a
  // FirstGraph. (based on makeGraph from Boruvka.)
  
  double** matrix2;
  double* rhs;
  int matrix_size2 = 15000; 
  double l_boundary_condition = -100; 
  double r_boundary_condition = 100; 
  
  get_matrix_and_rhs(matrix_size2, &matrix2, &rhs, l_boundary_condition, r_boundary_condition);
  /*
  for(int i = 0; i<matrix_size2; i++){
	for(int j = 0; j<matrix_size2; j++){
		
		printf("%lf ", matrix2[i][j]);
	} 
	
	printf(" | %lf\n", rhs[i]); 
  }
  */
  
  
  clock_t t_start = clock();
  makeGraph(graph,matrix2,matrix_size2); 
  //makeGraph(graph, inputFilename.c_str());
  nodecount = graph.size();
  std::cout << "Num nodes: " << nodecount << "\n";

  // Verify IDs assigned to each node
  {
    unsigned int i = 0;
    for (SymbolicGraph::iterator ii = graph.begin(), ei = graph.end();
         ii != ei; ++ii) {
      Node& data = graph.getData(*ii);
      assert(data.id == i++);
      assert(!data.seen);
    }
    assert(i == nodecount);
  }

  // Initialize dependency ordering
  depgraph = new DepItem[nodecount];
  assert(depgraph);

  Tinitial.stop();

  //galois::preAlloc(numThreads);
  galois::reportPageAlloc("MeminfoPre");

  // First run the symbolic factorization
  std::cout << "Symbolic factorization\n";
  run(SymbolicAlgo<SymbolicGraph,Graph>(graph, outgraph), "SymbolicTime");

  // Clear the seen flags for the numeric factorization.
  for (SymbolicGraph::iterator ii = graph.begin(), ei = graph.end();
       ii != ei; ++ii) {
    Node& data = graph.getData(*ii);
    assert(data.seen);
    data.seen = false;
  }

  // We should now have built a directed graph (outgraph) and total
  // ordering. Now run the numeric factorization.
  //
  // FIXME: Convert back to a LC_Graph?
  std::cout << "Numeric factorization\n";
  run(NumericAlgo<Graph>(outgraph), "NumericTime");

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify && !verify(outgraph)) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }
  
  get_solution_from_graph(outgraph, rhs);
  
  printf("\nTime taken: %.4fs\n", (float)(clock() - t_start)/CLOCKS_PER_SEC);
  return 0;
}


