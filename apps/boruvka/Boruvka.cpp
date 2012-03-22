/** Boruvka application -*- C++ -*-
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Rashid Kaleem <rashid@cs.utexas.edu>
 */

#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph2.h"
#include "Galois/Timer.h"
#include "Galois/Galois.h"
#include "Galois/Graphs/LCGraph.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#ifdef GALOIS_EXP
#include "Galois/PriorityScheduling.h"
#endif

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <map>
#include <set>

#define BORUVKA_DEBUG 0

using namespace std;
namespace cll = llvm::cl;

static const char* name = "Boruvka MST";
static const char* desc = "Computes the Minimal Spanning Tree using Boruvka\n";
static const char* url = "boruvkas_algorithm";

static cll::opt<std::string> inputfile(cll::Positional, cll::desc("<input file>"), cll::Required);

static int nodeID = 0;
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
struct Node {
//   int id;
   Node(){};
//   Node(int i = -1) :
//      id(i) {
//   }
   std::string toString() {
      std::ostringstream s;
//      s << "N(" << id << ")";
      return s.str();
   }
};
std::ostream& operator<<(std::ostream& s, Node& n) {
   s << n.toString();
   return s;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
typedef Galois::Graph::FirstGraph<Node, int, false> Graph;
typedef Graph::GraphNode GNode;
//The graph.
Graph graph;
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void printGraph() {
   int numEdges = 0;
      for (Graph::iterator src = graph.begin(), esrc = graph.end(); src != esrc; ++src) {
      Node& sdata = graph.getData(*src, Galois::NONE);
      if (graph.containsNode(*src))
         for (Graph::edge_iterator dst = graph.edge_begin(*src, Galois::NONE), edst = graph.edge_end(*src, Galois::NONE); dst != edst; ++dst) {
            int w = graph.getEdgeData(dst);
            assert(w>=0);
            Node & ddata = graph.getData(graph.getEdgeDst(dst));
            std::cout << "1) " << sdata.toString() << " => " << ddata.toString() << " [ " << w << " ] " << std::endl;
            numEdges++;
         }
   }
   std::cout << "Num edges " << numEdges << std::endl;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
GaloisRuntime::PerCPU<long> MSTWeight;
//GaloisRuntime::PerCPU<long> NumIncrements;
struct process {
   template<typename ContextTy>
   void __attribute__((noinline)) operator()(GNode& src, ContextTy& lwl) {
      if (graph.containsNode(src) == false)
         return;
      graph.getData(src, Galois::ALL);
      GNode * minNeighbor = 0;
#if BORUVKA_DEBUG
      std::cout<<"Processing "<<graph.getData(src).toString()<<std::endl;
#endif
      int minEdgeWeight = std::numeric_limits<int>::max();
      //Acquire locks on neighborhood.
      for (Graph::edge_iterator dst = graph.edge_begin(src, Galois::ALL), edst = graph.edge_end(src, Galois::ALL); dst != edst; ++dst) {
         graph.getData(graph.getEdgeDst(dst));
      }
      //Find minimum neighbor
      for (Graph::edge_iterator e_it = graph.edge_begin(src, Galois::ALL), edst = graph.edge_end(src, Galois::ALL); e_it != edst; ++e_it) {
         int w = graph.getEdgeData(e_it);
         assert(w>=0);
         if (w < minEdgeWeight) {
            minNeighbor = &((*e_it).first());
            minEdgeWeight = w;
         }
      }
      //If there are no outgoing neighbors.
      if (minEdgeWeight == std::numeric_limits<int>::max()) {
         graph.removeNode(src, Galois::ALL);
         //XXX remove the return stmt to have a single point of exit.
         return;
      }
#if BORUVKA_DEBUG
      		std::cout << " Min edge from "<<graph.getData(src).toString() << " to "<<graph.getData minNeighbor.toString()<<" " <<minEdgeWeight << " "<<std::endl;
#endif
      //Acquire locks on neighborhood of min neighbor.
      for (Graph::edge_iterator e_it = graph.edge_begin(*minNeighbor, Galois::ALL), edst = graph.edge_end(*minNeighbor, Galois::ALL); e_it != edst; ++e_it) {
         graph.getData(graph.getEdgeDst(e_it));
      }
      assert(minEdgeWeight>=0);
      //update MST weight.
      MSTWeight.get() += minEdgeWeight;
//      NumIncrements.get()+=1;
      typedef std::pair<GNode, int> EdgeData;
      typedef std::set<EdgeData, std::less<EdgeData>, Galois::PerIterAllocTy::rebind<EdgeData>::other> edsetTy;
      edsetTy toAdd(std::less<EdgeData>(), Galois::PerIterAllocTy::rebind<EdgeData>::other(lwl.getPerIterAlloc()));
      for (Graph::edge_iterator mdst = graph.edge_begin(*minNeighbor, Galois::ALL), medst = graph.edge_end(*minNeighbor, Galois::ALL); mdst != medst; ++mdst) {
         GNode dstNode = graph.getEdgeDst(mdst);
         int edgeWeight = graph.getEdgeData(mdst);
         if (dstNode != src) { //Do not add the edge being contracted
            Graph::edge_iterator dup_edge = graph.findEdge(src, dstNode, Galois::ALL);
            if (dup_edge != graph.edge_end(src, Galois::ALL)) {
                  int dup_wt = graph.getEdgeData(dup_edge);
                  graph.getEdgeData(dup_edge) = std::min<int>(edgeWeight, dup_wt);
                  assert(std::min<int>(edgeWeight, dup_wt)>=0);
            } else {
                  toAdd.insert(EdgeData(dstNode, edgeWeight));
                  assert(edgeWeight>=0);
            }
         }
      }
      graph.removeNode(*minNeighbor, Galois::ALL);
      for (edsetTy::iterator it = toAdd.begin(), endIt = toAdd.end(); it != endIt; it++) {
         graph.getEdgeData(graph.addEdge(src, it->first, Galois::ALL)) = it->second;
      }
      lwl.push(src);

   }
};
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
struct Indexer: public std::unary_function<const GNode&, int> {
   unsigned operator()(const GNode& n) {
      return std::distance(graph.edge_begin(n, Galois::NONE), graph.edge_end(n, Galois::NONE));
   }
   static unsigned foo(const GNode& n) {
      return std::distance(graph.edge_begin(n, Galois::NONE), graph.edge_end(n, Galois::NONE));
   }
};
struct seq_less: public std::binary_function<const GNode&, const GNode&, bool> {
   bool operator()(const GNode& lhs, const GNode& rhs) const {
      if (Indexer::foo(lhs) < Indexer::foo(rhs))
         return true;
      if (Indexer::foo(lhs) > Indexer::foo(rhs))
         return false;
      return lhs < rhs;
   }
};
struct seq_gt: public std::binary_function<const GNode&, const GNode&, bool> {
   bool operator()(const GNode& lhs, const GNode& rhs) const {
      if (Indexer::foo(lhs) > Indexer::foo(rhs))
         return true;
      if (Indexer::foo(lhs) < Indexer::foo(rhs))
         return false;
      return lhs > rhs;
   }
};
//End body of for-each.
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void runBodyParallel() {
   using namespace GaloisRuntime::WorkList;
   typedef dChunkedFIFO<64> dChunk;
   typedef ChunkedFIFO<64> Chunk;
   typedef OrderedByIntegerMetric<Indexer, dChunk> OBIM;

#if BORUVKA_DEBUG
   std::cout<<"Graph size "<<graph.size()<<std::endl;
#endif

   for (size_t i = 0; i < MSTWeight.size(); i++){
      MSTWeight.get(i) = 0;
   }

   Galois::StatTimer T;

   T.start();
#ifdef GALOIS_EXP
   Exp::WorklistExperiment<OBIM, dChunk, Chunk, Indexer, seq_less, seq_gt>().for_each(std::cout, graph.begin(), graph.end(), process());
#else
   Galois::for_each<dChunk>(graph.begin(), graph.end(), process());
#endif
   T.stop();

   //TODO: use a reduction variable here
   long res = 0;
//   long sum_incs = 0;
   for (size_t i = 0; i < MSTWeight.size(); i++) {
#if BORUVKA_DEBUG
      std::cout<<"MST +=" << MSTWeight.get(i)<<std::endl;
#endif
      res += MSTWeight.get(i);
//      sum_incs += NumIncrements.get(i);
   }
//   std::cout << "MST Weight is " << res << " number of increments is " << sum_incs <<std::endl;
   std::cout << "MST Weight is " << res <<std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
static void makeGraph(const char* input) {
   std::vector<GNode> nodes;
   //Create local computation graph.
   typedef Galois::Graph::LC_CRS_Graph<Node, int> InGraph;
   typedef InGraph::GraphNode InGNode;
   InGraph in_graph;
   //Read graph from file.
   in_graph.structureFromFile(input);
   std::cout << "Read " << in_graph.size() << " nodes\n";
   //A node and a int is an element.
   typedef std::pair<InGNode, int> Element;
   //A vector of element is 'Elements'
   typedef std::vector<Element> Elements;
   //A vector of 'Elements' is a 'Map'
   typedef std::vector<Elements> Map;
   //'in_edges' is a vector of vector of pairs of nodes and int.
   Map edges(in_graph.size());
   //
   int numEdges = 0;
   for (InGraph::iterator src = in_graph.begin(), esrc = in_graph.end(); src != esrc; ++src) {
      for (InGraph::edge_iterator dst = in_graph.edge_begin(*src, Galois::NONE), edst = in_graph.edge_end(*src, Galois::NONE); dst != edst; ++dst) {
         if (*src == *dst) {
#if BORUVKA_DEBUG
            std::cout<<"ERR:: Self loop at "<<*src<<std::endl;
#endif
            continue;
         }
         int w = in_graph.getEdgeData(dst);
         Element e(*src, w);
         edges[in_graph.getEdgeDst(dst)].push_back(e);
         numEdges++;
      }
   }
#if BORUVKA_DEBUG
   std::cout<<"Number of edges "<<numEdges<<std::endl;
#endif
   int id = 0;
   nodes.resize(in_graph.size());
   for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
//      Node n(nodeID);
      Node n;
      GNode node = graph.createNode(n);
      nodes[nodeID] = node;
      nodeID++;
   }

   id = 0;
   numEdges = 0;
   long long edge_sum = 0;
   int numDups = 0;
   for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
      GNode src = nodes[id];
      for (Elements::iterator j = i->begin(), ej = i->end(); j != ej; ++j) {
         Graph::edge_iterator it = graph.findEdge(src, nodes[j->first], Galois::NONE);
         if (it != graph.edge_end(src, Galois::NONE)) {
            numDups++;
            int w = (graph.getEdgeData(it));
            if (j->second < w) {
               graph.getEdgeData(it) = j->second;
               edge_sum += (j->second-w);
            }
         } else {
            graph.getEdgeData(graph.addEdge(src, nodes[j->first], Galois::NONE)) = j->second;
            edge_sum += j->second;
         }
         numEdges++;
         assert(edge_sum < std::numeric_limits<long>::max());
      }
      id++;
   }
#if BORUVKA_DEBUG
   std::cout << "Final num edges " << numEdges << " Dups " << numDups << " sum :" << edge_sum << std::endl;
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
   LonestarStart(argc, argv, std::cout, name, desc, url);
   makeGraph(inputfile.c_str());
#if BORUVKA_DEBUG
   printGraph();
#endif
   runBodyParallel();
   return 0;
}
