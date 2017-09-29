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

#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#ifdef GALOIS_USE_EXP
#include "galois/PriorityScheduling.h"
#endif

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>

#define BORUVKA_DEBUG 0
#define COMPILE_STATISICS 0
#if BORUVKA_DEBUG
#include "UnionFind.h"
#endif
#if COMPILE_STATISICS
#include <time.h>
int BORUVKA_SAMPLE_FREQUENCY= 1000000;
#endif

using namespace std;
namespace cll = llvm::cl;

static const char* name = "Boruvka's Minimum Spanning Tree Algorithm";
static const char* desc = "Computes a minimum weight spanning tree of a graph";
static const char* url = "mst";

static cll::opt<std::string> inputfile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<bool> use_weighted_rmat("wrmat",cll::desc("Weighted RMAT"), cll::Optional,cll::init(false));
static cll::opt<bool> verify_via_kruskal("verify",cll::desc("Verify MST result via Serial Kruskal"), cll::Optional,cll::init(false));
static int nodeID = 0;
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
struct Node {
   //Do not include node data if not debugging since
   //it is a useless overhead. Useful for debugging though.
#if BORUVKA_DEBUG
   int id;
   Node(int i=-1) :
      id(i) {
   }
   std::string toString() {
      std::ostringstream s;
      s << "N(" << id << ")";
      return s.str();
   }
#else
   Node(int){};
   Node(){};
#endif
};
std::ostream& operator<<(std::ostream& s, Node& n) {
#if BORUVKA_DEBUG
   s << n.toString();
#endif
   return s;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
typedef long NodeDataType;
typedef int EdgeDataType;

typedef galois::graphs::FirstGraph<Node, EdgeDataType, false> Graph;
typedef Graph::GraphNode GNode;
//The graph.
Graph graph;
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
#if COMPILE_STATISICS
struct GraphStats{
   std::vector<float> average_degrees;
   std::vector<int> time_vals;
   std::vector<unsigned long> max_degrees;
   unsigned long counter;
   GraphStats(){
      counter=0;
   }
   GraphStats &  tick(){
      ++counter;
      return *this;
   }
   void snap(){
      unsigned long num_nodes=0;
      unsigned long num_edges=0;
      unsigned long current_degree = 0;
      unsigned long max_degree=0;
      for(Graph::iterator it = graph.begin(), end_it = graph.end(); it!=end_it; ++it){
         ++num_nodes;
         current_degree=0;
         auto d = std::distance(graph.edge_begin(*it, galois::MethodFlag::UNPROTECTED), graph.edge_end(*it, galois::MethodFlag::UNPROTECTED));
         num_edges += d;
         current_degree += e;
         }
         if(current_degree > max_degree) max_degree = current_degree;
      }
      time_vals.push_back(time(0));
      average_degrees.push_back((float)(num_edges)/(num_nodes));
      max_degrees.push_back(max_degree);
   }
   void dump(ostream & out){
      out<<"\nMax degrees,";
      for(size_t i=0;i<max_degrees.size(); ++i){
         out << " " << max_degrees[i]<<",";
      }
      out<<"\nAverage degrees,";
      for(size_t i=0;i<average_degrees.size(); ++i){
         out << " " << average_degrees[i]<<",";
      }
      out<<"\nTime, ";
      for(size_t i=0;i<time_vals.size(); ++i){
         out << " " << time_vals[i]<<",";
      }
      out<<"\n";
   }
};
GraphStats stat_collector;
#endif

/////////////////////////////////////////////////////////////////////////////////////////
void printGraph() {
   int numEdges = 0;
   for (auto src : graph) {
      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      if (graph.containsNode(src))
        for (auto dst : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
            EdgeDataType w = graph.getEdgeData(dst);
            assert(w>=0);
            Node & ddata = graph.getData(graph.getEdgeDst(dst));
            std::cout << "1) " << sdata << " => " << ddata << " [ " << w << " ] " << std::endl;
            numEdges++;
         }
   }
   std::cout << "Num edges " << numEdges << std::endl;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
//! [define per thread storage]
galois::substrate::PerThreadStorage<long long> MSTWeight;
//! [define per thread storage]
struct process {
   template<typename ContextTy>
   void operator()(GNode& src, ContextTy& lwl) {
      if (graph.containsNode(src) == false)
         return;
      graph.getData(src, galois::MethodFlag::WRITE);
      GNode minNeighbor = 0;
#if BORUVKA_DEBUG
      std::cout<<"Processing "<<graph.getData(src).toString()<<std::endl;
#endif
      EdgeDataType minEdgeWeight = std::numeric_limits<EdgeDataType>::max();
      //Acquire locks on neighborhood.
      for (auto dst : graph.edges(src, galois::MethodFlag::WRITE)) {
         graph.getData(graph.getEdgeDst(dst));
      }
      //Find minimum neighbor
      for (auto e_it : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
         EdgeDataType w = graph.getEdgeData(e_it, galois::MethodFlag::UNPROTECTED);
         assert(w>=0);
         if (w < minEdgeWeight) {
           minNeighbor = graph.getEdgeDst(e_it);
            minEdgeWeight = w;
         }
      }
      //If there are no outgoing neighbors.
      if (minEdgeWeight == std::numeric_limits<EdgeDataType>::max()) {
         graph.removeNode(src, galois::MethodFlag::UNPROTECTED);
         return;
      }
#if BORUVKA_DEBUG
            std::cout << " Min edge from "<<graph.getData(src) << " to "<<graph.getData(minNeighbor)<<" " <<minEdgeWeight << " "<<std::endl;
#endif
      //Acquire locks on neighborhood of min neighbor.
      for (auto e_it : graph.edges(minNeighbor, galois::MethodFlag::WRITE)) {
         graph.getData(graph.getEdgeDst(e_it));
      }
      assert(minEdgeWeight>=0);
      //update MST weight.
      //! [access perThreadStorage] 
      *MSTWeight.getLocal() += minEdgeWeight;
      //! [access perThreadStorage] 
      typedef std::pair<GNode, EdgeDataType> EdgeData;
      typedef std::set<EdgeData, std::less<EdgeData>, galois::PerIterAllocTy::rebind<EdgeData>::other> edsetTy;
      edsetTy toAdd(std::less<EdgeData>(), galois::PerIterAllocTy::rebind<EdgeData>::other(lwl.getPerIterAlloc()));
      for (auto mdst : graph.edges(minNeighbor, galois::MethodFlag::UNPROTECTED)) {
         GNode dstNode = graph.getEdgeDst(mdst);
         int edgeWeight = graph.getEdgeData(mdst,galois::MethodFlag::UNPROTECTED);
         if (dstNode != src) { //Do not add the edge being contracted
            Graph::edge_iterator dup_edge = graph.findEdge(src, dstNode, galois::MethodFlag::UNPROTECTED);
            if (dup_edge != graph.edge_end(src, galois::MethodFlag::UNPROTECTED)) {
               EdgeDataType dup_wt = graph.getEdgeData(dup_edge,galois::MethodFlag::UNPROTECTED);
                  graph.getEdgeData(dup_edge,galois::MethodFlag::UNPROTECTED) = std::min<EdgeDataType>(edgeWeight, dup_wt);
                  assert(std::min<EdgeDataType>(edgeWeight, dup_wt)>=0);
            } else {
                  toAdd.insert(EdgeData(dstNode, edgeWeight));
                  assert(edgeWeight>=0);
            }
         }
      }
      graph.removeNode(minNeighbor, galois::MethodFlag::UNPROTECTED);
      for (edsetTy::iterator it = toAdd.begin(), endIt = toAdd.end(); it != endIt; it++) {
         graph.getEdgeData(graph.addEdge(src, it->first, galois::MethodFlag::UNPROTECTED)) = it->second;
      }
      lwl.push(src);
#if COMPILE_STATISICS
      if(stat_collector.tick().counter%BORUVKA_SAMPLE_FREQUENCY==0)
         stat_collector.snap();
#endif
   }
};
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
struct Indexer: public std::unary_function<const GNode&, unsigned> {
   unsigned operator()(const GNode& n) {
      return std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED), graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
   }
   static unsigned foo(const GNode& n) {
      return std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED), graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
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
EdgeDataType runBodyParallel() {
   using namespace galois::worklists;
   typedef dChunkedFIFO<64> dChunk;
   typedef ChunkedFIFO<64> Chunk;
   typedef OrderedByIntegerMetric<Indexer, dChunk> OBIM;

#if BORUVKA_DEBUG
   std::cout<<"Graph size "<<graph.size()<<std::endl;
#endif
#if COMPILE_STATISICS
   BORUVKA_SAMPLE_FREQUENCY = graph.size()/200;//200 samples should be sufficient.
#endif
   for (size_t i = 0; i < MSTWeight.size(); i++){
      *MSTWeight.getRemote(i) = 0;
   }
   cout << "Starting loop body\n";
   galois::StatTimer T;
   T.start();
#ifdef GALOIS_USE_EXP
   Exp::PriAuto<64, Indexer, OBIM, seq_less, seq_gt>::for_each(graph.begin(), graph.end(), process());
#else
   galois::for_each(graph, process(), galois::wl<OBIM>(), galois::loopname("Main"));
#endif
   T.stop();

   EdgeDataType res = 0;
   for (size_t i = 0; i < MSTWeight.size(); i++) {
#if BORUVKA_DEBUG
      std::cout<<"MST +=" << *MSTWeight.getRemote(i)<<std::endl;
#endif
      res += *MSTWeight.getRemote(i);
   }
//   std::cout << "MST Weight is " << res <<std::endl;
   return res;
}
/////////////////////////////////////////////////////////////////////////////////////////
static void makeGraph(const char* input) {
   std::vector<GNode> nodes;
   //Create local computation graph.
   typedef galois::graphs::LC_CSR_Graph<Node, EdgeDataType> InGraph;
   typedef InGraph::GraphNode InGNode;
   InGraph in_graph;
   //Read graph from file.
   galois::graphs::readGraph(in_graph, input);
   std::cout << "Read " << in_graph.size() << " nodes\n";
   //A node and a int is an element.
   typedef std::pair<InGNode, EdgeDataType> Element;
   //A vector of element is 'Elements'
   typedef std::vector<Element> Elements;
   //A vector of 'Elements' is a 'Map'
   typedef std::vector<Elements> Map;
   //'in_edges' is a vector of vector of pairs of nodes and int.
   Map edges(in_graph.size());
   //
   int numEdges = 0;
   for (auto src : in_graph) {
     for (auto dst : in_graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
         if (src == *dst) {
#if BORUVKA_DEBUG
            std::cout<<"ERR:: Self loop at "<<*src<<std::endl;
#endif
            continue;
         }
         EdgeDataType w = in_graph.getEdgeData(dst);
         Element e(src, w);
         edges[in_graph.getEdgeDst(dst)].push_back(e);
         numEdges++;
      }
   }
#if BORUVKA_DEBUG
   std::cout<<"Number of edges "<<numEdges<<std::endl;
#endif
   nodes.resize(in_graph.size());
   for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
      Node n(nodeID);
      GNode node = graph.createNode(n);
      graph.addNode(node);
      nodes[nodeID] = node;
      nodeID++;
   }

   int id = 0;
   numEdges = 0;
   EdgeDataType edge_sum = 0;
   int numDups = 0;
   for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
      GNode src = nodes[id];
      for (Elements::iterator j = i->begin(), ej = i->end(); j != ej; ++j) {
         Graph::edge_iterator it = graph.findEdge(src, nodes[j->first], galois::MethodFlag::UNPROTECTED);
         if (it != graph.edge_end(src, galois::MethodFlag::UNPROTECTED)) {
            numDups++;
            EdgeDataType w = (graph.getEdgeData(it));
            if (j->second < w) {
               graph.getEdgeData(it) = j->second;
               edge_sum += (j->second-w);
            }
         } else {
            graph.getEdgeData(graph.addEdge(src, nodes[j->first], galois::MethodFlag::UNPROTECTED)) = j->second;
            edge_sum += j->second;
         }
         numEdges++;
         assert(edge_sum < std::numeric_limits<EdgeDataType>::max());
      }
      id++;
   }
#if BORUVKA_DEBUG
   std::cout << "Final num edges " << numEdges << " Dups " << numDups << " sum :" << edge_sum << std::endl;
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////RMAT Reading function//////////////////////////////////
template<typename NdTy, typename EdTy>
struct EdgeTuple{
   NdTy src;
   NdTy dst;
   EdTy wt;
   EdgeTuple(NdTy s, NdTy d, EdTy w):src(s),dst(d),wt(w){};
   void print()const{
      cout << "[" << src << ", " << dst << " , " << wt << "]\n";
   }
};
template<typename NdTy, typename EdTy>
struct LessThanEdgeTuple{
bool operator()(const EdgeTuple<NdTy,EdTy> & o1, const EdgeTuple<NdTy,EdTy> &o2){
   return (o1.wt==o2.wt)? o1.src < o2.src : o1.wt<o2.wt;
}
};
template<typename NdTy, typename EdTy>
std::ostream & operator << (std::ostream & out , const EdgeTuple<NdTy,EdTy> & e ){
   e.print();
   return out;
}
static void readWeightedRMAT(const char* input) {
   std::vector<EdgeTuple<NodeDataType,EdgeDataType> > et;
   et.reserve(100000000);
   ifstream inFile (input);
   NodeDataType src, dst;
   EdgeDataType wt;
   char header[30];
   inFile.seekg(0, ios::beg);
   inFile>>header;
   NodeDataType max_id=0;
   while(inFile.eof()==false){
      inFile>>src;
      inFile>>dst;
      inFile>>wt;
      max_id = max(max_id, (src>dst?src:dst));
      et.push_back(EdgeTuple<NodeDataType, EdgeDataType>(src,dst,wt));
   }

   std::vector<GNode> nodes;

   nodes.resize(max_id+1);
   for (NodeDataType l = 0; l < max_id+1 ; ++l) {
      Node n(nodeID);
      GNode node = graph.createNode(n);
      nodes[nodeID] = node;
      nodeID++;
   }

   long numEdges = 0;
   EdgeDataType edge_sum = 0;
   int numDups = 0;
   for (std::vector<EdgeTuple<NodeDataType, EdgeDataType> >::iterator eIt = et.begin(), end = et.end(); eIt!=end; ++eIt) {
      EdgeTuple<NodeDataType,EdgeDataType> e = *eIt;
      Graph::edge_iterator it = graph.findEdge(nodes[e.src], nodes[e.dst], galois::MethodFlag::UNPROTECTED);
         if (it != graph.edge_end(nodes[e.src], galois::MethodFlag::UNPROTECTED)) {
            numDups++;
            EdgeDataType w = (graph.getEdgeData(it));
            if (e.wt < w) {
               graph.getEdgeData(it) = e.wt;
               edge_sum += (e.wt-w);
            }
         } else {
            graph.getEdgeData(graph.addEdge(nodes[e.src], nodes[e.dst], galois::MethodFlag::UNPROTECTED)) = e.wt;
            edge_sum += e.wt;
         }
         numEdges++;
         assert(edge_sum < std::numeric_limits<EdgeDataType>::max());
   }
}

////////////////////////////End READ WRMAT////////////////////////////////////////////////
////////////////////////////Kruskal////////////////////////////////////////////////
#if BORUVKA_DEBUG
typedef EdgeTuple<NodeDataType, EdgeDataType> KEdgeTuple;
typedef std::vector<KEdgeTuple> KruskalGraph;
KruskalGraph read_edges(Graph  &g){
   KruskalGraph  * ret = new KruskalGraph ();
   for (auto n : g) {
     for (auto e_it : g.edges(n)) {
      ret->push_back(KEdgeTuple(g.getData(n).id,g.getData(g.getEdgeDst(e_it)).id, g.getEdgeData(e_it) ));
   }
}
std::cout<<"Number of edge tuples " << ret->size() << "\n";
return *ret;
}
EdgeDataType kruskal_impl(const size_t num_nodes, KruskalGraph kg){
   std::sort(kg.begin(), kg.end(), LessThanEdgeTuple<NodeDataType,EdgeDataType>());
   UnionFind<NodeDataType,-1> uf(num_nodes);
   size_t mst_size = 0;
   EdgeDataType mst_sum = 0;
   for(size_t i = 0; i < kg.size(); ++i){
      KEdgeTuple e  = kg[i];
      NodeDataType src = uf.uf_find(e.src);
      NodeDataType dst = uf.uf_find(e.dst);
      if(src!=dst){
         uf.uf_union(src,dst);
         mst_sum+=e.wt;
         mst_size++;
         if(mst_size>=num_nodes-1)
            return mst_sum;
      }
   }
   return -1;
}
EdgeDataType verify(Graph & g){
   return kruskal_impl(g.size(), read_edges(g));
}
#endif
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   galois::SharedMemSys G;
   LonestarStart(argc, argv, name, desc, url);
   if(use_weighted_rmat)
      readWeightedRMAT(inputfile.c_str());
   else
      makeGraph(inputfile.c_str());
#if BORUVKA_DEBUG
   EdgeDataType kruskal_wt;
   if(verify_via_kruskal){
      kruskal_wt= verify(graph);
      cout<<"Kruskal MST Result is " << kruskal_wt <<"\n";
   }
#endif
   cout << "Starting loop body\n";
   EdgeDataType mst_wt = runBodyParallel();
   cout<<"Boruvka MST Result is " << mst_wt <<"\n";
#if BORUVKA_DEBUG
   if(verify_via_kruskal){
      assert(kruskal_wt==mst_wt);
   }
#endif

#if COMPILE_STATISICS
   cout<< " \n==================================================\n";
   stat_collector.dump(cout);
   cout<< " \n==================================================\n";
#endif
   return 0;
}
