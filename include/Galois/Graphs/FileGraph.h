/** Basic serialized graphs -*- C++ -*-
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
 * @section Description
 *
 * There are two main classes, ::FileGraph and ::LC_FileGraph. The former
 * represents the pure structure of a graph (i.e., whether an edge exists between
 * two nodes) and cannot be modified. The latter allows values to be stored on
 * nodes and edges, but the structure of the graph cannot be modified.
 *
 * An example of use:
 * 
 * \code
 * typedef Galois::Graph::LC_FileGraph<int,int> Graph;
 * 
 * // Create graph
 * Graph g;
 * g.structureFromFile(inputfile);
 *
 * // Traverse graph
 * for (Graph::active_iterator i = g.active_begin(), iend = g.active_end();
 *      i != iend;
 *      ++i) {
 *   Graph::GraphNode src = *i;
 *   for (Graph::neighbor_iterator j = g.neighbor_begin(src),
 *                                 jend = g.neighbor_end(src);
 *        j != jend;
 *        ++j) {
 *     Graph::GraphNode dst = *j;
 *     int edgeData = g.getEdgeData(src, dst);
 *     int nodeData = g.getData(dst);
 *   }
 * }
 * \endcode
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_GRAPHS_FILEGRAPH_H
#define GALOIS_GRAPHS_FILEGRAPH_H

#include "Galois/ConflictFlags.h"
#include "Galois/Runtime/Context.h"
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/detail/endian.hpp>
#include <map>
#include <fstream>

#ifdef GALOIS_NUMA
#include <numa.h>
#endif

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

//! Graph serialized to a file
class FileGraph {
public:
  typedef uint32_t GraphNode;

protected:
  void* volatile masterMapping;
  size_t masterLength;
  uint64_t sizeEdgeTy;
  int masterFD;

  uint64_t* outIdx;
  uint32_t* outs;

  char* edgeData;

  uint64_t numEdges;
  uint64_t numNodes;

  uint64_t getEdgeIdx(GraphNode src, GraphNode dst) {
    for (uint32_t* ii = raw_neighbor_begin(src),
	   *ee = raw_neighbor_end(src); ii != ee; ++ii)
      if (convert32(*ii) == dst)
	return std::distance(outs, ii);
    return ~(uint64_t)0;
  }

  static uint32_t swap32(int32_t x) {
#ifdef __GNUC__
    return __builtin_bswap32(x);
#else
    return (x >> 24) | 
           ((x << 8) & 0x00FF0000) |
           ((x >> 8) & 0x0000FF00) |
           (x << 24);
#endif
  }

  static uint64_t swap64(int64_t x) {
#ifdef __GNUC__
    return __builtin_bswap64(x);
#else
    return (x>>56) | 
        ((x<<40) & 0x00FF000000000000) |
        ((x<<24) & 0x0000FF0000000000) |
        ((x<<8)  & 0x000000FF00000000) |
        ((x>>8)  & 0x00000000FF000000) |
        ((x>>24) & 0x0000000000FF0000) |
        ((x>>40) & 0x000000000000FF00) |
        (x<<56);
#endif
  }

  static uint32_t convert32(int32_t x) {
#ifdef BOOST_BIG_ENDIAN
    return swap32(x);
#else
    return x;
#endif
  }

  static uint64_t convert64(int64_t x) {
#ifdef BOOST_BIG_ENDIAN
    return swap64(x);
#else
    return x;
#endif
  }

  uint32_t* raw_neighbor_begin(GraphNode N, MethodFlag mflag = ALL) const {
    return (N == 0) ? &outs[0] : &outs[convert64(outIdx[N-1])];
  }

  uint32_t* raw_neighbor_end(GraphNode N, MethodFlag mflag = ALL) const {
    return &outs[convert64(outIdx[N])];
  }

  struct Convert : public std::unary_function<uint32_t, uint32_t> {
    uint32_t operator()(uint32_t x) const {
      return convert32(x);
    }
  };

  void parse(void* m);

public:
  bool isLoaded() {
    return masterMapping != 0;
  }

  void* getBasePtr() {
    return masterMapping;
  }
  size_t getBaseLength() {
    return masterLength;
  }

  // Node Handling

  //! Check if a node is in the graph (already added)
  bool containsNode(const GraphNode n) const {
    return n < numNodes;
  }

  // Edge Handling
  template<typename EdgeTy>
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    assert(sizeEdgeTy == sizeof(EdgeTy));
    return ((EdgeTy*)edgeData)[getEdgeIdx(src, dst)];
  }

  // General Things
#ifdef BOOST_LITTLE_ENDIAN
  typedef uint32_t* neighbor_iterator;
#else
  typedef boost::transform_iterator<Convert, uint32_t*> neighbor_iterator;
#endif

  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL) const {
#ifdef BOOST_LITTLE_ENDIAN
      return raw_neighbor_begin(N, mflag);
#else
      return boost::make_transform_iterator(raw_neighbor_begin(N, mflag), Convert());
#endif
  }

  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL) const {
#ifdef BOOST_LITTLE_ENDIAN
      return raw_neighbor_end(N, mflag);
#else
      return boost::make_transform_iterator(raw_neighbor_end(N, mflag), Convert());
#endif
  }

  bool has_neighbor(GraphNode N1, GraphNode N2, MethodFlag mflag = ALL) const {
    return std::find(neighbor_begin(N1), neighbor_end(N1), N2) != neighbor_end(N1);
  }

  typedef boost::counting_iterator<uint64_t> active_iterator;

  //! Iterate over nodes in graph (not thread safe)
  active_iterator active_begin() const {
    return active_iterator(0);
  }

  active_iterator active_end() const {
    return active_iterator(numNodes);
  }

  //! The number of nodes in the graph
  unsigned int size() const {
    return numNodes;
  }

  //! The number of edges in the graph
  unsigned int sizeEdges () const {
    return numEdges;
  }

  FileGraph();
  ~FileGraph();

  //! Read graph connectivity information from file
  void structureFromFile(const char* filename);

  //! Read graph connectivity information from memory
  void structureFromMem(void* mem, size_t len, bool clone = false);

  //! Read graph connectivity information from graph
  template<typename TyG>
  void structureFromGraph(TyG& G) {
    //version
    uint64_t version = 1;
    uint64_t sizeof_edge_data = sizeof(typename TyG::EdgeDataTy);
    uint64_t num_nodes = G.size();

    uint64_t nBytes = sizeof(uint64_t) * 3;

    typedef typename TyG::GraphNode GNode;
    typedef std::vector<GNode> Nodes;
    Nodes nodes(G.active_begin(), G.active_end());

    //num edges and outidx computation
    uint64_t offset = 0;
    std::vector<uint64_t> out_idx;
    std::map<typename TyG::GraphNode, uint32_t> node_ids;
    for (uint32_t id = 0; id < num_nodes; ++id) {
      GNode& node = nodes[id];
      node_ids[node] = id;
      offset += G.neighborsSize(node);
      out_idx.push_back(offset);
    }

    nBytes += sizeof(uint64_t);
    nBytes += sizeof(uint64_t) * out_idx.size();

    //outs
    std::vector<uint32_t> outs;
    size_t num_edges = 0;
    for (typename Nodes::iterator ii = nodes.begin(), ee = nodes.end();
	 ii != ee; ++ii) {
      for (typename TyG::neighbor_iterator ni = G.neighbor_begin(*ii),
	     ne = G.neighbor_end(*ii); ni != ne; ++ni, ++num_edges) {
	uint32_t id = node_ids[*ni];
	outs.push_back(id);
      }
    }
    if (num_edges % 2) {
      outs.push_back(0);
    }

    nBytes += sizeof(uint32_t) * outs.size();
    
    //edgeData
    std::vector<typename TyG::EdgeDataTy> edgeData;
    for (typename Nodes::iterator ii = nodes.begin(), ee = nodes.end();
	 ii != ee; ++ii) {
      for (typename TyG::neighbor_iterator ni = G.neighbor_begin(*ii),
	     ne = G.neighbor_end(*ii); ni != ne; ++ni) {
	edgeData.push_back(G.getEdgeData(*ii, *ni));
      }
    }
    
    nBytes += sizeof(typename TyG::EdgeDataTy) * edgeData.size();

    char* t = (char*)malloc(nBytes);
    char* base = t;
    memcpy(t, &version, sizeof(uint64_t));
    t += sizeof(uint64_t);
    memcpy(t, &sizeof_edge_data, sizeof(uint64_t));
    t += sizeof(uint64_t);
    memcpy(t, &num_nodes, sizeof(uint64_t));
    t += sizeof(uint64_t);
    memcpy(t, &offset, sizeof(uint64_t));
    t += sizeof(uint64_t);
    memcpy(t, &out_idx[0], sizeof(uint64_t) * out_idx.size());
    t += sizeof(uint64_t) * out_idx.size();
    memcpy(t, &outs[0], sizeof(uint32_t) * outs.size());
    t += sizeof(uint32_t) * outs.size();
    memcpy(t, &edgeData[0], sizeof(typename TyG::EdgeDataTy) * edgeData.size());
    
    structureFromMem(base, nBytes, true);
    free(t);
  }

  //! Write graph connectivity information to file
  void structureToFile(char* file);

  void swap(FileGraph& other);
  void clone(FileGraph& other);

};

#ifdef GALOIS_NUMA
class NumaFileGraph {
public:
  typedef uint32_t GraphNode;

protected:
  GaloisRuntime::PerLevel<FileGraph> graphs;
  GaloisRuntime::PerLevel<GaloisRuntime::SimpleLock<int, true> > locks;

  FileGraph& ldIfNeeded() {
    FileGraph& g = graphs.get();
    while (!g.isLoaded()) {
      __sync_synchronize();
      if (locks.get().try_lock()) {
	if (!g.isLoaded()) {
	  g.structureFromMem(graphs.get(0).getBasePtr(), graphs.get(0).getBaseLength(), true);
	  //std::cout << "Fixed " << graphs.myEffectiveID() << " by " << graphs.myID() << "\n";
	}
	locks.get().unlock();
      }
      //std::cout << graphs.myID() << ' ';
    }
    return g;
  }

public:
  // Node Handling

  //! Check if a node is in the graph (already added)
  bool containsNode(const GraphNode n) {
    return ldIfNeeded().containsNode(n);
  }

  // Edge Handling
  template<typename EdgeTy>
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    return ldIfNeeded().getEdgeData<EdgeTy>(src,dst,mflag);
  }

  // General Things
  typedef FileGraph::neighbor_iterator neighbor_iterator;

  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL) {
    return ldIfNeeded().neighbor_begin(N,mflag);
  }

  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL) {
    return ldIfNeeded().neighbor_end(N,mflag);
  }

  bool has_neighbor(GraphNode N1, GraphNode N2, MethodFlag mflag = ALL) {
    return ldIfNeeded().has_neighbor(N1,N2, mflag);
  }

  typedef boost::counting_iterator<uint64_t> active_iterator;

  //! Iterate over nodes in graph (not thread safe)
  active_iterator active_begin() {
    return ldIfNeeded().active_begin();
  }

  active_iterator active_end() {
    return ldIfNeeded().active_end();
  }

  //! The number of nodes in the graph
  unsigned int size() {
    return ldIfNeeded().size();
  }

  //! The number of edges in the graph
  unsigned int sizeEdges () {
    return ldIfNeeded().sizeEdges();
  }

  NumaFileGraph();
  ~NumaFileGraph();

  //! Read graph connectivity information from file
  void structureFromFile(const char* filename) {
    graphs.get(0).structureFromFile(filename);
  }
};
#endif

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_FileGraph : public FileGraph {
  typedef FileGraph Par;

  struct gNode : public GaloisRuntime::Lockable {
    NodeTy data;
    gNode() {}
  };

  //null if type is void
  cache_line_storage<gNode>* NodeData;

public:
  LC_FileGraph() :NodeData(0) {}
  ~LC_FileGraph() {
    if (NodeData)
      delete[] NodeData;
      //numa_free(NodeData,sizeof(cache_line_storage<gNode>)*size());
  }
  
  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(&NodeData[N].data, mflag);
    return NodeData[N].data.data;
  }

  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    return Par::getEdgeData<EdgeTy>(src, dst, mflag);
  }

  //! Loads node data from file
  void nodeDataFromFile(const char* filename) {
    emptyNodeData();
    std::ifstream file(filename);
    for (uint64_t i = 0; i < size(); ++i)
      file >> NodeData[i];
  }
  
  //! Initializes node data for the graph to default values
  void emptyNodeData(NodeTy init = NodeTy()) {
    NodeData = new cache_line_storage<gNode>[size()];
    //NodeData = (cache_line_storage<gNode>*)numa_alloc_interleaved(sizeof(cache_line_storage<gNode>)*size());
    for (uint64_t i = 0; i < size(); ++i)
      NodeData[i].data.data = init;
  }

  void swap(LC_FileGraph& other) {
    std::swap(NodeData, other.NodeData);
    FileGraph::swap(other);
  }

  void clone(LC_FileGraph& other) {
    NodeData = other.NodeData;
    FileGraph::clone(other);
  }

  template<typename GTy>
  void copyGraph(GTy& graph) {
    structureFromGraph(graph);
    emptyNodeData();
    int i = 0;
    for (typename GTy::active_iterator ii = graph.active_begin(),
	   ee = graph.active_end(); ii != ee; ++ii, ++i)
      NodeData[i].data.data = graph.getData(*ii);
  }
};

//! Local computation graph (i.e., graph structure does not change)
template<typename EdgeTy>
class LC_FileGraph<void, EdgeTy> : public FileGraph {

  struct gNode : public GaloisRuntime::Lockable {
    gNode() {}
  };

  //null if type is void
  cache_line_storage<gNode>* NodeData;

public:
  LC_FileGraph() :NodeData(0) {}
  ~LC_FileGraph() {
    if (NodeData)
      delete[] NodeData;
  }
  
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    return FileGraph::getEdgeData<EdgeTy>(src, dst, mflag);
  }
};

template<typename NodeTy>
class LC_FileGraph<NodeTy, void>: public FileGraph { 

  struct gNode : public GaloisRuntime::Lockable {
    NodeTy data;
    gNode() {}
  };

  //null if type is void
  cache_line_storage<gNode>* NodeData;

public:
  LC_FileGraph() :NodeData(0) {}
  ~LC_FileGraph() {
    if (NodeData)
      delete[] NodeData;
  }
  
  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(&NodeData[N].data, mflag);
    return NodeData[N].data.data;
  }

  void nodeDataFromFile(const char* filename) {
    emptyNodeData();
    std::ifstream file(filename);
    for (uint64_t i = 0; i < numNodes; ++i)
      file >> NodeData[i];
  }
  
  void emptyNodeData(NodeTy init = NodeTy()) {
    NodeData = new cache_line_storage<gNode>[numNodes];
    for (uint64_t i = 0; i < numNodes; ++i)
      NodeData[i].data.data = init;
  }
};

template<>
class LC_FileGraph<void, void>: public FileGraph { 
};

}
}
#endif
