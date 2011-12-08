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
#include <endian.h>
#include <map>
#include <vector>
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
      if (le32toh(*ii) == dst)
	return std::distance(outs, ii);
    return ~(uint64_t)0;
  }

  uint32_t* raw_neighbor_begin(GraphNode N, MethodFlag mflag = ALL) const {
    return (N == 0) ? &outs[0] : &outs[le64toh(outIdx[N-1])];
  }

  uint32_t* raw_neighbor_end(GraphNode N, MethodFlag mflag = ALL) const {
    return &outs[le64toh(outIdx[N])];
  }

  struct Convert32: public std::unary_function<uint32_t, uint32_t> {
    uint32_t operator()(uint32_t x) const {
      return le32toh(x);
    }
  };
  
  struct Convert64: public std::unary_function<uint64_t,uint64_t> {
    uint64_t operator()(uint64_t x) const {
      return le64toh(x);
    }
  };

  //! Initialize graph from block of memory
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
    return reinterpret_cast<EdgeTy*>(edgeData)[getEdgeIdx(src, dst)];
  }

  size_t neighborsSize(GraphNode N, Galois::MethodFlag mflag = ALL) const {
    return std::distance(neighbor_begin(N, mflag), neighbor_end(N, mflag));
  }

  // Iterators
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = ALL) const {
    return edge_iterator(N == 0 ? 0 : le64toh(outIdx[N-1]));
  }
  edge_iterator edge_end(GraphNode N, MethodFlag mflag = ALL) const {
    return edge_iterator(le64toh(outIdx[N]));
  }
  template<typename EdgeTy> EdgeTy& getEdgeData(edge_iterator it, MethodFlag mflag = ALL) const {
    return reinterpret_cast<EdgeTy*>(edgeData)[*it];
  }
  GraphNode getEdgeDst(edge_iterator it, MethodFlag mflag = ALL) const {
    return le32toh(outs[*it]);
  }
#ifdef BOOST_LITTLE_ENDIAN
  typedef uint32_t* neighbor_iterator;
  typedef uint32_t* nodeid_iterator;
  typedef uint64_t* edgeid_iterator;

  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL) const {
    return raw_neighbor_begin(N, mflag);
  }
  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL) const {
    return raw_neighbor_end(N, mflag);
  }
  nodeid_iterator nodeid_begin() const {
    return &outs[0];
  }
  nodeid_iterator nodeid_end() const {
    return &outs[numEdges];
  }
  edgeid_iterator edgeid_begin() const {
    return &outIdx[0];
  }
  edgeid_iterator edgeid_end() const {
    return &outIdx[numNodes];
  }
  template<typename EdgeTy>
  EdgeTy& getEdgeData(neighbor_iterator it, MethodFlag mflag = ALL) {
    return reinterpret_cast<EdgeTy*>(edgeData)[std::distance(outs, it)];
  }
#else
  typedef boost::transform_iterator<Convert32, uint32_t*> neighbor_iterator;
  typedef boost::transform_iterator<Convert32, uint32_t*> nodeid_iterator;
  typedef boost::transform_iterator<Convert64, uint64_t*> edgeid_iterator;
  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL) const {
    return boost::make_transform_iterator(raw_neighbor_begin(N, mflag), Convert32());
  }
  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL) const {
    return boost::make_transform_iterator(raw_neighbor_end(N, mflag), Convert32());
  }
  nodeid_iterator nodeid_begin() const {
    return boost::make_transform_iterator(&outs[0], Convert32());
  }
  nodeid_iterator nodeid_end() const {
    return boost::make_transform_iterator(&outs[numEdges], Convert32());
  }
  edgeid_iterator edgeid_begin() const {
    return boost::make_transform_iterator(&outIdx[0], Convert64());
  }
  edgeid_iterator edgeid_end() const {
    return boost::make_transform_iterator(&outIdx[numNodes], Convert64());
  }
  template<typename EdgeTy>
  EdgeTy& getEdgeData(neighbor_iterator it, MethodFlag mflag = ALL) {
    return reinterpret_cast<EdgeTy*>(edgeData)[std::distance(outs, it.base())];
  }
#endif

  template<typename EdgeTy> EdgeTy* edgedata_begin() const {
    return reinterpret_cast<EdgeTy*>(edgeData);
  }
  template<typename EdgeTy> EdgeTy* edgedata_end() const {
    assert(sizeof(EdgeTy) == sizeEdgeTy);
    EdgeTy* r = reinterpret_cast<EdgeTy*>(edgeData);
    return &r[numEdges];
  }

  bool hasNeighbor(GraphNode N1, GraphNode N2, MethodFlag mflag = ALL) const {
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
  unsigned int sizeEdges() const {
    return numEdges;
  }

  FileGraph();
  ~FileGraph();

  //! Read graph connectivity information from file
  void structureFromFile(const std::string& filename);

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

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_FileGraph : public FileGraph {

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
    return FileGraph::getEdgeData<EdgeTy>(src, dst, mflag);
  }

  EdgeTy& getEdgeData(FileGraph::edge_iterator it, MethodFlag mflag = ALL) {
    return FileGraph::getEdgeData<EdgeTy>(it, mflag);
  }
  EdgeTy& getEdgeData(FileGraph::neighbor_iterator it, MethodFlag mflag = ALL) {
    return FileGraph::getEdgeData<EdgeTy>(it, mflag);
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
  EdgeTy& getEdgeData(FileGraph::edge_iterator it, MethodFlag mflag = ALL) {
    return FileGraph::getEdgeData<EdgeTy>(it, mflag);
  }
  EdgeTy& getEdgeData(FileGraph::neighbor_iterator it, MethodFlag mflag = ALL) {
    return FileGraph::getEdgeData<EdgeTy>(it, mflag);
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
