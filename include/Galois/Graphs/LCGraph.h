/** Local Computation graphs -*- C++ -*-
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
 * for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
 *   Graph::GraphNode src = *ii;
 *   for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src); jj != ej; ++jj) {
 *     Graph::GraphNode dst = g.getEdgeDst(jj);
 *     int edgeData = g.getEdgeData(jj);
 *     int nodeData = g.getData(dst);
 *   }
 * }
 * \endcode
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_GRAPHS_LCGRAPH_H
#define GALOIS_GRAPHS_LCGRAPH_H

#include "Galois/Galois.h"
#include "Galois/LargeArray.h"
#include "Galois/LazyObject.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Runtime/MethodFlags.h"
#include "Galois/Runtime/mm/Mem.h"

#include <iterator>
#include <boost/iterator/iterator_facade.hpp>
#include <new>

namespace Galois {
namespace Graph {

template<typename GraphTy>
void structureFromFile(GraphTy& g, const std::string& fname) {
  FileGraph graph;
  graph.structureFromFile(fname);
  g.structureFromGraph(graph);
}

template<typename EdgeContainerTy,typename CompTy>
struct EdgeSortCompWrapper {
  const CompTy& comp;

  EdgeSortCompWrapper(const CompTy& c): comp(c) { }
  bool operator()(const EdgeContainerTy& a, const EdgeContainerTy& b) const {
    return comp(a.get(), b.get());
  }
};

namespace HIDDEN {
uint64_t static localStart(uint64_t numNodes) {
  unsigned int id = Galois::Runtime::LL::getTID();
  unsigned int num = Galois::getActiveThreads();
  return (numNodes + num - 1) / num * id;
}

uint64_t static localEnd(uint64_t numNodes) {
  unsigned int id = Galois::Runtime::LL::getTID();
  unsigned int num = Galois::getActiveThreads();
  uint64_t end = (numNodes + num - 1) / num * (id + 1);
  return std::min(end, numNodes);
}
}

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_CSR_Graph: boost::noncopyable {
protected:
  typedef LargeArray<EdgeTy,true> EdgeData;
  typedef typename EdgeData::value_type edge_data_type;
  typedef LargeArray<uint32_t,true> EdgeDst;

  struct NodeInfo : public Galois::Runtime::Lockable {
    NodeTy data;
  };

  struct EdgeValue: public StrictObject<EdgeTy> {
    typedef StrictObject<EdgeTy> Super;
    typedef typename Super::value_type value_type;

    uint32_t dst;
    
    EdgeValue(uint32_t d, const value_type& v): Super(v), dst(d) { }

    template<typename ER>
    EdgeValue(const ER& ref) {
      ref.initialize(*this);
    }
  };

  //! Proxy object to facilitate sorting
  struct EdgeReference {
    uint64_t at;
    EdgeDst* edgeDst;
    EdgeData* edgeData;

    EdgeReference(uint64_t x, EdgeDst* dsts, EdgeData* data): at(x), edgeDst(dsts), edgeData(data) { }

    EdgeReference operator=(const EdgeValue& x) {
      edgeDst->at(at) = x.dst;
      edgeData->set(at, x.get());
      return *this;
    }

    EdgeReference operator=(const EdgeReference& x) {
      edgeDst->at(at) = edgeDst->at(x.at);
      edgeData->set(at, edgeData->at(x.at));
      return *this;
    }

    EdgeValue operator*() const {
      return EdgeValue(edgeDst->at(at), edgeData->at(at));
    }

    void initialize(EdgeValue& value) const {
      value = *(*this);
    }
  };

  //! Iterator to facilitate sorting
  class EdgeSortIterator: public boost::iterator_facade<
                          EdgeSortIterator,
                          EdgeValue,
                          boost::random_access_traversal_tag,
                          EdgeReference
                          > {
    uint64_t at;
    EdgeDst* edgeDst;
    EdgeData* edgeData;
  public:
    EdgeSortIterator(): at(~0) { }
    EdgeSortIterator(uint64_t x, EdgeDst* dsts, EdgeData* data):
      at(x), edgeDst(dsts), edgeData(data) { }
  private:
    friend class boost::iterator_core_access;
    
    bool equal(const EdgeSortIterator& other) const { return at == other.at; }
    EdgeReference dereference() const { return EdgeReference(at, edgeDst, edgeData); }
    ptrdiff_t distance_to(const EdgeSortIterator& other) const { return other.at - (ptrdiff_t) at; }
    void increment() { ++at; }
    void decrement() { --at; }
    void advance(ptrdiff_t n) { at += n; }
  };

  LargeArray<NodeInfo,false> nodeData;
  LargeArray<uint64_t,true> edgeIndData;
  EdgeDst edgeDst;
  EdgeData edgeData;

  uint64_t numNodes;
  uint64_t numEdges;

  uint64_t raw_neighbor_begin(uint32_t N) const {
    return (N == 0) ? 0 : edgeIndData[N-1];
  }

  uint64_t raw_neighbor_end(uint32_t N) const {
    return edgeIndData[N];
  }

  uint64_t getEdgeIdx(uint32_t src, uint32_t dst) {
    for (uint64_t ii = raw_neighbor_begin(src),
	   ee = raw_neighbor_end(src); ii != ee; ++ii)
      if (edgeDst[ii] == dst)
        return ii;
    return ~static_cast<uint64_t>(0);
  }

  EdgeSortIterator edge_sort_begin(uint32_t src) {
    return EdgeSortIterator(raw_neighbor_begin(src), &edgeDst, &edgeData);
  }

  EdgeSortIterator edge_sort_end(uint32_t src) {
    return EdgeSortIterator(raw_neighbor_end(src), &edgeDst, &edgeData);
  }
  
public:
  typedef uint32_t GraphNode;
  typedef typename EdgeData::reference edge_data_reference;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef boost::counting_iterator<uint32_t> iterator;
  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;

  ~LC_CSR_Graph() {
    edgeData.destroy();
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    Galois::Runtime::acquire(&NI, mflag);
    return NI.data;
  }

  bool hasNeighbor(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
    return getEdgeIdx(src, dst) != ~static_cast<uint64_t>(0);
  }

//  edge_data_reference getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
//    Galois::Runtime::checkWrite(mflag);
//    Galois::Runtime::acquire(&nodeData[src], mflag);
//    return EdgeData.get(getEdgeIdx(src, dst));
//  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) {
    Galois::Runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return edgeDst[*ni];
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  local_iterator local_begin() const { return iterator(HIDDEN::localStart(numNodes)); }
  local_iterator local_end() const { return iterator(HIDDEN::localEnd(numNodes)); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&nodeData[N], mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (uint64_t ii = raw_neighbor_begin(N), ee = raw_neighbor_end(N); ii != ee; ++ii) {
        Galois::Runtime::acquire(&nodeData[edgeDst[ii]], mflag);
      }
    }
    return edge_iterator(raw_neighbor_begin(N));
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    NodeInfo& NI = nodeData[N];
    Galois::Runtime::acquire(&NI, mflag);
    return edge_iterator(raw_neighbor_end(N));
  }

  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&nodeData[N], mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), EdgeSortCompWrapper<EdgeValue,CompTy>(comp));
  }

  void structureFromFile(const std::string& fname) { Galois::Graph::structureFromFile(*this, fname); }

  void structureFromGraph(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    nodeData.allocate(numNodes);
    edgeIndData.allocate(numNodes);
    edgeDst.allocate(numEdges);
    edgeData.allocate(numEdges);

    if (EdgeData::has_value)
      edgeData.copyIn(graph.edge_data_begin<edge_data_type>(), graph.edge_data_end<edge_data_type>());
    std::copy(graph.edge_id_begin(), graph.edge_id_end(), edgeIndData.data());
    std::copy(graph.node_id_begin(), graph.node_id_end(), edgeDst.data());
  }
};

template<typename NodeInfoTy,typename EdgeTy>
struct EdgeInfoBase: public LazyObject<EdgeTy> {
  typedef LazyObject<EdgeTy> Super;
  typedef typename Super::reference reference;
  typedef typename Super::value_type value_type;
  const static bool has_value = Super::has_value;

  NodeInfoTy* dst;
};

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_CSRInline_Graph: boost::noncopyable {
protected:
  struct NodeInfo;
  typedef EdgeInfoBase<NodeInfo, EdgeTy> EdgeInfo;
  typedef typename EdgeInfo::value_type edge_data_type;
  
  struct NodeInfo : public Galois::Runtime::Lockable {
    NodeTy data;
    EdgeInfo* edgeBegin;
    EdgeInfo* edgeEnd;
  };

  LargeArray<NodeInfo,false> nodeData;
  LargeArray<EdgeInfo,true> edgeData;
  uint64_t numNodes;
  uint64_t numEdges;
  NodeInfo* endNode;

  uint64_t getEdgeIdx(uint64_t src, uint64_t dst) {
    NodeInfo& NI = nodeData[src];
    for (uint64_t x = NI.edgeBegin; x < NI.edgeEnd; ++x)
      if (edgeData[x].dst == dst)
        return x;
    return ~static_cast<uint64_t>(0);
  }

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeInfo* edge_iterator;
  typedef typename EdgeInfo::reference edge_data_reference;

  class iterator : std::iterator<std::random_access_iterator_tag, GraphNode> {
    NodeInfo* at;

  public:
    iterator(NodeInfo* a) :at(a) {}
    iterator(const iterator& m) :at(m.at) {}
    iterator& operator++() { ++at; return *this; }
    iterator operator++(int) { iterator tmp(*this); ++at; return tmp; }
    iterator& operator--() { --at; return *this; }
    iterator operator--(int) { iterator tmp(*this); --at; return tmp; }
    bool operator==(const iterator& rhs) { return at == rhs.at; }
    bool operator!=(const iterator& rhs) { return at != rhs.at; }
    GraphNode operator*() { return at; }
  };

  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;

  ~LC_CSRInline_Graph() {
    if (!EdgeInfo::has_value) return;
    if (nodeData.data() == endNode) return;

    for (EdgeInfo* ii = nodeData[0]->edgeBegin, ei = endNode->edgeEnd; ii != ei; ++ii) {
      ii->destroy();
    }
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    Galois::Runtime::acquire(N, mflag);
    return N->data;
  }
  
//  edge_data_reference getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
//    Galois::Runtime::checkWrite(mflag);
//    Galois::Runtime::acquire(src, mflag);
//    return EdgeData[getEdgeIdx(src,dst)].getData();
//  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) const {
    Galois::Runtime::checkWrite(mflag, false);
    return ni->get();
   }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(nodeData.data()); }
  iterator end() const { return iterator(endNode); }

  local_iterator local_begin() const { return iterator(&nodeData[HIDDEN::localStart(numNodes)]); }
  local_iterator local_end() const { return iterator(&nodeData[HIDDEN::localEnd(numNodes)]); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin, ee = N->edgeEnd; ii != ee; ++ii) {
        Galois::Runtime::acquire(ii->dst, mflag);
      }
    }
    return N->edgeBegin;
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    return N->edgeEnd;
  }

  void structureFromFile(const std::string& fname) { Galois::Graph::structureFromFile(*this, fname); }

  void structureFromGraph(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    nodeData.allocate(numNodes);
    edgeData.allocate(numEdges);

    std::vector<NodeInfo*> node_ids;
    node_ids.resize(numNodes);
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
      NodeInfo* curNode = &nodeData[*ii];
      node_ids[*ii] = curNode;
    }
    endNode = &nodeData[numNodes];

    //layout the edges
    EdgeInfo* curEdge = edgeData.data();
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
      node_ids[*ii]->edgeBegin = curEdge;
      for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii), 
          ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
        if (EdgeInfo::has_value)
          curEdge->construct(graph.getEdgeData<edge_data_type>(ni));
        curEdge->dst = node_ids[*ni];
        ++curEdge;
      }
      node_ids[*ii]->edgeEnd = curEdge;
    }
  }
};

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_Linear_Graph: boost::noncopyable {
protected:
  struct NodeInfo;
  typedef EdgeInfoBase<NodeInfo,EdgeTy> EdgeInfo;
  typedef typename EdgeInfo::value_type edge_data_type;
  typedef LargeArray<NodeInfo*,true> Nodes;

  struct NodeInfo : public Galois::Runtime::Lockable {
    NodeTy data;
    int numEdges;

    EdgeInfo* edgeBegin() {
      NodeInfo* n = this;
      ++n; //start of edges
      return reinterpret_cast<EdgeInfo*>(n);
    }

    EdgeInfo* edgeEnd() {
      EdgeInfo* ei = edgeBegin();
      ei += numEdges;
      return ei;
    }

    NodeInfo* next() {
      NodeInfo* ni = this;
      EdgeInfo* ei = edgeEnd();
      while (reinterpret_cast<char*>(ni) < reinterpret_cast<char*>(ei))
        ++ni;
      return ni;
    }
  };
 
  LargeArray<char,true> data;
  uint64_t numNodes;
  uint64_t numEdges;
  Nodes nodes;

  EdgeInfo* getEdgeIdx(NodeInfo* src, NodeInfo* dst) {
    EdgeInfo* eb = src->edgeBegin();
    EdgeInfo* ee = src->edgeEnd();
    for (; eb != ee; ++eb)
      if (eb->dst == dst)
        return eb;
    return 0;
  }

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeInfo* edge_iterator;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef NodeInfo** iterator;
  typedef NodeInfo*const * const_iterator;
  typedef iterator local_iterator;
  typedef const_iterator const_local_iterator;

  LC_Linear_Graph() { }

  ~LC_Linear_Graph() { 
    for (typename Nodes::iterator ii = nodes.begin(), ei = nodes.end(); ii != ei; ++ii) {
      NodeInfo* n = *ii;
      EdgeInfo* edgeBegin = n->edgeBegin();
      EdgeInfo* edgeEnd = n->edgeEnd();

      (&n->data)->~NodeTy();
      if (EdgeInfo::has_value) {
        while (edgeBegin != edgeEnd) {
          edgeBegin->destroy();
          ++edgeBegin;
        }
      }
    }
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    Galois::Runtime::acquire(N, mflag);
    return N->data;
  }
  
//  edge_data_reference getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
//    Galois::Runtime::checkWrite(mflag);
//    Galois::Runtime::acquire(src, mflag);
//    return getEdgeIdx(src,dst)->getData();
//  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) const {
    Galois::Runtime::checkWrite(mflag, false);
    return ni->get();
  }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }
  iterator begin() { return &nodes[0]; }
  iterator end() { return &nodes[numNodes]; }
  const_iterator begin() const { return &nodes[0]; }
  const_iterator end() const { return &nodes[numNodes]; }
  local_iterator local_begin() { return &nodes[HIDDEN::localStart(numNodes)]; }
  local_iterator local_end() { return &nodes[HIDDEN::localEnd(numNodes)]; }
  const_local_iterator local_begin() const { return &nodes[HIDDEN::localStart(numNodes)]; }
  const_local_iterator local_end() const { return &nodes[HIDDEN::localEnd(numNodes)]; }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee; ++ii) {
        Galois::Runtime::acquire(ii->dst, mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    return N->edgeEnd();
  }

  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(N->edgeBegin(), N->edgeEnd(), EdgeSortCompWrapper<EdgeInfo,CompTy>(comp));
  }

  void structureFromFile(const std::string& fname) { Galois::Graph::structureFromFile(*this, fname); }

  void structureFromGraph(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    data.allocate(sizeof(NodeInfo) * numNodes * 2 + sizeof(EdgeInfo) * numEdges);
    nodes.allocate(numNodes);
    NodeInfo* curNode = reinterpret_cast<NodeInfo*>(data.data());
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
      new (&curNode->data) NodeTy; //inplace new
      curNode->numEdges = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
      nodes[*ii] = curNode;
      curNode = curNode->next();
    }

    //layout the edges
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
      EdgeInfo* edge = nodes[*ii]->edgeBegin();
      for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
          ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
        if (EdgeInfo::has_value)
          edge->construct(graph.getEdgeData<edge_data_type>(ni));
        edge->dst = nodes[*ni];
	      ++edge;
      }
    }
  }
};

//! Local computation graph (i.e., graph structure does not change)
//! Specialization of LC_Linear_Graph for NUMA architectures
template<typename NodeTy, typename EdgeTy>
class LC_Numa_Graph: boost::noncopyable {
protected:
  struct NodeInfo;
  typedef EdgeInfoBase<NodeInfo,EdgeTy> EdgeInfo;
  typedef typename EdgeInfo::value_type edge_data_type;
  typedef LargeArray<NodeInfo*,true> Nodes;

  struct NodeInfo : public Galois::Runtime::Lockable {
    NodeTy data;
    int numEdges;

    EdgeInfo* edgeBegin() {
      NodeInfo* n = this;
      ++n; //start of edges
      return reinterpret_cast<EdgeInfo*>(n);
    }

    EdgeInfo* edgeEnd() {
      EdgeInfo* ei = edgeBegin();
      ei += numEdges;
      return ei;
    }

    NodeInfo* next() {
      NodeInfo* ni = this;
      EdgeInfo* ei = edgeEnd();
      while (reinterpret_cast<char*>(ni) < reinterpret_cast<char*>(ei))
        ++ni;
      return ni;
    }
  };

  struct Header {
    NodeInfo* begin;
    NodeInfo* end;
    size_t size;
  };

  Galois::Runtime::PerThreadStorage<Header*> headers;
  Nodes nodes;
  uint64_t numNodes;
  uint64_t numEdges;

  EdgeInfo* getEdgeIdx(NodeInfo* src, NodeInfo* dst) {
    EdgeInfo* eb = src->edgeBegin();
    EdgeInfo* ee = src->edgeEnd();
    for (; eb != ee; ++eb)
      if (eb->dst == dst)
        return eb;
    return 0;
  }

  struct DistributeInfo {
    uint64_t numNodes;
    uint64_t numEdges;
    FileGraph::iterator begin;
    FileGraph::iterator end;
  };

  //! Divide graph into equal sized chunks
  void distribute(FileGraph& graph, Galois::Runtime::PerThreadStorage<DistributeInfo>& dinfo) {
    size_t total = sizeof(NodeInfo) * numNodes + sizeof(EdgeInfo) * numEdges;
    unsigned int num = Galois::getActiveThreads();
    size_t blockSize = total / num;
    size_t curSize = 0;
    FileGraph::iterator ii = graph.begin();
    FileGraph::iterator ei = graph.end();
    FileGraph::iterator last = ii;
    uint64_t nnodes = 0;
    uint64_t nedges = 0;
    uint64_t runningNodes = 0;
    uint64_t runningEdges = 0;

    unsigned int tid;
    for (tid = 0; tid + 1 < num; ++tid) {
      for (; ii != ei; ++ii) {
        if (curSize >= (tid + 1) * blockSize) {
          DistributeInfo& d = *dinfo.getRemote(tid);
          d.numNodes = nnodes;
          d.numEdges = nedges;
          d.begin = last;
          d.end = ii;

          runningNodes += nnodes;
          runningEdges += nedges;
          nnodes = nedges = 0;
          last = ii;
          break;
        }
        size_t nneighbors = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
        nedges += nneighbors;
        nnodes += 1;
        curSize += sizeof(NodeInfo) + sizeof(EdgeInfo) * nneighbors;
      }
    }

    DistributeInfo& d = *dinfo.getRemote(tid);
    d.numNodes = numNodes - runningNodes;
    d.numEdges = numEdges - runningEdges;
    d.begin = last;
    d.end = ei;
  }

  struct AllocateNodes {
    Galois::Runtime::PerThreadStorage<DistributeInfo>& dinfo;
    Galois::Runtime::PerThreadStorage<Header*>& headers;
    NodeInfo** nodes;
    FileGraph& graph;

    AllocateNodes(
        Galois::Runtime::PerThreadStorage<DistributeInfo>& d,
        Galois::Runtime::PerThreadStorage<Header*>& h, NodeInfo** n, FileGraph& g):
      dinfo(d), headers(h), nodes(n), graph(g) { }

    void operator()(unsigned int tid, unsigned int num) {
      //DistributeInfo& d = dinfo.get(tid);
      DistributeInfo& d = *dinfo.getLocal();

      // extra 2 factors are for alignment purposes
      size_t size =
          sizeof(Header) * 2 +
          sizeof(NodeInfo) * d.numNodes * 2 +
          sizeof(EdgeInfo) * d.numEdges;

      void *raw = Galois::Runtime::MM::largeAlloc(size);
      memset(raw, 0, size);

      Header*& h = *headers.getLocal();
      h = reinterpret_cast<Header*>(raw);
      h->size = size;
      h->begin = h->end = reinterpret_cast<NodeInfo*>(h + 1);

      if (!d.numNodes)
        return;

      for (FileGraph::iterator ii = d.begin, ee = d.end; ii != ee; ++ii) {
        new (&h->end->data) NodeTy; //inplace new
        h->end->numEdges = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
        nodes[*ii] = h->end;
        h->end = h->end->next();
      }
    }
  };

  struct AllocateEdges {
    Galois::Runtime::PerThreadStorage<DistributeInfo>& dinfo;
    NodeInfo** nodes;
    FileGraph& graph;

    AllocateEdges(Galois::Runtime::PerThreadStorage<DistributeInfo>& d, NodeInfo** n, FileGraph& g):
      dinfo(d), nodes(n), graph(g) { }

    //! layout the edges
    void operator()(unsigned int tid, unsigned int num) {
      //DistributeInfo& d = *dinfo.getRemote(tid);
      DistributeInfo& d = *dinfo.getLocal();
      if (!d.numNodes)
        return;

      for (FileGraph::iterator ii = d.begin, ee = d.end; ii != ee; ++ii) {
        EdgeInfo* edge = nodes[*ii]->edgeBegin();
        for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
               ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
          if (EdgeInfo::has_value)
            edge->construct(graph.getEdgeData<edge_data_type>(ni));
          edge->dst = nodes[*ni];
          ++edge;
        }
      }
    }
  };

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeInfo* edge_iterator;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef NodeInfo** iterator;
  typedef NodeInfo*const * const_iterator;

  class local_iterator : public std::iterator<std::forward_iterator_tag, GraphNode> {
    const Galois::Runtime::PerThreadStorage<Header*>* headers;
    unsigned int tid;
    Header* p;
    GraphNode v;

    bool init_thread() {
      p = tid < headers->size() ? *headers->getRemote(tid) : 0;
      v = p ? p->begin : 0;
      return p;
    }

    bool advance_local() {
      if (p) {
        v = v->next();
        return v != p->end;
      }
      return false;
    }

    void advance_thread() {
      while (tid < headers->size()) {
        ++tid;
        if (init_thread())
          return;
      }
    }

    void advance() {
      if (advance_local()) return;
      advance_thread();
    }

  public:
    local_iterator(): headers(0), tid(0), p(0), v(0) { }
    local_iterator(const Galois::Runtime::PerThreadStorage<Header*>* _headers, int _tid):
      headers(_headers), tid(_tid), p(0), v(0)
    {
      //find first valid item
      if (!init_thread())
        advance_thread();
    }

    //local_iterator(const iterator& it): headers(it.headers), tid(it.tid), p(it.p), v(it.v) { }
    local_iterator& operator++() { advance(); return *this; }
    local_iterator operator++(int) { local_iterator tmp(*this); operator++(); return tmp; }
    bool operator==(const local_iterator& rhs) const {
      return (headers == rhs.headers && tid == rhs.tid && p == rhs.p && v == rhs.v);
    }
    bool operator!=(const local_iterator& rhs) const {
      return !(headers == rhs.headers && tid == rhs.tid && p == rhs.p && v == rhs.v);
    }
    GraphNode operator*() const { return v; }
  };

  typedef local_iterator const_local_iterator;

  ~LC_Numa_Graph() {
    for (typename Nodes::iterator ii = nodes.begin(), ei = nodes.end(); ii != ei; ++ii) {
      NodeInfo* n = *ii;
      EdgeInfo* edgeBegin = n->edgeBegin();
      EdgeInfo* edgeEnd = n->edgeEnd();

      (&n->data)->~NodeTy();
      if (EdgeInfo::has_value) {
        while (edgeBegin != edgeEnd) {
          edgeBegin->destroy();
          ++edgeBegin;
        }
      }
    }

    for (unsigned i = 0; i < headers.size(); ++i) {
      Header* h = *headers.getRemote(i);
      if (h)
        Galois::Runtime::MM::largeFree(h, h->size);
    }
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    Galois::Runtime::acquire(N, mflag);
    return N->data;
  }
  
//  edge_data_reference getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
//    Galois::Runtime::checkWrite(mflag);
//    Galois::Runtime::acquire(src, mflag);
//    return getEdgeIdx(src,dst)->getData();
//  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) const {
    Galois::Runtime::checkWrite(mflag, false);
    return ni->get();
  }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }
  iterator begin() { return &nodes[0]; }
  iterator end() { return &nodes[numNodes]; }
  const_iterator begin() const { return &nodes[0]; }
  const_iterator end() const { return &nodes[numNodes]; }

  local_iterator local_begin() const {
    return local_iterator(&headers, Galois::Runtime::LL::getTID());
  }

  local_iterator local_end() const {
    return local_iterator(&headers, Galois::Runtime::LL::getTID() + 1);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee; ++ii) {
        Galois::Runtime::acquire(ii->dst, mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    return N->edgeEnd();
  }

  void structureFromFile(const std::string& fname) { Galois::Graph::structureFromFile(*this, fname); }

  void structureFromGraph(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();

    Galois::Runtime::PerThreadStorage<DistributeInfo> dinfo;
    distribute(graph, dinfo);

    nodes.allocate(numNodes);

    Galois::on_each(AllocateNodes(dinfo, headers, nodes.data(), graph));
    Galois::on_each(AllocateEdges(dinfo, nodes.data(), graph));
  }
};


}
}
#endif
