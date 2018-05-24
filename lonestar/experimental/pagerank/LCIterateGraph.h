#ifndef GALOIS_GRAPHS_LCITERATEGRAPH_H
#define GALOIS_GRAPHS_LCITERATEGRAPH_H

#include "galois/Galois.h"
#include "galois/Graph/FileGraph.h"
#include "galois/runtime/mm/Mem.h"

#include <iterator>
#include <new>

namespace galois {
namespace graphs {

//! Small wrapper to have value void specialization
template<typename ETy>
struct EdgeDataWrapper {
  typedef ETy& reference;
  ETy* data;
  uint64_t numEdges;
  
  reference get(ptrdiff_t x) const { return data[x]; }
  EdgeDataWrapper(): data(0) { }
  ~EdgeDataWrapper() {
    if (data)
      Galoisruntime::MM::largeInterleavedFree(data, sizeof(ETy) * numEdges);
  }
  void readIn(FileGraph& g) {
    numEdges = g.sizeEdges();
    data = reinterpret_cast<ETy*>(Galoisruntime::MM::largeInterleavedAlloc(sizeof(ETy) * numEdges));
    std::copy(g.edgedata_begin<ETy>(), g.edgedata_end<ETy>(), &data[0]);
  }
};

template<>
struct EdgeDataWrapper<void> {
  typedef bool reference;
  reference get(ptrdiff_t x) const { return false; }
  void readIn(FileGraph& g) { }
};

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_CSR_Graph {
protected:
  struct NodeInfo : public Galoisruntime::Lockable {
    NodeTy data;
  };

  NodeInfo* NodeData;
  uint64_t* EdgeIndData;
  uint32_t* EdgeDst;
  EdgeDataWrapper<EdgeTy> EdgeData;

  uint64_t numNodes;
  uint64_t numEdges;

  uint64_t raw_neighbor_begin(uint32_t N) const {
    return (N == 0) ? 0 : EdgeIndData[N-1];
  }

  uint64_t raw_neighbor_end(uint32_t N) const {
    return EdgeIndData[N];
  }

  uint64_t getEdgeIdx(uint32_t src, uint32_t dst) {
    for (uint64_t ii = raw_neighbor_begin(src),
	   ee = raw_neighbor_end(src); ii != ee; ++ii)
      if (EdgeDst[ii] == dst)
	return ii;
    return ~static_cast<uint64_t>(0);
  }

public:
  typedef uint32_t GraphNode;
  typedef typename EdgeDataWrapper<EdgeTy>::reference edge_data_reference;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef boost::counting_iterator<uint32_t> iterator;

  LC_CSR_Graph(): NodeData(0), EdgeIndData(0), EdgeDst(0) { }

  ~LC_CSR_Graph() {
    // TODO(ddn): call destructors of user data
    if (EdgeDst)
      Galoisruntime::MM::largeInterleavedFree(EdgeDst, sizeof(uint32_t) * numEdges);
    if (EdgeIndData)
      Galoisruntime::MM::largeInterleavedFree(EdgeIndData, sizeof(uint64_t) * numNodes);
    if (NodeData)
      Galoisruntime::MM::largeInterleavedFree(NodeData, sizeof(NodeInfo) * numNodes);
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    // Galoisruntime::checkWrite(mflag);
    NodeInfo& NI = NodeData[N];
    Galoisruntime::acquire(&NI, mflag);
    return NI.data;
  }

  bool hasNeighbor(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    return getEdgeIdx(src, dst) != ~static_cast<uint64_t>(0);
  }

  edge_data_reference getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    // Galoisruntime::checkWrite(mflag);
    Galoisruntime::acquire(&NodeData[src], mflag);
    return EdgeData.get(getEdgeIdx(src, dst));
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // Galoisruntime::checkWrite(mflag);
    return EdgeData.get(*ni);
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return EdgeDst[*ni];
  }

  uint64_t size() const {
    return numNodes;
  }

  uint64_t sizeEdges() const {
    return numEdges;
  }

  iterator begin() const {
    return iterator(0);
  }

  iterator end() const {
    return iterator(numNodes);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = ALL) {
    Galoisruntime::acquire(&NodeData[N], mflag);
    if (Galoisruntime::shouldLock(mflag)) {
      for (uint64_t ii = raw_neighbor_begin(N), ee = raw_neighbor_end(N);
	   ii != ee; ++ii) {
	Galoisruntime::acquire(&NodeData[EdgeDst[ii]], mflag);
      }
    }
    return edge_iterator(raw_neighbor_begin(N));
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = ALL) {
    NodeInfo& NI = NodeData[N];
    Galoisruntime::acquire(&NI, mflag);
    return edge_iterator(raw_neighbor_end(N));
  }

  void fromFile(const std::string& fname) {
    FileGraph graph;
    graph.fromFile(fname);
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    NodeData = reinterpret_cast<NodeInfo*>(Galoisruntime::MM::largeInterleavedAlloc(sizeof(NodeInfo) * numNodes));
    EdgeIndData = reinterpret_cast<uint64_t*>(Galoisruntime::MM::largeInterleavedAlloc(sizeof(uint64_t) * numNodes));
    EdgeDst = reinterpret_cast<uint32_t*>(Galoisruntime::MM::largeInterleavedAlloc(sizeof(uint32_t) * numEdges));
    EdgeData.readIn(graph);
    std::copy(graph.edgeid_begin(), graph.edgeid_end(), &EdgeIndData[0]);
    std::copy(graph.nodeid_begin(), graph.nodeid_end(), &EdgeDst[0]);

    for (unsigned x = 0; x < numNodes; ++x)
      new (&NodeData[x]) NodeTy; // inplace new
  }
};

/**
 * Wrapper class to have a valid type on void edges
 */
template<typename NITy, typename EdgeTy>
struct EdgeInfoWrapper {
  typedef EdgeTy& reference;
  EdgeTy data;
  NITy* dst;

  void allocateEdgeData(FileGraph& g, FileGraph::neighbor_iterator& ni) {
    new (&data) EdgeTy(g.getEdgeData<EdgeTy>(ni));
  }

  reference getData() {
    return data;
  }
};

template<typename NITy>
struct EdgeInfoWrapper<NITy,void> {
  typedef void reference;
  NITy* dst;
  void allocateEdgeData(FileGraph& g, FileGraph::neighbor_iterator& ni) { }
  reference getData() { }
};

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_CSRInline_Graph {
protected:
  struct NodeInfo;
  typedef EdgeInfoWrapper<NodeInfo, EdgeTy> EdgeInfo;
  
  struct NodeInfo : public Galoisruntime::Lockable {
    NodeTy data;
    EdgeInfo* edgebegin;
    EdgeInfo* edgeend;
  };

  NodeInfo* NodeData;
  EdgeInfo* EdgeData;
  uint64_t numNodes;
  uint64_t numEdges;
  NodeInfo* endNode;

  uint64_t getEdgeIdx(uint64_t src, uint64_t dst) {
    NodeInfo& NI = NodeData[src];
    for (uint64_t x = NI.edgebegin; x < NI.edgeend; ++x)
      if (EdgeData[x].dst == dst)
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

  LC_CSRInline_Graph(): NodeData(0), EdgeData(0) { }

  ~LC_CSRInline_Graph() {
    // TODO(ddn): call destructors of user data
    if (EdgeData)
      Galoisruntime::MM::largeInterleavedFree(EdgeData, sizeof(*EdgeData) * numEdges);
    if (NodeData)
      Galoisruntime::MM::largeInterleavedFree(NodeData, sizeof(*NodeData) * numEdges);
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    // Galoisruntime::checkWrite(mflag);
    Galoisruntime::acquire(N, mflag);
    return N->data;
  }
  
  edge_data_reference getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    // Galoisruntime::checkWrite(mflag);
    Galoisruntime::acquire(src, mflag);
    return EdgeData[getEdgeIdx(src,dst)].getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) const {
    // Galoisruntime::checkWrite(mflag);
    return ni->getData();
   }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const {
    return numNodes;
  }

  uint64_t sizeEdges() const {
    return numEdges;
  }

  iterator begin() const {
    return iterator(&NodeData[0]);
  }

  iterator end() const {
    return iterator(endNode);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = ALL) {
    Galoisruntime::acquire(N, mflag);
    if (Galoisruntime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgebegin, ee = N->edgeend;
	   ii != ee; ++ii) {
	Galoisruntime::acquire(ii->dst, mflag);
      }
    }
    return N->edgebegin;
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = ALL) {
    Galoisruntime::acquire(N, mflag);
    return N->edgeend;
  }

  void fromFile(const std::string& fname) {
    FileGraph graph;
    graph.fromFile(fname);
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    NodeData = reinterpret_cast<NodeInfo*>(Galoisruntime::MM::largeInterleavedAlloc(sizeof(*NodeData) * numNodes));
    EdgeData = reinterpret_cast<EdgeInfo*>(Galoisruntime::MM::largeInterleavedAlloc(sizeof(*EdgeData) * numEdges));
    std::vector<NodeInfo*> node_ids;
    node_ids.resize(numNodes);
    for (FileGraph::iterator ii = graph.begin(),
	   ee = graph.end(); ii != ee; ++ii) {
      NodeInfo* curNode = &NodeData[*ii];
      new (&curNode->data) NodeTy; //inplace new
      node_ids[*ii] = curNode;
    }
    endNode = &NodeData[numNodes];

    //layout the edges
    EdgeInfo* curEdge = &EdgeData[0];
    for (FileGraph::iterator ii = graph.begin(),
	   ee = graph.end(); ii != ee; ++ii) {
      node_ids[*ii]->edgebegin = curEdge;
      for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
	     ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
        curEdge->allocateEdgeData(graph, ni);
	curEdge->dst = node_ids[*ni];
	++curEdge;
      }
      node_ids[*ii]->edgeend = curEdge;
    }
  }
};

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_Linear_Graph {
protected:
  struct NodeInfo;
  typedef EdgeInfoWrapper<NodeInfo,EdgeTy> EdgeInfo;

  struct NodeInfo : public Galoisruntime::Lockable {
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

  void* Data;
  uint64_t numNodes;
  uint64_t numEdges;
  NodeInfo** nodes;

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

  LC_Linear_Graph(): Data(0), nodes(0) { }

  ~LC_Linear_Graph() { 
    // TODO(ddn): call destructors of user data
    if (nodes)
      Galoisruntime::MM::largeInterleavedFree(nodes, sizeof(NodeInfo*) * numNodes);
    if (Data)
      Galoisruntime::MM::largeInterleavedFree(Data, sizeof(NodeInfo) * numNodes * 2 + sizeof(EdgeInfo) * numEdges);
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    // Galoisruntime::checkWrite(mflag);
    Galoisruntime::acquire(N, mflag);
    return N->data;
  }
  
  edge_data_reference getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    // Galoisruntime::checkWrite(mflag);
    Galoisruntime::acquire(src, mflag);
    return getEdgeIdx(src,dst)->getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) const {
    // Galoisruntime::checkWrite(mflag);
    return ni->getData();
  }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const {
    return numNodes;
  }

  uint64_t sizeEdges() const {
    return numEdges;
  }

  iterator begin() const {
    return nodes;
  }

  iterator end() const {
    return &nodes[numNodes];
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = ALL) {
    Galoisruntime::acquire(N, mflag);
    if (Galoisruntime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd();
	   ii != ee; ++ii) {
	Galoisruntime::acquire(ii->dst, mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = ALL) {
    Galoisruntime::acquire(N, mflag);
    return N->edgeEnd();
  }

  void fromFile(const std::string& fname) {
    FileGraph graph;
    graph.fromFile(fname);
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    Data = Galoisruntime::MM::largeInterleavedAlloc(sizeof(NodeInfo) * numNodes * 2 +
					 sizeof(EdgeInfo) * numEdges);
    nodes = reinterpret_cast<NodeInfo**>(Galoisruntime::MM::largeInterleavedAlloc(sizeof(NodeInfo*) * numNodes));
    NodeInfo* curNode = reinterpret_cast<NodeInfo*>(Data);
    for (FileGraph::iterator ii = graph.begin(),
	   ee = graph.end(); ii != ee; ++ii) {
	new (&curNode->data) NodeTy; //inplace new
      curNode->numEdges = graph.neighborsSize(*ii);
      nodes[*ii] = curNode;
      curNode = curNode->next();
    }

    //layout the edges
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
      EdgeInfo* edge = nodes[*ii]->edgeBegin();
      for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
	     ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
        edge->allocateEdgeData(graph, ni);
	edge->dst = nodes[*ni];
	++edge;
      }
    }
  }
};

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_Linear2_Graph {
protected:
  struct NodeInfo;
  typedef EdgeInfoWrapper<NodeInfo,EdgeTy> EdgeInfo;

  struct NodeInfo : public Galoisruntime::Lockable {
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

  Galoisruntime::PerThreadStorage<Header*> headers;
  NodeInfo** nodes;
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
  void distribute(FileGraph& graph, Galoisruntime::PerThreadStorage<DistributeInfo>& dinfo) {
    size_t total = sizeof(NodeInfo) * numNodes + sizeof(EdgeInfo) * numEdges;
    unsigned int num = galois::getActiveThreads();
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
        size_t nneighbors = graph.neighborsSize(*ii);
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
    Galoisruntime::PerThreadStorage<DistributeInfo>& dinfo;
    Galoisruntime::PerThreadStorage<Header*>& headers;
    NodeInfo** nodes;
    FileGraph& graph;

    AllocateNodes(
        Galoisruntime::PerThreadStorage<DistributeInfo>& d,
        Galoisruntime::PerThreadStorage<Header*>& h, NodeInfo** n, FileGraph& g):
      dinfo(d), headers(h), nodes(n), graph(g) { }

    void operator()(unsigned int tid, unsigned int num) {
      //DistributeInfo& d = dinfo.get(tid);
      DistributeInfo& d = *dinfo.getLocal();

      // extra 2 factors are for alignment purposes
      size_t size =
          sizeof(Header) * 2 +
          sizeof(NodeInfo) * d.numNodes * 2 +
          sizeof(EdgeInfo) * d.numEdges;

      void *raw = Galoisruntime::MM::largeAlloc(size);
      memset(raw, 0, size);

      Header*& h = *headers.getLocal();
      h = reinterpret_cast<Header*>(raw);
      h->size = size;
      h->begin = h->end = reinterpret_cast<NodeInfo*>(h + 1);

      if (!d.numNodes)
        return;

      for (FileGraph::iterator ii = d.begin, ee = d.end; ii != ee; ++ii) {
        new (&h->end->data) NodeTy; //inplace new
        h->end->numEdges = graph.neighborsSize(*ii);
        nodes[*ii] = h->end;
        h->end = h->end->next();
      }
    }
  };

  struct AllocateEdges {
    Galoisruntime::PerThreadStorage<DistributeInfo>& dinfo;
    NodeInfo** nodes;
    FileGraph& graph;

    AllocateEdges(Galoisruntime::PerThreadStorage<DistributeInfo>& d, NodeInfo** n, FileGraph& g):
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
          edge->allocateEdgeData(graph, ni);
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

  class local_iterator : public std::iterator<std::forward_iterator_tag, GraphNode> {
    const Galoisruntime::PerThreadStorage<Header*>* headers;
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
    local_iterator(const Galoisruntime::PerThreadStorage<Header*>* _headers, int _tid):
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

  LC_Linear2_Graph(): nodes(0) { }

  ~LC_Linear2_Graph() {
    // TODO(ddn): call destructors of user data
    if (nodes)
      Galoisruntime::MM::largeInterleavedFree(nodes, sizeof(NodeInfo*) * numNodes);
    for (unsigned i = 0; i < headers.size(); ++i) {
      Header* h = *headers.getRemote(i);
      if (h)
        Galoisruntime::MM::largeFree(h, h->size);
    }
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    // Galoisruntime::checkWrite(mflag);
    Galoisruntime::acquire(N, mflag);
    return N->data;
  }
  
  edge_data_reference getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    // Galoisruntime::checkWrite(mflag);
    Galoisruntime::acquire(src, mflag);
    return getEdgeIdx(src,dst)->getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) const {
    // Galoisruntime::checkWrite(mflag);
    return ni->getData();
  }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const {
    return numNodes;
  }

  uint64_t sizeEdges() const {
    return numEdges;
  }

  iterator begin() const {
    return nodes;
  }

  iterator end() const {
    return &nodes[numNodes];
  }

  local_iterator local_begin() const {
    return local_iterator(&headers, Galoisruntime::LL::getTID());
  }

  local_iterator local_end() const {
    return local_iterator(&headers, Galoisruntime::LL::getTID() + 1);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = ALL) {
    Galoisruntime::acquire(N, mflag);
    if (Galoisruntime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee; ++ii) {
	Galoisruntime::acquire(ii->dst, mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = ALL) {
    Galoisruntime::acquire(N, mflag);
    return N->edgeEnd();
  }

  void fromFile(const std::string& fname) {
    FileGraph graph;
    graph.fromFile(fname);
    numNodes = graph.size();
    numEdges = graph.sizeEdges();

    Galoisruntime::PerThreadStorage<DistributeInfo> dinfo;
    distribute(graph, dinfo);

    size_t size = sizeof(NodeInfo*) * numNodes;
    nodes = reinterpret_cast<NodeInfo**>(Galoisruntime::MM::largeInterleavedAlloc(size));

    galois::on_each(AllocateNodes(dinfo, headers, nodes, graph));
    galois::on_each(AllocateEdges(dinfo, nodes, graph));
  }
};


}
}
#endif
