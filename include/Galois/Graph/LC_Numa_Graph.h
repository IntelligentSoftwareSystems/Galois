/** Local Computation graphs -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_GRAPH_LC_NUMA_GRAPH_H
#define GALOIS_GRAPH_LC_NUMA_GRAPH_H

#include "Galois/Galois.h"
#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Details.h"
#include "Galois/Runtime/MethodFlags.h"

namespace Galois {
namespace Graph {

//! Local computation graph (i.e., graph structure does not change)
//! Specialization of LC_Linear_Graph for NUMA architectures
template<typename NodeTy, typename EdgeTy>
class LC_Numa_Graph: boost::noncopyable {
protected:
  struct NodeInfo;
  typedef detail::EdgeInfoBase<NodeInfo*,EdgeTy> EdgeInfo;
  typedef LargeArray<NodeInfo*> Nodes;

  struct NodeInfo : public detail::NodeInfoBase<NodeTy,true> {
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
        h->end->construct();
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
      typedef typename EdgeInfo::value_type EDV;
      //DistributeInfo& d = *dinfo.getRemote(tid);
      DistributeInfo& d = *dinfo.getLocal();
      if (!d.numNodes)
        return;

      for (FileGraph::iterator ii = d.begin, ee = d.end; ii != ee; ++ii) {
        EdgeInfo* edge = nodes[*ii]->edgeBegin();
        for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
               ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
          if (EdgeInfo::has_value)
            edge->construct(graph.getEdgeData<EDV>(ni));
          edge->dst = nodes[*ni];
          ++edge;
        }
      }
    }
  };

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef typename NodeInfo::reference node_data_reference;
  typedef EdgeInfo* edge_iterator;
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

      n->destruct();
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

  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    Galois::Runtime::acquire(N, mflag);
    return N->getData();
  }
  
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

  detail::EdgesIterator<LC_Numa_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return detail::EdgesIterator<LC_Numa_Graph>(*this, N, mflag);
  }

  void structureFromFile(const std::string& fname) { Graph::structureFromFile(*this, fname); }

  void structureFromGraph(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();

    Galois::Runtime::PerThreadStorage<DistributeInfo> dinfo;
    distribute(graph, dinfo);

    nodes.create(numNodes);

    Galois::on_each(AllocateNodes(dinfo, headers, nodes.data(), graph));
    Galois::on_each(AllocateEdges(dinfo, nodes.data(), graph));
  }
};

} // end namespace
} // end namespace

#endif
