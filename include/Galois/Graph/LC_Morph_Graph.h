/** Appendable semi-LC graphs -*- C++ -*-
 * @file
 * @section License
 *Graph which is like LC graphs but it is appendable only
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
 * @author Nikunj Yadav nikunj@cs.utexas.edu
 */

#ifndef LC_MORPH_GRAPH_H_
#define LC_MORPH_GRAPH_H_

#include "Galois/Galois.h"
#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Graph.h"
#include "Galois/Graph/Util.h"
#include "Galois/Runtime/MethodFlags.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/mm/Mem.h"

namespace Galois {
namespace Graph {
//! Local computation graph (i.e., graph structure does not change)
//! Specialization of LC_Linear_Graph for NUMA architectures
template<typename NodeTy, typename EdgeTy>
class LC_Morph_Graph: private boost::noncopyable {
protected:
  struct NodeInfo;
  typedef GraphImpl::EdgeItem<NodeInfo, EdgeTy, true> EITy;
  //typedef Galois::gdeque<NodeInfo,64> NodeListTy;
  
  struct EdgeHolder {
    EITy* begin;
    EITy* end;
    EdgeHolder* next;
  };

  struct NodeInfo: public Galois::Runtime::Lockable {
    NodeTy data;
    unsigned debugEdges;
    EITy* edgeBegin;
    EITy* edgeEnd;
    template<typename... Args>
    NodeInfo(Args&& ...args):data(std::forward<Args>(args)...){
    }
  };

  typedef Galois::InsertBag<NodeInfo> NodeListTy;
  NodeListTy nodes;
  Galois::Runtime::PerThreadStorage<EdgeHolder*> edges;

  struct makeGraphNode: public std::unary_function<NodeInfo&, NodeInfo*> {
    NodeInfo* operator()(NodeInfo& data) const { return &data; }
  };
  
  struct first_equals {
    NodeInfo* dst;
    first_equals(NodeInfo* d): dst(d) { }
    bool operator()(const EITy& edge) { return edge.first() == dst; }
  };

public:

  typedef boost::transform_iterator<makeGraphNode,typename NodeListTy::iterator > iterator;
  typedef iterator local_iterator;
  typedef NodeInfo* GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EITy::reference edge_data_reference;
  typedef EITy* edge_iterator;
    
  std::vector<GraphNode> trackingG;//FIXE ME: keep the real order of the graph, usefull for output the partitioning. Need to be removed.

  NodeTy& getData(const GraphNode& N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    Galois::Runtime::acquire(N, mflag);
    return N->data;
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::ALL) const {
    Galois::Runtime::checkWrite(mflag, false);
    Galois::Runtime::acquire(ni->first(), mflag);
    return *ni->second();
  }

  GraphNode getEdgeDst(edge_iterator ni, MethodFlag mflag = MethodFlag::ALL) const {
    Galois::Runtime::checkWrite(mflag, false);
    Galois::Runtime::acquire(ni->first(), mflag);
    return GraphNode(ni->first());
  }

  /**
   * Returns an iterator to all the nodes in the graph. Not thread-safe.
   */
  iterator begin() {
    return boost::make_transform_iterator(nodes.begin(),makeGraphNode());
  }

  //! Returns the end of the node iterator. Not thread-safe.
  iterator end() {
    return boost::make_transform_iterator(nodes.end(),makeGraphNode());
  }

  local_iterator local_begin() {
    return boost::make_transform_iterator(nodes.local_begin(),makeGraphNode());
  }
  
  local_iterator local_end() {
    return boost::make_transform_iterator(nodes.local_end(),makeGraphNode());
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin, ee = N->edgeEnd; ii != ee; ++ii) {
        Galois::Runtime::acquire(ii->first(), mflag);
      }
    }
    return N->edgeBegin;
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return N->edgeEnd;
  }
  
  template<typename... Args>
  GraphNode createNode(int nedges, Args&&... args) {
    Galois::Runtime::checkWrite(MethodFlag::ALL, true);
    NodeInfo* N = &(nodes.emplace(std::forward<Args>(args)...));
    Galois::Runtime::acquire(N, MethodFlag::ALL);
    EdgeHolder*& local_edges = *edges.getLocal();
    if (!local_edges || std::distance(local_edges->begin, local_edges->end) < nedges) {
      EdgeHolder* old = local_edges;
      char* newblock = (char*)Runtime::MM::pageAlloc();
      local_edges = (EdgeHolder*)newblock;
      local_edges->next = old;
      char* estart = newblock + sizeof(EdgeHolder);
      if ((uintptr_t)estart % sizeof(EITy)) //not aligned
        estart += sizeof(EITy) - ((uintptr_t)estart % alignof(EITy));

      local_edges->begin = (EITy*)estart;
      char* eend = newblock + Runtime::MM::pageSize;
      eend -= (uintptr_t)eend % sizeof(EITy);
      local_edges->end = (EITy*)eend;
    }
    N->edgeBegin = N->edgeEnd = local_edges->begin;
    local_edges->begin += nedges;
    N->debugEdges = nedges;
    return GraphNode(N);
  }

  template<typename... Args>
  edge_iterator addEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag, Args&&... args) {
    Galois::Runtime::checkWrite(mflag, true);
    Galois::Runtime::acquire(src, mflag);
    assert(std::distance(src->edgeBegin, src->edgeEnd) < src->debugEdges);
    auto it = std::find_if(src->edgeBegin, src->edgeEnd, first_equals(dst));
    if (it == src->edgeEnd) {
      new (it) EITy(dst, 0, std::forward<Args>(args)...);
      src->edgeEnd++;
    }
    assert(std::distance(src->edgeBegin, src->edgeEnd) <= src->debugEdges);
    return it;
  }

  template<typename... Args>
  edge_iterator addEdgeWithoutCheck(GraphNode src, GraphNode dst, Galois::MethodFlag mflag, Args&&... args) {
    Galois::Runtime::checkWrite(mflag, true);
    Galois::Runtime::acquire(src, mflag);
    assert(std::distance(src->edgeBegin, src->edgeEnd) < src->debugEdges);
    auto it = src->edgeEnd;
    new (it) EITy(dst, 0, std::forward<Args>(args)...);
    src->edgeEnd++;
    assert(std::distance(src->edgeBegin, src->edgeEnd) <= src->debugEdges);
    return it;
  }
  
  edge_iterator findEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, true);
    Galois::Runtime::acquire(src, mflag);
    return std::find_if(src->edgeBegin, src->edgeEnd, first_equals(dst)); 
  }
  
  void structureFromFile(const std::string& fname) { Graph::structureFromFile(*this, fname); }
  
  struct CreateNodes {
    LC_Morph_Graph* self;
    std::vector<GraphNode>& tracking;
    FileGraph& graph;
    std::atomic<unsigned>& nNodes;
    
    CreateNodes(
      LC_Morph_Graph* _self,
      std::vector<GraphNode>& _tracking,
      FileGraph& _graph,
      std::atomic<unsigned>& _nNodes): self(_self), tracking(_tracking), graph(_graph), nNodes(_nNodes) { }

    void operator()(FileGraph::GraphNode gn) {
       tracking[gn] = self->createNode(std::distance(graph.edge_begin(gn), graph.edge_end(gn))); 
       ++nNodes;
    }
  };

  struct CreateEdges {
    LC_Morph_Graph* self;
    std::vector<GraphNode>& tracking;
    FileGraph& graph;
    std::atomic<unsigned>& nEdges;
    
    CreateEdges(
      LC_Morph_Graph* _self,
      std::vector<GraphNode>& _tracking,
      FileGraph& _graph,
      std::atomic<unsigned>& _nEdges): self(_self), tracking(_tracking), graph(_graph), nEdges(_nEdges) { }

    void operator()(FileGraph::GraphNode gn) {
       for (auto ii = graph.edge_begin(gn), ee = graph.edge_end(gn); ii != ee; ++ii) {
         self->addEdgeWithoutCheck(tracking[gn], tracking[graph.getEdgeDst(ii)], Galois::MethodFlag::NONE, graph.getEdgeData<uint32_t>(ii));
         ++nEdges;
       }
    }
  };

  void structureFromGraph(FileGraph& graph) {

    //if we can keep the node order we should delete trackingG and turn tracking into a vector and not a ref.
    //std::vector<GraphNode> tracking;
    std::vector<GraphNode>& tracking = trackingG;

    tracking.resize(graph.size());

    std::atomic<unsigned> nEdges(0), nNodes(0);
    Galois::do_all(graph.begin(), graph.end(), CreateNodes(this, tracking, graph, nNodes));
    Galois::do_all(graph.begin(), graph.end(), CreateEdges(this, tracking, graph, nEdges));
    std::cout << "Created Graph with " << nNodes << " nodes and " << nEdges << " edges\n";

  }

  void dump(std::ostream& out) {
    out << "digraph {\n";
    for (auto nn = begin(), en = end(); nn != en; ++nn) {
      out << '"' << *nn << "\" [shape=box];\n";
    }
    for (auto nn = begin(), en = end(); nn != en; ++nn) {
      for (auto ii = edge_begin(*nn), ee = edge_end(*nn); ii != ee; ++ii)
        out << '"' << *nn << "\" -> \"" << getEdgeDst(ii) << "\";\n";
    }

    out << "}\n";
  }
};

} // end namespace
} // end namespace


#endif /* LC_MORPH_GRAPH_H_ */
