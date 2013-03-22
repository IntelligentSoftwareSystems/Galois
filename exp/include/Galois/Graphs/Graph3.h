/** deque like structure with scalable allocator usage -*- C++ -*-
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_GRAPH3_H
#define GALOIS_GRAPH3_H

#include <iterator>
#include <deque>

#include "Galois/Threads.h"
#include "Galois/Runtime/DistSupport.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/MethodFlags.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/mm/Mem.h"

#include <boost/iterator/filter_iterator.hpp>

namespace Galois {
namespace Graph {

enum class EdgeDirection {Un, Out, InOut};

template<typename NodeTy, typename EdgeTy, EdgeDirection EDir>
class ThirdGraph;

template<typename NHTy>
class GraphNodeBase {
  NHTy nextNode;
  bool active;

protected:
  GraphNodeBase() :active(false) {}

  NHTy& getNextNode() { return nextNode; }

  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s, nextNode, active);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,nextNode, active);
  }

  void dump(std::ostream& os) const {
    os << "next: ";
    nextNode.dump(os);
    os << " active: ";
    os << active;
  }

public:
  bool getActive() {
    return active;
  }

  void setActive(bool b) {
    active = b;
  }
};


template<typename NodeDataTy>
class GraphNodeData {
  NodeDataTy data;
  
protected:

  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,data);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,data);
  }

  void dump(std::ostream& os) const {
    os << "data: " << data;
  }

public:
  template<typename... Args>
  GraphNodeData(Args&&... args) :data(std::forward<Args...>(args...)) {}
  GraphNodeData() :data() {}

  NodeDataTy& getData() {
    return data;
  }
};

template<>
class GraphNodeData<void> {};

template<typename NHTy, typename EdgeDataTy, EdgeDirection EDir>
class GraphNodeEdges;

template<typename NHTy, typename EdgeDataTy>
class Edge {
  NHTy dst;
  EdgeDataTy val;
public:
  template<typename... Args>
  Edge(const NHTy& d, Args&&... args) :dst(d), val(std::forward<Args...>(args...)) {}

  Edge() {}

  NHTy getDst() { return dst; }
  EdgeDataTy& getValue() { return val; }

  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s, dst, val);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,dst, val);
  }

  void dump(std::ostream& os) const {
    os << "<{Edge: dst: ";
    dst.dump(os);
    os << " dst active: ";
    os << dst->getActive();
    os << " val: ";
    os << val;
    os << "}>";
  }
};

template<typename NHTy>
class Edge<NHTy, void> {
  NHTy dst;
public:
  Edge(const NHTy& d) :dst(d) {}
  Edge() {}

  NHTy getDst() { return dst; }

  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,dst);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,dst);
  }

  void dump(std::ostream& os) const {
    os << "<{Edge: dst: ";
    dst.dump(os);
    os << " dst active: ";
    os << dst->getActive();
    os << "}>";
  }
};

template<typename NHTy, typename EdgeDataTy>
class GraphNodeEdges<NHTy, EdgeDataTy, EdgeDirection::Out> {
  typedef Edge<NHTy, EdgeDataTy> EdgeTy;
  typedef std::deque<EdgeTy> EdgeListTy;

  EdgeListTy edges;

protected:
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,edges);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,edges);
  }
  void dump(std::ostream& os) const {
    os << "numedges: " << edges.size();
    for (decltype(edges.size()) x = 0; x < edges.size(); ++x) {
      os << " ";
      edges[x].dump(os);
    }
  }
 public:
  typedef typename EdgeListTy::iterator iterator;

  template<typename... Args>
  iterator createEdge(const NHTy& src, const NHTy& dst, Args&&... args) {
    *src;
    return edges.emplace(edges.end(), dst, std::forward<Args...>(args...));
  }

  iterator createEdge(const NHTy& src, const NHTy& dst) {
    *src;
    return edges.emplace(edges.end(), dst);
  }

  void clearEdges() {
    edges.clear();
  }

  iterator begin() {
    return edges.begin();
  }

  iterator end() {
    return edges.end();
  }
};

template<typename NHTy, typename EdgeDataTy>
class GraphNodeEdges<NHTy, EdgeDataTy, EdgeDirection::InOut> {
  //FIXME
};

template<typename NHTy>
class GraphNodeEdges<NHTy, void, EdgeDirection::Un> {
  typedef Edge<NHTy, void> EdgeTy;
  typedef std::deque<EdgeTy> EdgeListTy;

  EdgeListTy edges;

protected:
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,edges);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,edges);
  }
  void dump(std::ostream& os) const {
    os << "numedges: " << edges.size();
    for (auto ii = edges.begin(), ee = edges.end(); ii != ee; ++ii) {
      os << " ";
      ii->dump(os);
    }
  }
 public:
  typedef typename EdgeListTy::iterator iterator;

  iterator createEdge(NHTy& src, NHTy& dest) {
    //assert(*src == this);
    dest->edges.emplace(dest->edges.end(), src);
    return edges.emplace(edges.end(), dest);
  }

  void clearEdges() {
    edges.clear();
  }

  iterator begin() {
    return edges.begin();
  }

  iterator end() {
    return edges.end();
  }
};

template<typename NHTy, typename EdgeDataTy>
class GraphNodeEdges<NHTy, EdgeDataTy, EdgeDirection::Un> {
  //FIXME
};


#define SHORTHAND Galois::Runtime::Distributed::gptr<GraphNode<NodeDataTy, EdgeDataTy, EDir> >

template<typename NodeDataTy, typename EdgeDataTy, EdgeDirection EDir>
class GraphNode
  : public Galois::Runtime::Lockable,
    public GraphNodeBase<SHORTHAND >,
    public GraphNodeData<NodeDataTy>,
    public GraphNodeEdges<SHORTHAND, EdgeDataTy, EDir>
{
  friend class ThirdGraph<NodeDataTy, EdgeDataTy, EDir>;

  using GraphNodeBase<SHORTHAND >::getNextNode;

public:
  typedef SHORTHAND Handle;
  typedef typename Galois::Graph::Edge<SHORTHAND,EdgeDataTy> EdgeType;
  typedef typename GraphNodeEdges<SHORTHAND,EdgeDataTy,EDir>::iterator edge_iterator;

  template<typename... Args>
  GraphNode(Args&&... args) :GraphNodeData<NodeDataTy>(std::forward<Args...>(args...)) {}

  GraphNode() {}

  //serialize
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    GraphNodeBase<SHORTHAND >::serialize(s);
    GraphNodeData<NodeDataTy>::serialize(s);
    GraphNodeEdges<SHORTHAND, EdgeDataTy, EDir>::serialize(s);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    GraphNodeBase<SHORTHAND >::deserialize(s);
    GraphNodeData<NodeDataTy>::deserialize(s);
    GraphNodeEdges<SHORTHAND, EdgeDataTy, EDir>::deserialize(s);
  }
  void dump(std::ostream& os) const {
    os << this << " ";
    os << "<{GN: ";
    GraphNodeBase<SHORTHAND >::dump(os);
    os << " ";
    GraphNodeData<NodeDataTy>::dump(os);
    os << " ";
    GraphNodeEdges<SHORTHAND, EdgeDataTy, EDir>::dump(os);
    os << "}>";
  }
};

#undef SHORTHAND

/**
 * A Graph
 *
 * @param NodeTy type of node data (may be void)
 * @param EdgeTy type of edge data (may be void)
 * @param IsDir  bool indicated if graph is directed
 *
*/
template<typename NodeTy, typename EdgeTy, EdgeDirection EDir>
class ThirdGraph { //: public Galois::Runtime::Distributed::DistBase<ThirdGraph> {
  typedef GraphNode<NodeTy, EdgeTy, EDir> gNode;

  struct SubGraphState : public Galois::Runtime::Lockable {
    typename gNode::Handle head;
    Galois::Runtime::Distributed::gptr<SubGraphState> next;
    Galois::Runtime::Distributed::gptr<SubGraphState> master;
    typedef int tt_has_serialize;
    void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
      gSerialize(s, head, next, master);
    }
    void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
      gDeserialize(s, head, next, master);
    }
    void dump(std::ostream& os) const {
      os << "Subgraph " << this << " with master ";
      master.dump(os);
      os << " and next ";
      next.dump(os);
      os << " and head ";
      head.dump(os);
      os << "\n";
    }
    SubGraphState() :head(), next(), master(this) {}
  };

  Galois::Runtime::PerThreadStorage<SubGraphState> localState;

  struct is_edge : public std::unary_function<typename gNode::EdgeType&, bool> {
    bool operator()(typename gNode::EdgeType& n) const { return n.getDst()->getActive(); }
  };

  Galois::Runtime::MM::FixedSizeAllocator heap;

public:
  typedef typename gNode::Handle NodeHandle;
  //! Edge iterator
  typedef typename boost::filter_iterator<is_edge,typename gNode::edge_iterator> edge_iterator;

  template<typename... Args>
  NodeHandle createNode(Args&&... args) {
    void* vp = heap.allocate(sizeof(gNode));
    NodeHandle N(new (vp) gNode(std::forward<Args...>(args...)));
    // lock the localState before adding the node
    gptr<SubGraphState> lStatePtr(localState.getLocal());
    SubGraphState* p = transientAcquire(lStatePtr);
    N->getNextNode() = p->head;
    p->head = N;
    transientRelease(lStatePtr);
    return N;
  }

  NodeHandle createNode() {
    void* vp = heap.allocate(sizeof(gNode));
    NodeHandle N(new (vp) gNode());
    // lock the localState before adding the node
    gptr<SubGraphState> lStatePtr(localState.getLocal());
    SubGraphState* p = transientAcquire(lStatePtr);
    N->getNextNode() = p->head;
    p->head = N;
    transientRelease(lStatePtr);
    return N;
  }
  
  void addNode(NodeHandle& N) {
    N->setActive(true);
  }
  
  void removeNode(NodeHandle& N) {
    if (N->getActive()) {
      N->setActive(false);
      // delete all the edges (in the deque)
      N->clearEdges();
    }
  }
  
  class iterator : public std::iterator<std::forward_iterator_tag, NodeHandle> {
    NodeHandle n;
    Galois::Runtime::Distributed::gptr<SubGraphState> s;
    void next() {
      // skip node if not active!
      do {
        n = n->getNextNode();
        while (!n && s->next) {
          s = s->next;
          n = s->head;
        }
      } while (n && !(n->getActive()));
      if (!n) s.initialize(nullptr);
    }

  public:
    using typename std::iterator<std::forward_iterator_tag, NodeHandle>::pointer;
    using typename std::iterator<std::forward_iterator_tag, NodeHandle>::reference;

  iterator() :n(), s() {}
    explicit iterator(const Galois::Runtime::Distributed::gptr<SubGraphState> ms) :n(ms->head), s(ms) {
      assert(n);
      assert(s);
      while (!n && s->next) {
        s = s->next;
        n = s->head;
      }
      // skip node if not active!
      if (n && !(n->getActive()))
        next();
      if (!n) s.initialize(nullptr);
    }

    reference operator*() { return n; }
    pointer operator->() const { return &n; }
    iterator& operator++() { next(); return *this; }
    iterator operator++(int) { iterator tmp(*this); next(); return tmp; }
    bool operator==(const iterator& rhs) const { return n == rhs.n; }
    bool operator!=(const iterator& rhs) const { return n != rhs.n; }

    void dump(std::ostream& os) const {
      n.dump(os);
      s.dump(os);
    }
  };

  iterator begin() { return iterator(localState.getLocal()->master); }
  iterator end() { return iterator(); }

  class local_iterator : public std::iterator<std::forward_iterator_tag, NodeHandle> {
    NodeHandle n;
    void next() {
      NodeHandle tn = n;
      // skip node if not active!
      do {
        tn = tn->getNextNode();
      } while (tn && !tn->getActive());
      // require a temp so that iterator doesn't get incremented and still end
      // up throwing an exception when dereferencing for the getActive() call
      n = tn;
    }
  public:
    explicit local_iterator(NodeHandle N) :n(N) {}
    local_iterator() :n() {}
    local_iterator(const local_iterator& mit) : n(mit.n) {
      // skip node if not active!
      if (n && !n->getActive())
        next();
    }

    NodeHandle& operator*() { return n; }
    local_iterator& operator++() { next(); return *this; }
    local_iterator operator++(int) { local_iterator tmp(*this); operator++(); return tmp; }
    bool operator==(const local_iterator& rhs) { return n == rhs.n; }
    bool operator!=(const local_iterator& rhs) { return n != rhs.n; }
  };

  local_iterator local_begin() { return local_iterator(localState.getLocal()->head); }
  local_iterator local_end() { return local_iterator(); }

  //! Returns an iterator to the neighbors of a node 
  edge_iterator edge_begin(NodeHandle N) {
    assert(N);
    // prefetch all the nodes
    for (auto ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
      ii->getDst().prefetch();
    }
    // lock all the nodes
    for (auto ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
      // NOTE: Andrew thinks acquire may be needed for inactive nodes too
      //       not sure why though. he had to do this in the prev graph
      if (ii->getDst()->getActive()) {
        // modify the call when local nodes aren't looked up in directory
	//        ii->getDst().acquire();
      }
    }
    return boost::make_filter_iterator(is_edge(), N->begin(), N->end());
  }

  //! Returns the end of the neighbor iterator 
  edge_iterator edge_end(NodeHandle N) {
    assert(N);
    return boost::make_filter_iterator(is_edge(), N->end(), N->end());
  }

  void addEdge(NodeHandle src, NodeHandle dst) {
    assert(src);
    assert(dst);
    src->createEdge(src, dst);
  }

  NodeHandle getEdgeDst(edge_iterator ii) {
    assert(ii->getDst()->getActive());
    return ii->getDst();
  }

  NodeTy& getData(const NodeHandle& N) {
    assert(N);
    return N->getData();
  }

  bool containsNode(const NodeHandle& N) {
    assert(N);
    return N->getActive();
  }

  unsigned size() {
    return std::distance(begin(), end());
  }

  void dump(std::ostream& os) const {
    os << "Graph at " << Galois::Runtime::Distributed::networkHostID << " with " << localState.size() << " subgraphs\n";
    for (unsigned int i = 0; i < localState.size(); i++)
      localState.getRemote(i)->dump(os);
    os << "\nDone Graph\n";
  }

  // mark the graph as persistent so that it is distributed
  typedef int tt_is_persistent;
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    //This is what is called on the source of a replicating source
    gptr<SubGraphState> lStatePtr(localState.getRemote(0));
    assert(lStatePtr);
    gSerialize(s,lStatePtr);
  }
  
 ThirdGraph() :heap(sizeof(gNode)) {
    //    std::cout << "TG " << Galois::Runtime::Distributed::networkHostID << " " << Galois::Runtime::LL::getTID() << "\n";
    for (unsigned int i = 0; i < localState.size(); ++i)
      localState.getRemote(i)->master.initialize(localState.getRemote(0));
    for (unsigned i = 0; i < localState.size() - 1; ++i)
      localState.getRemote(i)->next.initialize(localState.getRemote(i+1));
    //    dump(std::cout);
  }

  SubGraphState* getMasterState() __attribute__((used)) {
    return localState.getRemote(0);
  }

 ThirdGraph(Galois::Runtime::Distributed::DeSerializeBuffer& s) :heap(sizeof(gNode)) {
    //    std::cout << "TG " << Galois::Runtime::Distributed::networkHostID << " " << Galois::Runtime::LL::getTID() << "\n";
   //This constructs the local node of the distributed graph
    gptr<SubGraphState> master;
    gDeserialize(s,master);
    assert(master);
    for (unsigned i = 0; i < localState.size() - 1; ++i)
      localState.getRemote(i)->next.initialize(localState.getRemote(i+1));
    for (unsigned int i = 0; i < localState.size(); i++)
      localState.getRemote(i)->master = master;

    SubGraphState* masterPtr = transientAcquire(master);
    localState.getRemote(localState.size() - 1)->next = masterPtr->next;
    masterPtr->next.initialize(localState.getRemote(0));
    transientRelease(master);
    //    dump(std::cout);
  }

};

// used to find the size of the graph
struct R : public Galois::Runtime::Lockable {
  unsigned i;

  R(): i(0) {}

  void add(unsigned v) {
    i += v;
    return;
  }

  typedef int tt_dir_blocking;

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,i);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,i);
  }
};

template <typename GTy>
struct f {
  GTy graph;
  gptr<R> r;

  f(const gptr<R>& p, GTy g): graph(g), r(p) {}
  f() {}

  template<typename Context>
  void operator()(unsigned x, Context& cnx) const {
    unsigned size = std::distance(graph->local_begin(),graph->local_end());
    r->add(size);
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,r);
    gSerialize(s,graph);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,r);
    gDeserialize(s,graph);
  }
};

template <typename GraphTy>
unsigned ThirdGraphSize(GraphTy g) {
  // should only be called from outside the for_each
  assert(!Galois::Runtime::inGaloisForEach);
  R tmp;
  gptr<R> r(&tmp);
  Galois::on_each(f<GraphTy>(r,g));
  return r->i;
}

template <typename GraphTy>
struct ThirdGraph_for_size {
  ThirdGraph_for_size() {}
  bool operator()(typename GraphTy::element_type::NodeHandle n) const {
    return true;
  }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
  }
};

template <typename GraphTy>
ptrdiff_t NThirdGraphSize(GraphTy g) {
  // should only be called from outside the for_each
  assert(!Galois::Runtime::inGaloisForEach);
  return Galois::ParallelSTL::count_if_local(g,ThirdGraph_for_size<GraphTy>());
}

} //namespace Graph
} //namespace Galois

#endif
