#include <iterator>
#include <deque>

#include "Galois/Runtime/DistSupport.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/MethodFlags.h"
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

  void dump(std::ostream& os) {
    os << "next: ";
    nextNode.dump();
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

  void dump(std::ostream& os) {
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

  void dump(std::ostream& os) {
    os << "<{Edge: dst: ";
    dst.dump();
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

  void dump(std::ostream& os) {
    os << "<{Edge: dst: ";
    dst.dump();
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
  void dump(std::ostream& os) {
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
  void dump(std::ostream& os) {
    os << "numedges: " << edges.size();
    for (decltype(edges.size()) x = 0; x < edges.size(); ++x) {
      os << " ";
      edges[x].dump(os);
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
  void dump(std::ostream& os) {
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
    typedef int tt_dir_blocking;
    void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
      gSerialize(s, head, next, master);
    }
    void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
      gDeserialize(s,head, next, master);
    }
    SubGraphState() :head(), next(), master(this) {}
  };

  SubGraphState localState;

  struct is_edge : public std::unary_function<typename gNode::EdgeType&, bool> {
    bool operator()(typename gNode::EdgeType& n) const { return n.getDst()->getActive(); }
  };

public:
  typedef typename gNode::Handle NodeHandle;
  //! Edge iterator
  typedef typename boost::filter_iterator<is_edge,typename gNode::edge_iterator> edge_iterator;

  template<typename... Args>
  NodeHandle createNode(Args&&... args) {
    NodeHandle N(new gNode(std::forward<Args...>(args...)));
    N->getNextNode() = localState.head;
    localState.head = N;
    return N;
  }

  NodeHandle createNode() {
    NodeHandle N(new gNode());
    N->getNextNode() = localState.head;
    localState.head = N;
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
  
  class iterator : public std::iterator<std::forward_iterator_tag, const NodeHandle> {
    NodeHandle n;
    Galois::Runtime::Distributed::gptr<SubGraphState> s;
    void next() {
      n = n->getNextNode();
      while (!n && s->next) {
        s = s->next;
        n = s->head;
      }
      // skip node if not active!
      if (n && !n->getActive())
        next();
      if (!n) s.initialize(nullptr);
    }

  public:
    using typename std::iterator<std::forward_iterator_tag, const NodeHandle>::pointer;
    using typename std::iterator<std::forward_iterator_tag, const NodeHandle>::reference;

  iterator() :n(), s() {}
    explicit iterator(const Galois::Runtime::Distributed::gptr<SubGraphState> ms) :n(ms->head), s(ms) {
      while (!n && s->next) {
        s = s->next;
        n = s->head;
      }
      // skip node if not active!
      if (n && !n->getActive())
        next();
      if (!n) s.initialize(nullptr);
    }

    reference operator*() const { return n; }
    pointer operator->() const { return &n; }
    iterator& operator++() { next(); return *this; }
    iterator operator++(int) { iterator tmp(*this); next(); return tmp; }
    bool operator==(const iterator& rhs) const { return n == rhs.n; }
    bool operator!=(const iterator& rhs) const { return n != rhs.n; }

    void dump() {
      n.dump();
      s.dump();
    }
  };

  iterator begin() { return iterator(localState.master); }
  iterator end() { return iterator(); }

  class local_iterator : public std::iterator<std::forward_iterator_tag, NodeHandle> {
    NodeHandle n;
    void next() {
      n = n->getNextNode();
      // skip node if not active!
      if (n && !n->getActive())
        next();
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

  local_iterator local_begin() { return local_iterator(localState.head); }
  local_iterator local_end() { return local_iterator(); }

  //! Returns an iterator to the neighbors of a node 
  edge_iterator edge_begin(NodeHandle N) {
    assert(N);
    N.acquire();
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

  unsigned int size() {
    return std::distance(begin(), end());
  }

  ThirdGraph() {}
  // mark the graph as persistent so that it is distributed
  typedef int tt_is_persistent;
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    //This is what is called on the source of a replicating source
    gSerialize(s,localState.master);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    //This constructs the local node of the distributed graph
    gDeserialize(s,localState.master);
    localState.next = localState.master->next;
    localState.master->next.initialize(&localState);
  }
  
};


} //namespace Graph
} //namespace Galois
