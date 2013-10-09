#ifndef GALOIS_GRAPHLABEXECUTOR_H
#define GALOIS_GRAPHLABEXECUTOR_H

#include "Galois/Bag.h"
#include "Galois/Reduction.h"
#include "Galois/Runtime/PerHostStorage.h"

#include <boost/mpl/has_xxx.hpp>

namespace Galois {
namespace GraphLab {

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_gather_in_edges)
template<typename T>
struct needs_gather_in_edges: public has_tt_needs_gather_in_edges<T> {};

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_gather_out_edges)
template<typename T>
struct needs_gather_out_edges: public has_tt_needs_gather_out_edges<T> {};

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_scatter_in_edges)
template<typename T>
struct needs_scatter_in_edges: public has_tt_needs_scatter_in_edges<T> {};

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_scatter_out_edges)
template<typename T>
struct needs_scatter_out_edges: public has_tt_needs_scatter_out_edges<T> {};

struct EmptyMessage {
  EmptyMessage& operator+=(const EmptyMessage&) { return *this; }
};

//! Turn types with operator+= into binary function
template<typename T>
struct AddBinFunc {
  T operator()(const T& a, const T& b) const {
    T ret(a);
    ret += b;
    return ret;
  }
};

template<typename Graph, typename Operator> 
struct Context {
  typedef typename Graph::GraphNode GNode;
  typedef typename Operator::message_type message_type;
  typedef std::pair<GNode,message_type> WorkItem;
  typedef DGReducibleVector<message_type, AddBinFunc<message_type>> Messages;
  typedef LargeArray<int,false> Scoreboard;
  typedef Runtime::PerHost<Galois::InsertBag<GNode>> Next;

private:
  template<typename,typename> friend class AsyncEngine;
  template<typename,typename> friend class SyncEngine;

  Galois::UserContext<WorkItem>* ctx;
  Graph* graph;
  Scoreboard* scoreboard;
  Next* next;
  Messages* messages;

  Context() { }

  void initializeForSync(Graph* g, Scoreboard* s, Next* n, Messages* m) {
    graph = g;
    scoreboard = s;
    next = n;
    messages = m;
    ctx = 0;
  }

  void initializeForAsync(Galois::UserContext<WorkItem>* c) {
    ctx = c;  
  }

  void pushInternal(GNode node, size_t id) {
    // XXX check if remote node 
    int val = (*scoreboard)[id];
    if (val == 0 && __sync_bool_compare_and_swap(&(*scoreboard)[id], 0, 1)) {
      assert(0 && "Fixme");
      //next->push(node);
    }
  }

public:
  void push(GNode node, const message_type& message) {
    if (ctx) {
      ctx->push(WorkItem(node, message));
    } else {
      size_t id = graph->idFromNode(node);
      pushInternal(node, id);

      if (messages) {
        messages->update(id, message);
      }
    }
  }
};

template<typename Graph, typename Operator>
class AsyncEngine {
  typedef typename Operator::message_type message_type;
  typedef typename Operator::gather_type gather_type;
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::in_edge_iterator in_edge_iterator;
  typedef typename Graph::edge_iterator edge_iterator;

  typedef typename Context<Graph,Operator>::WorkItem WorkItem;

  struct Initialize {
    AsyncEngine* self;
    Galois::InsertBag<WorkItem>& bag;

    Initialize(AsyncEngine* s, Galois::InsertBag<WorkItem>& b): self(s), bag(b) { }

    void operator()(GNode n) {
      bag.push(WorkItem(n, message_type()));
    }
  };

  struct Process {
    AsyncEngine* self;
    Process(AsyncEngine* s): self(s) { }

    void operator()(const WorkItem& item, Galois::UserContext<WorkItem>& ctx) {
      Operator op(self->origOp);

      GNode node = item.first;
      message_type msg = item.second;
      
      if (needs_gather_in_edges<Operator>::value || needs_scatter_in_edges<Operator>::value) {
        self->graph.in_edge_begin(node, Galois::MethodFlag::ALL);
      }

      if (needs_gather_out_edges<Operator>::value || needs_scatter_out_edges<Operator>::value) {
        self->graph.edge_begin(node, Galois::MethodFlag::ALL);
      }

      op.init(self->graph, node, msg);
      
      gather_type sum;
      if (needs_gather_in_edges<Operator>::value) {
        for (in_edge_iterator ii = self->graph.in_edge_begin(node, Galois::MethodFlag::NONE),
            ei = self->graph.in_edge_end(node, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          op.gather(self->graph, node, self->graph.getInEdgeDst(ii), node, sum, self->graph.getInEdgeData(ii));
        }
      }
      if (needs_gather_out_edges<Operator>::value) {
        for (edge_iterator ii = self->graph.edge_begin(node, Galois::MethodFlag::NONE), 
            ei = self->graph.edge_end(node, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          op.gather(self->graph, node, node, self->graph.getEdgeDst(ii), sum, self->graph.getEdgeData(ii));
        }
      }

      op.apply(self->graph, node, sum);

      if (!op.needsScatter(self->graph, node))
        return;

      Context<Graph,Operator> context;
      context.initializeForAsync(&ctx);

      if (needs_scatter_in_edges<Operator>::value) {
        for (in_edge_iterator ii = self->graph.in_edge_begin(node, Galois::MethodFlag::NONE),
            ei = self->graph.in_edge_end(node, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          op.scatter(self->graph, node, self->graph.getInEdgeDst(ii), node, context, self->graph.getInEdgeData(ii));
        }
      }
      if (needs_scatter_out_edges<Operator>::value) {
        for (edge_iterator ii = self->graph.edge_begin(node, Galois::MethodFlag::NONE), 
            ei = self->graph.edge_end(node, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          op.scatter(self->graph, node, node, self->graph.getEdgeDst(ii), context, self->graph.getEdgeData(ii));
        }
      }
    }
  };

  Graph& graph;
  Operator origOp;

public:
  AsyncEngine(Graph& g, Operator o): graph(g), origOp(o) { }

  void execute() {
    typedef typename Context<Graph,Operator>::WorkItem WorkItem;
    typedef Galois::WorkList::dChunkedFIFO<256> WL;

    Galois::InsertBag<WorkItem> bag;
    Galois::do_all_local(graph, Initialize(this, bag));
    Galois::for_each_local<WL>(bag, Process(this));
  }
};

using namespace Galois::Runtime::Distributed;

template<typename Graph, typename Operator>
class SyncEngine {
  typedef typename Operator::message_type message_type;
  typedef typename Operator::gather_type gather_type;
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::in_edge_iterator in_edge_iterator;
  typedef typename Graph::edge_iterator edge_iterator;
  static const bool NeedMessages = !std::is_same<EmptyMessage,message_type>::value;
  typedef Galois::WorkList::dChunkedFIFO<256> WL;
  typedef DGReducibleVector<message_type, AddBinFunc<message_type>> Messages;

  template<typename T>
  struct or_ {
    T operator()(const T& a, const T& b) const {
      return a || b;
    }
  };

  typedef DGReducible<bool, or_<bool>> Terminator;

  struct PerHostData {
    gptr<Graph> graph;
    gptr<Messages> messages;
    gptr<Terminator> terminator;
    Operator origOp;
    Galois::LargeArray<Operator,false> ops;
    Galois::LargeArray<int,false> scoreboard;
    Context<Graph,Operator> context;
    Graph* pGraph;
    Messages* pMessages;
    Terminator* pTerminator;

    PerHostData(gptr<Graph> g, gptr<Messages> m, gptr<Terminator> t): graph(g), messages(m), terminator(t) { 
      //Runtime::allocatePerHost(this);
    }

    PerHostData(DeSerializeBuffer& s) {
      gDeserialize(s, graph, messages, terminator);
    }
    
    void allocate(size_t s) {
      pGraph = &*graph;
      pMessages = &*messages;
      pTerminator = &*terminator;
      ops.allocate(s);
      scoreboard.allocate(s);
    }

    typedef int tt_has_serialize;
    typedef int tt_is_persistent;

    void serialize(SerializeBuffer& s) const { gSerialize(s, graph, messages, terminator); }
  };

  gptr<PerHostData> data;
  gptr<Messages> messages;
  gptr<Terminator> terminator;
  bool hasInitialMessages;
  
  struct Gather {
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    gptr<PerHostData> origPointer;
    PerHostData* pData;

    Gather() { }
    Gather(gptr<PerHostData> p): origPointer(p) { pData = &*origPointer; }

    typedef int tt_has_serialize;

    void serialize(SerializeBuffer& s) const { gSerialize(s, origPointer); }
    void deserialize(DeSerializeBuffer& s) { gDeserialize(s, origPointer); pData = &*origPointer; }

    void operator()(GNode node, Galois::UserContext<GNode>&) {
      size_t id = pData->pGraph->idFromNode(node);
      Operator& op = pData->ops[id];
      gather_type sum;

      if (needs_gather_in_edges<Operator>::value) {
        for (in_edge_iterator ii = pData->pGraph->in_edge_begin(node, Galois::MethodFlag::NONE),
            ei = pData->pGraph->in_edge_end(node, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          op.gather(*pData->pGraph, node, pData->pGraph->getInEdgeDst(ii), node, sum, pData->pGraph->getInEdgeData(ii));
        }
      }

      if (needs_gather_out_edges<Operator>::value) {
        for (edge_iterator ii = pData->pGraph->edge_begin(node, Galois::MethodFlag::NONE), 
            ei = pData->pGraph->edge_end(node, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          op.gather(*pData->pGraph, node, node, pData->pGraph->getEdgeDst(ii), sum, pData->pGraph->getEdgeData(ii));
        }
      }

      op.apply(*pData->pGraph, node, sum);
    }
  };

  template<typename Container>
  struct Scatter {
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    gptr<PerHostData> origPointer;
    gptr<Container> next;
    PerHostData* pData;

    Scatter() { }
    Scatter(gptr<PerHostData> p, gptr<Container> n): origPointer(p), next(n) { init(); }

    typedef int tt_has_serialize;

    void serialize(SerializeBuffer& s) const { gSerialize(s, origPointer); gSerialize(s, next); }
    void deserialize(DeSerializeBuffer& s) { gDeserialize(s, origPointer); gDeserialize(s, next); init(); }

    void init() {
      pData = &*origPointer;
      assert(0 && "fixme");
      //pData->context.initializeForSync(pData->pGraph, &pData->scoreboard, &*next, pData->pMessages);
    }
    
    void operator()(GNode node, Galois::UserContext<GNode>&) {
      size_t id = pData->pGraph->idFromNode(node);

      Operator& op = pData->ops[id];
      
      if (!op.needsScatter(*pData->pGraph, node))
        return;

      if (needs_scatter_in_edges<Operator>::value) {
        for (in_edge_iterator ii = pData->pGraph->in_edge_begin(node, Galois::MethodFlag::NONE),
            ei = pData->pGraph->in_edge_end(node, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          op.scatter(*pData->pGraph, node, pData->pGraph->getInEdgeDst(ii), node,
              pData->context, pData->pGraph->getInEdgeData(ii));
        }
      }
      if (needs_scatter_out_edges<Operator>::value) {
        for (edge_iterator ii = pData->pGraph->edge_begin(node, Galois::MethodFlag::NONE), 
            ei = pData->pGraph->edge_end(node, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          op.scatter(*pData->pGraph, node, node, pData->pGraph->getEdgeDst(ii),
              pData->context, pData->pGraph->getEdgeData(ii));
        }
      }
    }
  };

  template<bool IsFirst>
  struct Initialize {
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    gptr<PerHostData> origPointer;
    PerHostData* pData;

    Initialize() { }
    Initialize(gptr<PerHostData> p): origPointer(p) { pData = &*origPointer; }

    typedef int tt_has_serialize;

    void serialize(SerializeBuffer& s) const { gSerialize(s, origPointer); }
    void deserialize(DeSerializeBuffer& s) { gDeserialize(s, origPointer); pData = &*origPointer; }

    message_type getMessage(size_t id) {
      if (NeedMessages) {
        return pData->pMessages->get(id);
      } else {
        return message_type();
      }
    }

    void operator()(GNode n, Galois::UserContext<GNode>&) {
      size_t id = pData->pGraph->idFromNode(n);
      if (!IsFirst) {
        pData->scoreboard[id] = 0;
      }

      Operator& op = pData->ops[id];
      op = pData->origOp;
      op.init(*pData->pGraph, n, getMessage(id));

      // Hoist as much as work as possible behind first barrier
      if (needs_gather_in_edges<Operator>::value || needs_gather_out_edges<Operator>::value)
        return;
      
      gather_type sum;
      op.apply(*pData->pGraph, n, sum);

      if (needs_scatter_in_edges<Operator>::value || needs_scatter_out_edges<Operator>::value)
        return;
    }
  };

  template<bool IsFirst,typename Container1,typename Container2>
  struct Check {
    typedef int tt_does_not_need_parallel_push;
    typedef int tt_does_not_need_aborts;

    gptr<PerHostData> data;
    gptr<Container1> cur;
    gptr<Container2> next;

    Check() { }
    Check(gptr<PerHostData> d, gptr<Container1> c, gptr<Container2> n): data(d), cur(c), next(n) { }

    typedef int tt_has_serialize;

    void serialize(SerializeBuffer& s) const { gSerialize(s, data, cur, next); }
    void deserialize(DeSerializeBuffer& s) { gDeserialize(s, data, cur, next); }

    template<bool B> void clear(typename std::enable_if<B>::type* = 0) { }
    template<bool B> void clear(typename std::enable_if<!B>::type* = 0) { 
      cur->clear(); 
    }

    void operator()(unsigned tid, unsigned) {
      if (tid == 0) {
        data->pTerminator->get() = !next->empty();
        clear<IsFirst>();
      }
    }
  };

  struct Allocate {
    gptr<PerHostData> data;
    size_t size;

    Allocate() { }
    Allocate(gptr<PerHostData> d, size_t s): data(d), size(s) { }

    void operator()(unsigned tid, unsigned) {
      if (tid == 0)
        data->allocate(size);
    }

    typedef int tt_has_serialize;
    void serialize(SerializeBuffer& s) const { gSerialize(s, data, size); }
    void deserialize(DeSerializeBuffer& s) { gDeserialize(s, data, size); }
  };

  template<bool IsFirst,typename Container1, typename Container2>
  bool executeStep(gptr<PerHostData> p, gptr<Container1> cur, gptr<Container2> next) {
    Galois::for_each_local<WL>(cur, Initialize<IsFirst>(p));
    
    if (needs_gather_in_edges<Operator>::value || needs_gather_out_edges<Operator>::value 
        || needs_scatter_in_edges<Operator>::value || needs_scatter_out_edges<Operator>::value) {
      // XXX: Need to update node data
    }

    if (needs_gather_in_edges<Operator>::value || needs_gather_out_edges<Operator>::value) {
      Galois::for_each_local<WL>(cur, Gather(p));
    }

    if (NeedMessages) {
      data->pMessages->doReset();
    }

    if (needs_scatter_in_edges<Operator>::value || needs_scatter_out_edges<Operator>::value) {
      Galois::for_each_local<WL>(cur, Scatter<Container2>(p, next));
    }

    if (NeedMessages) {
      data->pMessages->doAllReduce();
    }

    Galois::on_each(Check<IsFirst,Container1,Container2>(p, cur, next));
    
    bool retval = data->pTerminator->doReduce();
    data->pTerminator->doBroadcast(false);
    return retval;
  }

public:
  SyncEngine(Graph& g): hasInitialMessages(false) {
    messages = gptr<Messages>(new Messages);
    terminator = gptr<Terminator>(new Terminator);
    data = gptr<PerHostData>(new PerHostData(gptr<Graph>(&g), messages, terminator));

    size_t size = g.size();
    Galois::on_each(Allocate(data, size));
    if (NeedMessages)
      data->messages->allocate(size);
  }

  ~SyncEngine() {
    // XXX cannot deallocate master pointer otherwise new pointers may point
    // to garbage
    //Runtime::deallocatePerHost(data);
    //Runtime::deallocatePerHost(messages);
    //Runtime::deallocatePerHost(terminator);
  }

  void signal(GNode node, const message_type& msg) {
    if (NeedMessages) {
      assert(Runtime::LL::getTID() == 0);
      hasInitialMessages = true;
      data->pMessages->update(data.pGraph->idFromNode(node), msg);
    }
  }

  void execute() {
    Galois::Statistic rounds("GraphLabRounds");
    typedef Runtime::PerHost<Galois::InsertBag<GNode>> PerHostBag;
    
    gptr<PerHostBag> next(new PerHostBag);
    gptr<PerHostBag> cur(new PerHostBag);

    if (NeedMessages && hasInitialMessages)
      data->pMessages->doBroadcast();
    Runtime::Distributed::distWait();
    hasInitialMessages = false;

    rounds += 1;
    bool more = executeStep<true>(data, data->graph, next);
    while (more) {
      std::swap(cur, next);
      more = executeStep<false>(data, cur, next);
      rounds += 1;
    }

    // XXX cannot deallocate master pointer otherwise new pointers may point
    // to garbage
    //deallocatePerHost(next);
    //deallocatePerHost(cur);
  }
};

}
}
#endif
