#ifndef GALOIS_GRAPHLABEXECUTOR_H
#define GALOIS_GRAPHLABEXECUTOR_H

#include "Galois/Bag.h"
#include "Galois/PerHostStorage.h"
#include "Galois/Runtime/ParallelWorkDistributed.h"

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

template<typename Graph, typename Operator> 
struct Context {
  typedef typename Graph::GraphNode GNode;
  typedef typename Operator::message_type message_type;
  typedef std::pair<GNode,message_type> WorkItem;

private:
  template<typename,typename> friend class AsyncEngine;
  template<typename,typename> friend class SyncEngine;

  typedef std::pair<int,message_type> Message;
  typedef std::deque<Message> MyMessages;
  typedef Galois::Runtime::PerPackageStorage<MyMessages> Messages;

  Galois::UserContext<WorkItem>* ctx;
  Graph* graph;
  Galois::LargeArray<int,false>* scoreboard;
  Galois::PerHostStorage<Galois::InsertBag<GNode>>* next;
  Messages* messages;

  Context() { }

  void initializeForSync(Graph* g, Galois::LargeArray<int,false>* s, Galois::PerHostStorage<Galois::InsertBag<GNode>>* n, Messages* m) {
    graph = g;
    scoreboard = s;
    next = n;
    messages = m;
    ctx = 0;
  }

  void initializeForAsync(Galois::UserContext<WorkItem>* c) {
    ctx = c;  
  }

public:

  void push(GNode node, const message_type& message) {
    if (ctx) {
      ctx->push(WorkItem(node, message));
    } else {
      size_t id = graph->idFromNode(node);
      { 
        int val = (*scoreboard)[id];
        if (val == 0 && __sync_bool_compare_and_swap(&(*scoreboard)[id], 0, 1)) {
          next->push(node);
        }
      }

      if (messages) {
        MyMessages& m = *messages->getLocal();
        int val; 
        while (true) {
          val = m[id].first;
          if (val == 0 && __sync_bool_compare_and_swap(&m[id].first, 0, 1)) {
            m[id].second += message;
            m[id].first = 0;
            return;
          }
        }
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
  typedef std::pair<int,message_type> Message;
  typedef std::deque<Message> MyMessages;
  typedef Galois::Runtime::PerPackageStorage<MyMessages> Messages;

  struct PerHostData {
    gptr<Graph> graph;
    Operator origOp;
    Galois::LargeArray<Operator,false> ops;
    Messages messages;
    Galois::LargeArray<int,false> scoreboard;
    Galois::Runtime::LL::SimpleLock<true> lock;
    Context<Graph,Operator> context;
    Graph* pGraph;
    size_t size;

    PerHostData(gptr<Graph> g): graph(g) { allocate(graph->size()); pGraph = &*graph; }
    PerHostData(DeSerializeBuffer& s) { gDeserialize(s, graph); allocate(graph->size()); }
    
    ~PerHostData() {
      std::cout << "Deallocate\n";
    }

    typedef int tt_has_serialize;
    typedef int tt_is_persistent;

    void serialize(SerializeBuffer& s) const { gSerialize(s, graph); }
    void deserialize(DeSerializeBuffer& s) { gDeserialize(s, graph); allocate(graph->size()); pGraph = &*graph; }

    void allocate(size_t s) {
      std::cout << "Allocate\n";
      ops.allocate(s);
      scoreboard.allocate(s);
      if (NeedMessages)
        messages.getLocal()->resize(s);
    }
  };

  PerHostData data;
  
  struct Gather {
    typedef int tt_does_not_need_parallel_push;
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
    typedef int tt_does_not_need_parallel_push;
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
      pData->context.initializeForSync(pData->pGraph, &pData->scoreboard, &*next, &pData->messages);
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
    typedef int tt_does_not_need_parallel_push;
    typedef int tt_does_not_need_aborts;

    gptr<PerHostData> origPointer;
    PerHostData* pData;

    Initialize() { }
    Initialize(gptr<PerHostData> p): origPointer(p) { pData = &*origPointer; }

    typedef int tt_has_serialize;

    void serialize(SerializeBuffer& s) const { gSerialize(s, origPointer); }
    void deserialize(DeSerializeBuffer& s) { gDeserialize(s, origPointer); pData = &*origPointer; }

    void allocateMessages() {
      unsigned tid = Galois::Runtime::LL::getTID();
      if (!Galois::Runtime::LL::isPackageLeader(tid) || tid == 0)
        return;
      MyMessages& m = *pData->messages.getLocal();
      pData->lock.lock();
      m.resize(pData->pGraph->size());
      pData->lock.unlock();
    }

    message_type getMessage(size_t id) {
      message_type ret;
      if (NeedMessages) {
        for (unsigned int i = 0; i < pData->messages.size(); ++i) {
          MyMessages& m = *pData->messages.getRemote(i);
          if (m.empty())
            continue;
          ret += m[id].second;
          m[id] = std::make_pair(0, message_type());
          // During initialization, only messages from thread zero
          if (IsFirst)
            break;
        }
      }
      return ret;
    }

    void operator()(GNode n, Galois::UserContext<GNode>&) {
      size_t id = pData->pGraph->idFromNode(n);
      if (IsFirst && NeedMessages) {
        allocateMessages();
      } else if (!IsFirst) {
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

  template<bool IsFirst,typename Container1, typename Container2>
  void executeStep(gptr<PerHostData> p, gptr<Container1> cur, gptr<Container2> next) {
    Galois::for_each_local<WL>(cur, Initialize<IsFirst>(p));
    
    if (needs_gather_in_edges<Operator>::value || needs_gather_out_edges<Operator>::value) {
      Galois::for_each_local<WL>(cur, Gather(p));
    }

    if (needs_scatter_in_edges<Operator>::value || needs_scatter_out_edges<Operator>::value) {
      Galois::for_each_local<WL>(cur, Scatter<Container2>(p, next));
    }
  }

public:
  SyncEngine(Graph* g): data(gptr<Graph>(g)) { }
  SyncEngine(gptr<Graph> g): data(g) { }

  void signal(GNode node, const message_type& msg) {
    if (NeedMessages) {
      MyMessages& m = *data.messages.getLocal();
      m[data.pGraph->idFromNode(node)].second = msg;
    }
  }

  void execute() {
    Galois::Statistic rounds("GraphLabRounds");
    typedef PerHostStorage<Galois::InsertBag<GNode>> PerHostBag;
    
    PerHostBag wls[2];
    gptr<PerHostBag> next(&wls[0]);
    gptr<PerHostBag> cur(&wls[1]);
    gptr<PerHostData> p(&data);

    executeStep<true>(p, data.graph, next);
    rounds += 1;
    while (!next->empty()) { // XXX check
      std::swap(cur, next);
      executeStep<false>(p, cur, next);
      rounds += 1;
      cur->clear();
    }

    deallocatePerHost(next);
    deallocatePerHost(cur);
    deallocatePerHost(p);
  }
};

}
}
#endif
