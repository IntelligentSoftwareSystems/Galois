#ifndef GALOIS_GRAPHLABEXECUTOR_H
#define GALOIS_GRAPHLABEXECUTOR_H

#include "galois/Bag.h"

#include <boost/mpl/has_xxx.hpp>

namespace galois {
//! Implementation of GraphLab v2/PowerGraph DSL in Galois
namespace graphLab {

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
  typedef galois::substrate::PerPackageStorage<MyMessages> Messages;

  galois::UserContext<WorkItem>* ctx;
  Graph* graph;
  galois::LargeArray<int>* scoreboard;
  galois::InsertBag<GNode>* next;
  Messages* messages;

  Context(galois::UserContext<WorkItem>* c): ctx(c) { }

  Context(Graph* g, galois::LargeArray<int>* s, galois::InsertBag<GNode>* n, Messages* m):
    graph(g), scoreboard(s), next(n), messages(m) { }

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
    galois::InsertBag<WorkItem>& bag;

    Initialize(AsyncEngine* s, galois::InsertBag<WorkItem>& b): self(s), bag(b) { }

    void operator()(GNode n) const {
      bag.push(WorkItem(n, message_type()));
    }
  };

  struct Process {
    AsyncEngine* self;
    Process(AsyncEngine* s): self(s) { }

    void operator()(const WorkItem& item, galois::UserContext<WorkItem>& ctx) {
      Operator op(self->origOp);

      GNode node = item.first;
      message_type msg = item.second;
      
      if (needs_gather_in_edges<Operator>::value || needs_scatter_in_edges<Operator>::value) {
        self->graph.in_edge_begin(node, galois::MethodFlag::WRITE);
      }

      if (needs_gather_out_edges<Operator>::value || needs_scatter_out_edges<Operator>::value) {
        self->graph.edge_begin(node, galois::MethodFlag::WRITE);
      }

      op.init(self->graph, node, msg);
      
      gather_type sum;
      if (needs_gather_in_edges<Operator>::value) {
        for (in_edge_iterator ii = self->graph.in_edge_begin(node, galois::MethodFlag::UNPROTECTED),
            ei = self->graph.in_edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
          op.gather(self->graph, node, self->graph.getInEdgeDst(ii), node, sum, self->graph.getInEdgeData(ii));
        }
      }
      if (needs_gather_out_edges<Operator>::value) {
        for (edge_iterator ii = self->graph.edge_begin(node, galois::MethodFlag::UNPROTECTED), 
            ei = self->graph.edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
          op.gather(self->graph, node, node, self->graph.getEdgeDst(ii), sum, self->graph.getEdgeData(ii));
        }
      }

      op.apply(self->graph, node, sum);

      if (!op.needsScatter(self->graph, node))
        return;

      Context<Graph,Operator> context(&ctx);

      if (needs_scatter_in_edges<Operator>::value) {
        for (in_edge_iterator ii = self->graph.in_edge_begin(node, galois::MethodFlag::UNPROTECTED),
            ei = self->graph.in_edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
          op.scatter(self->graph, node, self->graph.getInEdgeDst(ii), node, context, self->graph.getInEdgeData(ii));
        }
      }
      if (needs_scatter_out_edges<Operator>::value) {
        for (edge_iterator ii = self->graph.edge_begin(node, galois::MethodFlag::UNPROTECTED), 
            ei = self->graph.edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
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
    typedef galois::worklists::dChunkedFIFO<256> WL;

    galois::InsertBag<WorkItem> bag;
    galois::do_all_local(graph, Initialize(this, bag));
    galois::for_each_local(bag, Process(this), galois::wl<WL>());
  }
};

template<typename Graph, typename Operator>
class SyncEngine {
  typedef typename Operator::message_type message_type;
  typedef typename Operator::gather_type gather_type;
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::in_edge_iterator in_edge_iterator;
  typedef typename Graph::edge_iterator edge_iterator;
  static const bool NeedMessages = !std::is_same<EmptyMessage,message_type>::value;
  typedef galois::worklists::dChunkedFIFO<256> WL;
  typedef std::pair<int,message_type> Message;
  typedef std::deque<Message> MyMessages;
  typedef galois::substrate::PerPackageStorage<MyMessages> Messages;

  Graph& graph;
  Operator origOp;
  galois::LargeArray<Operator> ops;
  Messages messages;
  galois::LargeArray<int> scoreboard;
  galois::InsertBag<GNode> wls[2];
  galois::substrate::SimpleLock lock;

  struct Gather {
    SyncEngine* self;
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    Gather(SyncEngine* s): self(s) { }
    void operator()(GNode node, galois::UserContext<GNode>&) {
      size_t id = self->graph.idFromNode(node);
      Operator& op = self->ops[id];
      gather_type sum;

      if (needs_gather_in_edges<Operator>::value) {
        for (in_edge_iterator ii = self->graph.in_edge_begin(node, galois::MethodFlag::UNPROTECTED),
            ei = self->graph.in_edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
          op.gather(self->graph, node, self->graph.getInEdgeDst(ii), node, sum, self->graph.getInEdgeData(ii));
        }
      }

      if (needs_gather_out_edges<Operator>::value) {
        for (edge_iterator ii = self->graph.edge_begin(node, galois::MethodFlag::UNPROTECTED), 
            ei = self->graph.edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
          op.gather(self->graph, node, node, self->graph.getEdgeDst(ii), sum, self->graph.getEdgeData(ii));
        }
      }

      op.apply(self->graph, node, sum);
    }
  };

  template<typename Container>
  struct Scatter {
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    SyncEngine* self;
    Context<Graph,Operator> context;

    Scatter(SyncEngine* s, Container& next):
      self(s),
      context(&self->graph, &self->scoreboard, &next, NeedMessages ? &self->messages : 0) 
      { }

    void operator()(GNode node, galois::UserContext<GNode>&) {
      size_t id = self->graph.idFromNode(node);

      Operator& op = self->ops[id];
      
      if (!op.needsScatter(self->graph, node))
        return;

      if (needs_scatter_in_edges<Operator>::value) {
        for (in_edge_iterator ii = self->graph.in_edge_begin(node, galois::MethodFlag::UNPROTECTED),
            ei = self->graph.in_edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
          op.scatter(self->graph, node, self->graph.getInEdgeDst(ii), node, context, self->graph.getInEdgeData(ii));
        }
      }
      if (needs_scatter_out_edges<Operator>::value) {
        for (edge_iterator ii = self->graph.edge_begin(node, galois::MethodFlag::UNPROTECTED), 
            ei = self->graph.edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
          op.scatter(self->graph, node, node, self->graph.getEdgeDst(ii), context, self->graph.getEdgeData(ii));
        }
      }
    }
  };

  template<bool IsFirst>
  struct Initialize {
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    SyncEngine* self;
    Initialize(SyncEngine* s): self(s) { }

    void allocateMessages() {
      unsigned tid = galois::substrate::ThreadPool::getTID();
      if (!galois::substrate::ThreadPool::isLeader() || tid == 0)
        return;
      MyMessages& m = *self->messages.getLocal();
      self->lock.lock();
      m.resize(self->graph.size());
      self->lock.unlock();
    }

    message_type getMessage(size_t id) {
      message_type ret;
      if (NeedMessages) {
        auto& tp = galois::substrate::getThreadPool();
        for (unsigned int i = 0; i < self->messages.size(); ++i) {
          if (!tp.isLeader(i))
            continue;
          MyMessages& m = *self->messages.getRemote(i);
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

    void operator()(GNode n, galois::UserContext<GNode>&) {
      size_t id = self->graph.idFromNode(n);
      if (IsFirst && NeedMessages) {
        allocateMessages();
      } else if (!IsFirst) {
        self->scoreboard[id] = 0;
      }

      Operator& op = self->ops[id];
      op = self->origOp;
      op.init(self->graph, n, getMessage(id));

      // Hoist as much as work as possible behind first barrier
      if (needs_gather_in_edges<Operator>::value || needs_gather_out_edges<Operator>::value)
        return;
      
      gather_type sum;
      op.apply(self->graph, n, sum);

      if (needs_scatter_in_edges<Operator>::value || needs_scatter_out_edges<Operator>::value)
        return;
    }
  };

  template<bool IsFirst,typename Container1, typename Container2>
  void executeStep(Container1& cur, Container2& next) {
    galois::for_each_local(cur, Initialize<IsFirst>(this), galois::wl<WL>());
    
    if (needs_gather_in_edges<Operator>::value || needs_gather_out_edges<Operator>::value) {
      galois::for_each_local(cur, Gather(this), galois::wl<WL>());
    }

    if (needs_scatter_in_edges<Operator>::value || needs_scatter_out_edges<Operator>::value) {
      galois::for_each_local(cur, Scatter<Container2>(this, next), galois::wl<WL>());
    }
  }

public:
  SyncEngine(Graph& g, Operator op): graph(g), origOp(op) {
    ops.create(graph.size());
    scoreboard.create(graph.size());
    if (NeedMessages)
      messages.getLocal()->resize(graph.size());
  }

  void signal(GNode node, const message_type& msg) {
    if (NeedMessages) {
      MyMessages& m = *messages.getLocal();
      m[graph.idFromNode(node)].second = msg;
    }
  }

  void execute() {
    galois::Statistic rounds("GraphLabRounds");
    galois::InsertBag<GNode>* next = &wls[0];
    galois::InsertBag<GNode>* cur = &wls[1];

    executeStep<true>(graph, *next);
    rounds += 1;
    while (!next->empty()) {
      std::swap(cur, next);
      executeStep<false>(*cur, *next);
      rounds += 1;
      cur->clear();
    }
  }
};

}
}
#endif
