#ifndef PAGERANK_IKDG_H
#define PAGERANK_IKDG_H

#include <type_traits>

struct NodeData : public PData {
  std::atomic<int> mark;
  NodeData(void) {}
  NodeData(unsigned id, unsigned outdegree) : PData(outdegree) {}
};

typedef typename galois::graphs::LC_CSR_Graph<NodeData, void>::with_numa_alloc<
    true>::type InnerGraph;

template <bool UseAddRemove>
class PageRankIKDG : public PageRankBase<InnerGraph> {
  typedef PageRankBase<InnerGraph> Super;
  typedef typename Super::GNode GNode;

  galois::InsertBag<GNode> bags[2];

protected:
  struct ApplyOperator;

  struct LocalState {
    float value;
    bool mod;
    LocalState(ApplyOperator& self, galois::PerIterAllocTy& alloc) {}
  };

  void applyOperator(GNode src, galois::UserContext<GNode>& ctx,
                     galois::InsertBag<GNode>& next) {
    LocalState* localState = (LocalState*)ctx.getLocalState();
    if (ctx.isFirstPass()) {
      float v    = graph.getData(src).value;
      double sum = 0;
      for (auto jj : graph.in_edges(src, galois::MethodFlag::READ)) {
        GNode dst   = graph.getInEdgeDst(jj);
        auto& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sum += ddata.value / ddata.outdegree;
      }
      float newV        = (1.0 - alpha) * sum + alpha;
      localState->value = newV;
      localState->mod   = std::fabs(v - newV) > tolerance;
      return;
    }

    Super::numIter += 1;

    if (!localState->mod)
      return;

    auto& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    float value = localState->value;

    sdata.value = value;
    if (sdata.mark.load(std::memory_order_relaxed))
      sdata.mark.store(0, std::memory_order_relaxed);
    for (auto jj : graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
      GNode dst = graph.getEdgeDst(jj);
      next.push(dst);
    }
  }

  struct ApplyOperator {
    struct DeterministicId {
      uintptr_t operator()(const GNode& x) const { return x; }
    };

    typedef std::tuple<galois::per_iter_alloc, galois::intent_to_read,
                       galois::local_state<LocalState>,
                       galois::det_id<DeterministicId>>
        ikdg_function_traits;
    typedef std::tuple<galois::per_iter_alloc, galois::fixed_neighborhood,
                       // galois::intent_to_read<>, // TODO enable
                       galois::local_state<LocalState>,
                       galois::det_id<DeterministicId>>
        add_remove_function_traits;
    typedef
        typename std::conditional<UseAddRemove, add_remove_function_traits,
                                  ikdg_function_traits>::type function_traits;

    PageRankIKDG& outer;
    galois::InsertBag<GNode>& next;

    ApplyOperator(PageRankIKDG& o, galois::InsertBag<GNode>& n)
        : outer(o), next(n) {}

    void operator()(GNode src, galois::UserContext<GNode>& ctx) {
      outer.applyOperator(src, ctx, next);
    }
  };

  struct AddToBag {
    galois::InsertBag<GNode>& bag;
    void operator()(GNode x) const { bag.push(x); }
  };

  struct AddToNewBag {
    galois::InsertBag<GNode>& bag;
    InnerGraph& graph;
    void operator()(GNode x) const {
      std::atomic<int>& m = graph.getData(x).mark;
      int v               = 0;
      if (m.compare_exchange_strong(v, 1))
        bag.push(x);
    }
  };

  virtual void runPageRank() {
    galois::do_all(graph, AddToBag{bags[0]});

    while (!bags[0].empty()) {
      galois::for_each(graph.begin(), graph.end(),
                       ApplyOperator(*this, bags[1]),
                       galois::loopname("page-rank-ikdg"),
                       galois::wl<galois::worklists::Deterministic<>>());
      bags[0].clear();
      galois::do_all(bags[1], AddToNewBag{bags[0], graph});
    }
  }
};

#endif
