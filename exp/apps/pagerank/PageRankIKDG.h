#ifndef PAGERANK_IKDG_H
#define PAGERANK_IKDG_H

#include <type_traits>

struct NodeData: public PData {
  std::atomic<int> mark;
  NodeData (void) {}
  NodeData (unsigned id, unsigned outdegree): PData (outdegree) {}
};

typedef typename Galois::Graph::LC_CSR_Graph<NodeData, void>
  ::with_numa_alloc<true>::type InnerGraph;

template<bool UseAddRemove>
class PageRankIKDG: public PageRankBase<InnerGraph> {
  typedef PageRankBase<InnerGraph> Super;
  typedef typename Super::GNode GNode;

  Galois::InsertBag<GNode> bags[2];

protected:
  struct ApplyOperator;

  struct LocalState {
    float value;
    bool mod;
    LocalState(ApplyOperator& self, Galois::PerIterAllocTy& alloc) { }
  };

  void applyOperator(GNode src, Galois::UserContext<GNode>& ctx, Galois::InsertBag<GNode>& next) {
    bool used;
    LocalState* localState = (LocalState*) ctx.getLocalState(used);
    if (!used) {
      float v = graph.getData(src).value;
      double sum = 0;
      for (auto jj : graph.in_edges(src, Galois::MethodFlag::ALL | Galois::MethodFlag::INTENT_TO_READ)) {
        GNode dst = graph.getInEdgeDst(jj);
        auto& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
        sum += ddata.value / ddata.outdegree;
      }
      float newV = (1.0 - alpha) * sum + alpha;
      localState->value = newV;
      localState->mod = std::fabs(v - newV) > tolerance;
      return;
    }

    if (!localState->mod) 
      return;

    auto& sdata = graph.getData(src, Galois::MethodFlag::NONE);
    float value = localState->value;
    
    sdata.value = value;
    if (sdata.mark.load(std::memory_order_relaxed))
      sdata.mark.store(0, std::memory_order_relaxed);
    for (auto jj : graph.out_edges(src, Galois::MethodFlag::NONE)) {
      GNode dst = graph.getEdgeDst(jj);
      next.push(dst);
    }
  }

  struct ApplyOperator {
    struct DeterministicId {
      uintptr_t operator()(const GNode& x) const {
        return x;
      }
    };

    typedef std::tuple<
      Galois::needs_per_iter_alloc<>,
      Galois::has_intent_to_read<>,
      Galois::has_deterministic_local_state<LocalState>,
      Galois::has_deterministic_id<DeterministicId>
      > ikdg_function_traits;
    typedef std::tuple<
      Galois::needs_per_iter_alloc<>,
      Galois::has_fixed_neighborhood<>,
      //Galois::has_intent_to_read<>, // TODO enable
      Galois::has_deterministic_local_state<LocalState>,
      Galois::has_deterministic_id<DeterministicId>
      > add_remove_function_traits;
    typedef typename std::conditional<UseAddRemove, add_remove_function_traits, ikdg_function_traits>::type function_traits;

    PageRankIKDG& outer;
    Galois::InsertBag<GNode>& next;

    ApplyOperator(PageRankIKDG& o, Galois::InsertBag<GNode>& n): outer(o), next(n) { }

    void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
      outer.applyOperator(src, ctx, next);
    }
  };

  virtual void runPageRank() {
    Galois::do_all_local(graph, [this](GNode x) {
      bags[0].push(x);
    });

    while (!bags[0].empty()) {
      Galois::for_each(graph.begin(), graph.end(),
          ApplyOperator(*this, bags[1]),
          Galois::loopname("page-rank-ikdg"),
          Galois::wl<Galois::WorkList::Deterministic<>>());
      bags[0].clear();
      Galois::do_all_local(bags[1], [this](GNode x) {
        std::atomic<int>& m = graph.getData(x).mark;
        int v = 0;
        if (m.compare_exchange_strong(v, 1))
          bags[0].push(x);
      });
    }
  }
};

#endif
