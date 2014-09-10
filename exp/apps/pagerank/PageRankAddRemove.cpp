#include "PageRankDet.h"

struct NodeData: public PData {
  std::atomic<int> mark;
  NodeData (void) {}
  NodeData (unsigned id, unsigned outdegree): PData (outdegree) {}
};

typedef typename Galois::Graph::LC_CSR_Graph<NodeData, void>
  ::with_numa_alloc<true>::type
//  ::with_no_lockable<true>::type 
  InnerGraph;

class PageRankAddRemove: public PageRankBase<InnerGraph> {
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
      for (auto jj : graph.in_edges(src, Galois::MethodFlag::ALL)) {
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
    sdata.mark.store(0, std::memory_order_relaxed);
    for (auto jj : graph.out_edges(src, Galois::MethodFlag::NONE)) {
      GNode dst = graph.getEdgeDst(jj);
      next.push(dst);
    }
  }

  struct ApplyOperator {
    typedef int tt_needs_per_iter_alloc; // For LocalState
    typedef int tt_has_fixed_neighborhood;

    uintptr_t galoisDeterministicId(const GNode& x) const {
      return x;
    }
    static_assert(Galois::has_deterministic_id<ApplyOperator>::value, "Oops");

    typedef LocalState GaloisDeterministicLocalState;
    static_assert(Galois::has_deterministic_local_state<ApplyOperator>::value, "Oops");

    PageRankAddRemove& outer;
    Galois::InsertBag<GNode>& next;

    void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
      outer.applyOperator(src, ctx, next);
    }
  };

  virtual void runPageRank() {
    Galois::do_all_local(graph, [this](GNode x) {
      bags[0].push(x);
    });

    while (!bags[0].empty()) {
      Galois::for_each_det(graph.begin(), graph.end(),
          ApplyOperator { *this, bags[1] },
          "page-rank-add-remove");
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

int main(int argc, char* argv[]) {
  PageRankAddRemove p;

  return p.run(argc, argv);
}
