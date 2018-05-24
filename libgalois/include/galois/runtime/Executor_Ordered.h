#ifndef GALOIS_RUNTIME_EXECUTOR_ORDERED_H
#define GALOIS_RUNTIME_EXECUTOR_ORDERED_H

namespace galois {
namespace runtime {

// TODO(ddn): Pull in and integrate in executors from exp

#if 0
template <typename NhFunc, typename OpFunc>
struct OrderedTraits {
  static const bool NeedsPush = !galois::DEPRECATED::does_not_need_push<OpFunc>::value;
  static const bool HasFixedNeighborhood = galois::DEPRECATED::has_fixed_neighborhood<NhFunc>::value;
};
#endif

template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_impl(Iter beg, Iter end, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname) {
#if 0
  if (!OrderedTraits<NhFunc, OpFunc>::NeedsPush && OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    // TODO: Remove-only/DAG executor
    GALOIS_DIE("Remove-only executor not implemented yet");
  } else if (OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    for_each_ordered_lc (beg, end, cmp, nhFunc, opFunc, loopname);
  } else {
    for_each_ordered_2p (beg, end, cmp, nhFunc, opFunc, loopname);
  }
#else
  GALOIS_DIE("not yet implemented");
#endif
}


template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc, typename StableTest>
void for_each_ordered_impl(Iter beg, Iter end, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const StableTest& stabilityTest, const char* loopname) {
#if 0
  if (!OrderedTraits<NhFunc, OpFunc>::NeedsPush && OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    GALOIS_DIE("no-adds + fixed-neighborhood == stable-source");
  } else if (OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    for_each_ordered_lc (beg, end, cmp, nhFunc, opFunc, stabilityTest, loopname);
  } else {
    GALOIS_DIE("two-phase executor for unstable-source algorithms not implemented yet");
    // TODO: implement following
    // for_each_ordered_2p (beg, end, cmp, nhFunc, opFunc, stabilityTest, loopname); 
  }
#else
  GALOIS_DIE("not yet implemented");
#endif
}

} // end namespace runtime
} // end namespace galois

#endif
