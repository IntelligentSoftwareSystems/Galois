#ifndef GALOIS_RUNTIME_FOREACHTRAITS_H
#define GALOIS_RUNTIME_FOREACHTRAITS_H

#include "galois/gtuple.h"
#include "galois/Traits.h"
#include "galois/TypeTraits.h"

#include <tuple>

namespace galois {
namespace runtime {
namespace DEPRECATED {

template<typename FunctionTy>
class ForEachTraits {
  // special_decay of std::ref(t) is T& so apply twice
  typedef typename galois::DEPRECATED::special_decay<typename galois::DEPRECATED::special_decay<FunctionTy>::type>::type Fn;
public:
  enum {
    NeedsStats = !galois::DEPRECATED::does_not_need_stats<Fn>::value,
    NeedsBreak = galois::DEPRECATED::needs_parallel_break<Fn>::value,
    NeedsPush = !galois::DEPRECATED::does_not_need_push<Fn>::value,
    NeedsPIA = galois::DEPRECATED::needs_per_iter_alloc<Fn>::value,
    NeedsAborts = !galois::DEPRECATED::does_not_need_aborts<Fn>::value
  };
};

template<typename FunctionTy>
struct ExtractForEachTraits {
  typedef typename true_indices<
      !ForEachTraits<FunctionTy>::NeedsAborts,
      !ForEachTraits<FunctionTy>::NeedsStats,
      !ForEachTraits<FunctionTy>::NeedsPush,
      ForEachTraits<FunctionTy>::NeedsBreak,
      ForEachTraits<FunctionTy>::NeedsPIA>::type NotDefault;
  typedef std::tuple<
      no_conflicts_tag, 
      no_stats_tag,
      no_pushes_tag,
      parallel_break_tag,
      per_iter_alloc_tag> Tags;
  typedef std::tuple<
      no_conflicts,
      no_stats,
      no_pushes,
      parallel_break,
      per_iter_alloc> Values;
  typedef typename tuple_elements<Tags, NotDefault>::type tags_type;
  typedef typename tuple_elements<Values, NotDefault>::type values_type;
};

}
}
}

#endif // GALOIS_RUNTIME_FOREACHTRAITS_H
