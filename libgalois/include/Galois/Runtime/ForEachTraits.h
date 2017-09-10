/** Traits of the Foreach loop body functor -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Traits of the for_each loop body functor.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_FOREACHTRAITS_H
#define GALOIS_RUNTIME_FOREACHTRAITS_H

#include "Galois/gtuple.h"
#include "Galois/Traits.h"
#include "Galois/TypeTraits.h"

#include <tuple>

namespace Galois {
namespace Runtime {
namespace DEPRECATED {

template<typename FunctionTy>
class ForEachTraits {
  // special_decay of std::ref(t) is T& so apply twice
  typedef typename Galois::DEPRECATED::special_decay<typename Galois::DEPRECATED::special_decay<FunctionTy>::type>::type Fn;
public:
  enum {
    NeedsStats = !Galois::DEPRECATED::does_not_need_stats<Fn>::value,
    NeedsBreak = Galois::DEPRECATED::needs_parallel_break<Fn>::value,
    NeedsPush = !Galois::DEPRECATED::does_not_need_push<Fn>::value,
    NeedsPIA = Galois::DEPRECATED::needs_per_iter_alloc<Fn>::value,
    NeedsAborts = !Galois::DEPRECATED::does_not_need_aborts<Fn>::value
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
      does_not_need_aborts_tag, 
      no_stats_tag,
      does_not_need_push_tag,
      needs_parallel_break_tag,
      needs_per_iter_alloc_tag> Tags;
  typedef std::tuple<
      does_not_need_aborts,
      no_stats,
      does_not_need_push,
      needs_parallel_break,
      needs_per_iter_alloc> Values;
  typedef typename tuple_elements<Tags, NotDefault>::type tags_type;
  typedef typename tuple_elements<Values, NotDefault>::type values_type;
};

}
}
}

#endif // GALOIS_RUNTIME_FOREACHTRAITS_H
