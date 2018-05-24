#ifndef GALOIS_WORKLIST_WORKSET_H
#define GALOIS_WORKLIST_WORKSET_H

#include "galois/worklists/Chunked.h"
#include "galois/worklists/WorkListHelpers.h"
#include "galois/PriorityQueue.h"
#include "galois/UnorderedSet.h"
#include "galois/TwoLevelSet.h"
#include "galois/worklists/Separator.h"
#include "galois/Timer.h"

namespace galois {
namespace worklists {

namespace internal {

template<typename T,
	 typename Scheduler = dChunkedFIFO<64, T>,
         typename Set = galois::ThreadSafeOrderedSet<T> >
struct WorkSetMaster : private boost::noncopyable {
private:
  Scheduler scheduler;
  Set set;
  galois::Statistic* duplicate;

public:
  typedef T value_type;
  template<typename _T>
  using retype = WorkSetMaster<_T, typename Scheduler::template retype<_T>, typename Set::template retype<_T> >;

  WorkSetMaster() { duplicate = new galois::Statistic("SchedulerDuplicates"); }

  template<typename... Args>
  WorkSetMaster(galois::worklists::Separator dummy, Args... args): scheduler(std::forward<Args>(args)...)
  {
    duplicate = new galois::Statistic("SchedulerDuplicates");
  }

  ~WorkSetMaster() { delete duplicate; }

  void push(const value_type& val) {
    if(set.push(val)) {
      scheduler.push(val);
    } else {
      *duplicate += 1;
    }
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  galois::optional<value_type> pop() {
    auto defaultRetVal = galois::optional<value_type>();
    galois::optional<value_type> retval = scheduler.pop();

    if(retval == defaultRetVal)
      return defaultRetVal;

    set.remove(retval.get());
    return retval;
  }
};

}  // end namespace internal

template<int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedOrderedSetFIFO = internal::WorkSetMaster<T, dChunkedFIFO<ChunkSize,T,Concurrent>, galois::ThreadSafeOrderedSet<T> >;
GALOIS_WLCOMPILECHECK(dChunkedOrderedSetFIFO);

template<int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedUnorderedSetFIFO = internal::WorkSetMaster<T, dChunkedFIFO<ChunkSize,T,Concurrent>, galois::ThreadSafeUnorderedSet<T> >;
GALOIS_WLCOMPILECHECK(dChunkedUnorderedSetFIFO);

template<int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedTwoLevelHashFIFO = internal::WorkSetMaster<T, dChunkedFIFO<ChunkSize,T,Concurrent>, galois::ThreadSafeTwoLevelHash<T> >;
GALOIS_WLCOMPILECHECK(dChunkedTwoLevelHashFIFO);

template<int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedTwoLevelSetFIFO = internal::WorkSetMaster<T, dChunkedFIFO<ChunkSize,T,Concurrent>, galois::ThreadSafeTwoLevelSet<T> >;
GALOIS_WLCOMPILECHECK(dChunkedTwoLevelSetFIFO);

} // end namespace worklists
} // end namespace galois

#endif
