/** Stable Iterator worklist -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @description
 * This dereferences iterators lazily.  This is only safe if they are not
 * invalidated by the operator
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

namespace Galois {
namespace WorkList {

template<typename IterTy, bool steal = false>
class LazyIter {
public:
  typedef typename std::iterator_traits<IterTy>::value_type value_type;

private:
  struct state {
    IterTy localBegin;
    IterTy localEnd;
    IterTy stealBegin;
    IterTy stealEnd;
    Runtime::LL::SimpleLock<true> stealLock;
    bool stealAvail;
    unsigned int nextVictim;

    void resetAvail() {
      if (stealBegin == stealEnd)
	stealAvail = true;
    }

    void populateSteal() {
      if (steal && localBegin != localEnd) {// && std::distance(localBegin, localEnd) > 1) {
	stealLock.lock();
	stealEnd = localEnd;
	stealBegin = localEnd = Galois::split_range(localBegin, localEnd);
	resetAvail();
	stealLock.unlock();
      }
    }
  };

  Runtime::PerThreadStorage<state> TLDS;

  bool doSteal(state& dst, state& src) {
    if (src.stealAvail) {
      src.stealLock.lock();
      if (src.stealBegin != src.stealEnd) {
	dst.localBegin = src.stealBegin;
	src.stealBegin = dst.localEnd = Galois::split_range(src.stealBegin, src.stealEnd);
	src.resetAvail();
      }
      src.stealLock.unlock();
    }
    return dst.localBegin != dst.localEnd;
  }

  //pop already failed, try again with stealing
  boost::optional<value_type> pop_steal(state& data) {
    //only try stealing one
    if (doSteal(data, *TLDS.getRemote(data.nextVictim)))
      return *data.localBegin++;
    ++data.nextVictim;
    data.nextVictim %= Runtime::activeThreads;
    return boost::optional<value_type>();
  }

public:

  //! change the concurrency flag
  template<bool newconcurrent>
  using rethread = LazyIter<IterTy, steal>;
  
  //! change the type the worklist holds
  template<typename Tnew>
  using retype = LazyIter<IterTy, steal>;

  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename RangeTy>
  void push_initial(const RangeTy& r) {
    state& data = *TLDS.getLocal();
    data.localBegin = r.local_begin();
    data.localEnd = r.local_end();
    data.nextVictim = Runtime::LL::getTID();
    data.populateSteal();
  }

  //! pop a value from the queue.
  boost::optional<value_type> pop() {
    state& data = *TLDS.getLocal();
    if (data.localBegin != data.localEnd)
      return *data.localBegin++;
    if (steal)
      return pop_steal(data);
    return boost::optional<value_type>();
  }

  void push(value_type& val) {
    abort();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    abort();
  }

};


}
}
