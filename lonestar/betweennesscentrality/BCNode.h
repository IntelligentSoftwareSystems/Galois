#ifndef _BCNODE_H_
#define _BCNODE_H_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <limits>

#include "galois/substrate/SimpleLock.h"
#include "llvm/ADT/SmallVector.h"
#include "control.h"

constexpr unsigned infinity = std::numeric_limits<unsigned>::max() / 2;

template <bool UseMarking=false, bool Concurrent=true>
struct BCNode {
  using LockType = typename std::conditional<Concurrent, 
                                             galois::substrate::SimpleLock,
                                             char>::type;
  LockType spinLock;

  using predTY = galois::gstl::Vector<uint32_t>;
  predTY preds;

  unsigned distance;
  unsigned nsuccs;

  ShortPathType sigma; 
  double delta;
  double bc;
  int mark;

  BCNode() 
    : spinLock(), preds(), distance(infinity), nsuccs(0),
      sigma(0), delta(0), bc(0), mark(0) {}
  
  /**
   * @param a Node to check if predecessor of this node
   * @returns true if node a is in predecessors of this node
   */
  bool predsContain(const BCNode* a) const {
    typename predTY::const_iterator it = preds.end();
    return (std::find(preds.begin(), preds.end(), a) != it); 
  }

  template<bool C = Concurrent, typename std::enable_if<C>::type* = nullptr>
  void lock() {
    spinLock.lock();
  }

  template<bool C = Concurrent, typename std::enable_if<C>::type* = nullptr>
  bool try_lock() {
    return spinLock.try_lock();
  }

  template<bool C = Concurrent, typename std::enable_if<C>::type* = nullptr>
  void unlock() {
    spinLock.unlock();
  }

  // below are no-ops for when concurrent is false
  template<bool C = Concurrent, typename std::enable_if<!C>::type* = nullptr>
  void lock() {
    // no-op
  }

  template<bool C = Concurrent, typename std::enable_if<!C>::type* = nullptr>
  bool try_lock() {
    return true;
  }

  template<bool C = Concurrent, typename std::enable_if<!C>::type* = nullptr>
  void unlock() {
    // no-op
  }


  /**
   * Return node as string.
   *
   * WARNING NON SCALABLE FUNCTION
   */
  std::string toString() const {
    std::ostringstream s;

    s << " distance: " << distance << " sigma: " << sigma << " bc: " 
      << bc << " nsuccs: " << nsuccs << " npreds: " << preds.size();

    return s.str();
  }

  /**
   * Reset everything but the BC value
   */
  void reset() {
    preds.clear();
    distance = infinity;
    nsuccs = 0;
    sigma = 0;
    delta = 0;
    mark = 0;
  }

  /**
   * Sanity check to make sure node is reset
   */
  void checkClear() const {
    if (!preds.empty() || nsuccs != 0 || sigma != 0 || delta != 0)
      galois::gWarn("Problem, node not clear");

    assert(preds.empty());
    assert(distance == infinity);
    assert(nsuccs == 0 && sigma == 0 && delta == 0);
  }  

  /**
   * Initialize this node as the source
   */
  void initAsSource() {
    distance = 0;
    sigma = 1;
  }

  /**
   * Mark this as 0.
   */
  template<bool M = UseMarking, typename std::enable_if<M>::type* = nullptr>
  void markOut() {
    if (Concurrent) {
    __sync_fetch_and_and(&mark, 0);
    } else {
      mark = 0;
    }
  }

  template<bool M = UseMarking, typename std::enable_if<!M>::type* = nullptr>
  void markOut() {
    // no-op
  }

  /**
   * @returns true if mark is set to 1
   */
  template<bool M = UseMarking, typename std::enable_if<M>::type* = nullptr>
  int isAlreadyIn() {
    if (Concurrent) {
      return __sync_fetch_and_or(&mark, 1);
    } else {
      int retval = mark;
      mark = 1;
      return retval;
    }
  }

  /**
   * @returns 0
   */
  template<bool M = UseMarking, typename std::enable_if<!M>::type* = nullptr>
  int isAlreadyIn() {
    return 0;
  }

};
#endif // _BCNODE_H_
