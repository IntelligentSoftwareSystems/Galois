/** KDG two phase support -*- C++ -*-
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
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_KDG_TWO_PHASE_SUPPORT_H
#define GALOIS_RUNTIME_KDG_TWO_PHASE_SUPPORT_H

namespace Galois {
namespace Runtime {

template <typename T, typename Cmp>
class TwoPhaseContext: public SimpleRuntimeContext {

  using Base = SimpleRuntimeContext;
  // using NhoodList =  Galois::gdeque<Lockable*, 4>;

  T active;
  const Cmp& cmp;
  bool source = true;

public:

  explicit TwoPhaseContext (const T& x, const Cmp& cmp)
    : 
      Base (true),  // pass true so that Base::acquire invokes virtual subAcquire
      active (x), 
      cmp (cmp),
      source (true) 
  {
    assert (&cmp != nullptr);
  }

  bool isSrc (void) const {
    return source;
  }

  void disableSrc (void) {
    source = false;
  }

  void reset () { 
    source = true;
  }

  const T& getElem () const { return active; }

  T& getElem () { return active; }

  virtual void subAcquire (Lockable* l) {


    if (Base::tryLock (l)) {
      Base::addToNhood (l);
    }

    TwoPhaseContext* other = nullptr;

    do {
      other = static_cast<TwoPhaseContext*> (Base::getOwner (l));

      if (other == this) {
        return;
      }

      if (other) {
        bool conflict = PtrComparator::compare (other, this); // *other < *this
        if (conflict) {
          // A lock that I want but can't get
          this->source = false;
          return; 
        }
      }
    } while (!this->stealByCAS(l, other));

    // Disable loser
    if (other) {
      other->source = false; // Only need atomic write
    }

    return;


    // bool succ = false;
    // if (Base::tryAcquire (l) == Base::NEW_OWNER) {
      // Base::addToNhood (l);
      // succ = true;
    // }
// 
    // assert (Base::getOwner (l) != NULL);
// 
    // if (!succ) {
      // while (true) {
        // TwoPhaseContext* that = static_cast<TwoPhaseContext*> (Base::getOwner (l));
// 
        // assert (that != NULL);
        // assert (this != that);
// 
        // if (PtrComparator::compare (this, that)) { // this < that
          // if (Base::stealByCAS (that, this)) {
            // that->source = false;
            // break;
          // }
// 
        // } else { // this >= that
          // this->source = false; 
          // break;
        // }
      // }
    // } // end outer if
  } // end subAcquire


  struct PtrComparator {

    static inline bool compare (const TwoPhaseContext* left, const TwoPhaseContext* right) {
      assert (left != nullptr);
      assert (right != nullptr);
      assert (&left->cmp == &right->cmp);

      return left->cmp (left->active, right->active);
    }

    inline bool operator () (const TwoPhaseContext* left, const TwoPhaseContext* right) const {
      return compare (left, right);
    }
  };

};




} // end namespace Runtime
} // end namespace Galois
#endif // GALOIS_RUNTIME_KDG_TWO_PHASE_SUPPORT_H
