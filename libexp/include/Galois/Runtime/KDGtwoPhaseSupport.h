/** KDG two phase support -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_KDG_TWO_PHASE_SUPPORT_H
#define GALOIS_RUNTIME_KDG_TWO_PHASE_SUPPORT_H

#include "Galois/AltBag.h"

#include "Galois/Runtime/OrderedLockable.h"

#include <boost/iterator/filter_iterator.hpp>
#include <functional>
#include <boost/iterator/filter_iterator.hpp>

namespace Galois {
namespace Runtime {

namespace cll = llvm::cl;

static cll::opt<double> commitRatioArg("cratio", cll::desc("target commit ratio for two phase executor, 0.0 to disable windowing"), cll::init(0.80));

// TODO: figure out when to call startIteration

template <typename T, typename Cmp>
class TwoPhaseContext: public OrderedContextBase<T> {

  using Base = OrderedContextBase<T>;
  // using NhoodList =  Galois::gdeque<Lockable*, 4>;
  using CtxtCmp = ContextComparator<TwoPhaseContext, Cmp>;

  CtxtCmp ctxtCmp;
  bool source = true;

public:

  using value_type = T;

  explicit TwoPhaseContext (const T& x, const Cmp& cmp)
    : 
      Base (x),  // pass true so that Base::acquire invokes virtual subAcquire
      ctxtCmp (cmp),
      source (true) 
  {}

  bool isSrc (void) const {
    return source;
  }

  void disableSrc (void) {
    source = false;
  }

  void reset () { 
    source = true;
  }

  virtual void subAcquire (Lockable* l, Galois::MethodFlag) {


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
        bool conflict = ctxtCmp (other, this); // *other < *this
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



};

template <typename Ctxt, typename S>
class SafetyTestLoop {

  using T = typename Ctxt::value_type;

  struct GetActive: public std::unary_function<Ctxt, const T&> {
    const T& operator () (const Ctxt* c) const {
      assert (c != nullptr);
      return c->getActive ();
    }
  };

  struct GetLesserThan: public std::unary_function<const Ctxt*, bool> {

    const Ctxt* curr;
    typename Ctxt::PtrComparator cmp = typename Ctxt::PtrComparator ();

    bool operator () (const Ctxt* that) const { 
      return cmp (that, curr); 
    }
  };

  S safetyTest;

  static const unsigned DEFAULT_CHUNK_SIZE = 2;

public:

  explicit SafetyTestLoop (const S& safetyTest): safetyTest (safetyTest) {}

  template <typename R>
  void run (const R& range) const {

    Galois::do_all_choice (range,
        [this, &range] (const Ctxt* c) {

          auto beg_lesser = boost::make_filter_iterator (
            range.begin (), range.end (), GetLesserThan {c});

          auto end_lesser = boost::make_filter_iterator (
            range.end (), range.end (), GetLesserThan {c});


          auto bt = boost::make_transform_iterator (beg_lesser, GetActive ());
          auto et = boost::make_transform_iterator (end_lesser, GetActive ());


          if (!safetyTest (c->getActive (), bt, et)) {
            c->disableSrc ();
          }
        },
        "safety_test_loop",
        Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());
  }
};

template <typename Ctxt>
struct SafetyTestLoop<Ctxt, int> {

  SafetyTestLoop (int) {}

  template <typename R>
  void run (const R& range) const { 
  }
};


template <typename F, typename Ctxt, typename UserCtxt>
void runCatching (F& func, Ctxt* c, UserCtxt& uhand) {
  Galois::Runtime::setThreadContext (c);

  int result = 0;

#ifdef GALOIS_USE_LONGJMP
  if ((result = setjmp(hackjmp)) == 0) {
#else
    try {
#endif
      func (c->getActive (), uhand);

#ifdef GALOIS_USE_LONGJMP
    } else {
      // TODO
    }
#else 
  } catch (ConflictFlag f) {
    result = f;
  }
#endif

  switch (result) {
    case 0:
      break;
    case CONFLICT: 
      c->disableSrc ();
      break;
    default:
      GALOIS_DIE ("can't handle conflict flag type");
      break;
  }


  Galois::Runtime::setThreadContext (NULL);
}


// TODO: a common base class for IKDG executor
// template <typename T, typename Ctxt,  


} // end namespace Runtime
} // end namespace Galois
#endif // GALOIS_RUNTIME_KDG_TWO_PHASE_SUPPORT_H
