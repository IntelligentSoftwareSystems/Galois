/**  -*- C++ -*-
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
 */

#ifndef GALOIS_RUNTIME_COUPLEDEXECUTOR_H
#define GALOIS_RUNTIME_COUPLEDEXECUTOR_H

#include "Galois/GaloisForwardDecl.h"
#include "Galois/PerThreadContainer.h"

#include <tuple>

namespace Galois {
namespace Runtime {

enum ExecType {
  DOALL_WAKEUP=0,
  DOALL_EXPLICIT,
};

static const bool DO_STEAL = true;

static const char* EXEC_NAMES[] = { "DOALL_WAKEUP", "DOALL_EXPLICIT" };

static cll::opt<ExecType> execType (
    cll::desc ("Executor type"),
    cll::values (
      clEnumVal (DOALL_WAKEUP, "Wake up thread pool"),
      clEnumVal (DOALL_EXPLICIT, "Explicit parallel loop"),
      clEnumValEnd),
    cll::init (DOALL_WAKEUP));

template<typename T>
struct PushWrapper {
  T& container;
  PushWrapper(T& c): container(c) { }
  template<typename... Args>
  void push(Args&&... args) {
    container.emplace_back(std::forward<Args>(args)...);
  }
};

template <typename R, typename F>
void do_all_coupled_wake (const R& initRange, const F& func, const char* loopname=nullptr) {

  typedef typename R::value_type T;
  typedef PerThreadVector<T> WL_ty;

  WL_ty* curr = new WL_ty ();
  WL_ty* next = new WL_ty ();

  Substrate::getThreadPool ().burnPower (Galois::getActiveThreads ());

  Galois::on_each(
      [&initRange, &next] (const unsigned tid, const unsigned numT) {
        auto rp = initRange.local_pair ();
        for (auto i = rp.first, i_end = rp.second; i != i_end; ++i) {
          next->get ().push_back (*i);
        }
      });

  // std::printf ("Initial size: %zd\n", next->size_all ());
  F func_cpy (func);

  while (!next->empty_all ()) {
    std::swap (curr, next);

    Galois::on_each(
        [&next] (const unsigned tid, const unsigned numT) {
          next->get ().clear ();
        });

    // std::printf ("Current size: %zd\n", curr->size_all ());

    Galois::do_all_local(*curr,
        [&func_cpy, &next] (const T& t) {
          PushWrapper<typename WL_ty::Cont_ty> w(next->get());
          func_cpy (t, w);
        },
        Galois::loopname("do_all_bs"),
        Galois::do_all_steal<DO_STEAL>());
  }

  Substrate::getThreadPool ().beKind ();

  delete curr;
  delete next;
}

template <typename R, typename F>
void do_all_coupled_bs (const R& initRange, const F& func, const char* loopname=nullptr) {
  std::printf ("Running do_all_coupled_bs with executor: %s\n", EXEC_NAMES[execType]);

  switch (execType) {
    case DOALL_WAKEUP:
      do_all_coupled_wake (initRange, func, loopname);
      break;
    case DOALL_EXPLICIT:
    default:
      std::abort();
  } 
}

#if 0
namespace impl {

template <typename F, typename WL>
struct FunctorWrapper {
  typedef int tt_does_not_need_abort;
  typedef char tt_does_not_need_push;

  F& func;
  WL*& wl;

  typedef typename WL::value_type T;

  explicit FunctorWrapper (F& func, WL*& wl): func (func), wl (wl) {}

  template <typename C>
  void operator () (const T& x, C&) {
    func (x, *wl);
  }
};
} // end namespace impl

template <typename R, typename F>
void for_each_coupled_wake (const R& initRange, const F& func, const char* loopname=nullptr) {
  const unsigned CHUNK_SIZE = 64;
  typedef typename R::value_type T;
  typedef WorkList::WLsizeWrapper<typename Galois::WorkList::dChunkedFIFO<CHUNK_SIZE>::template retype<T>::type> WL_ty;

  WL_ty* curr = new WL_ty ();
  WL_ty* next = new WL_ty ();

  F func_cpy (func);

  Substrate::getThreadPool ().burnPower (Galois::getActiveThreads ());

  Galois::on_each(
      [&next, &initRange] (const unsigned tid, const unsigned numT) {
        next->push_initial (initRange);
      });

  while (next->size () != 0) {
    typedef Galois::WorkList::ExternalReference<WL_ty> WL;
    typedef typename WL_ty::value_type value_type;
    value_type* it = nullptr;

    std::swap (curr, next);
    next->reset_all ();

    Galois::for_each(it, it,
        impl::FunctorWrapper<F, WL_ty> (func_cpy, next),
        Galois::loopname("for_each_coupled"),
        Galois::wl<WL>(curr));
  }

  Substrate::getThreadPool ().beKind ();

  delete curr; 
  delete next;
}

template <typename R, typename F>
void for_each_coupled_explicit (const R& initRange, const F& func, const char* loopname=nullptr) {
  const unsigned CHUNK_SIZE = 128;

  typedef typename R::value_type T;

  typedef WorkList::WLsizeWrapper<typename Galois::WorkList::dChunkedFIFO<CHUNK_SIZE>::template retype<T>::type> WL_ty;

  WL_ty* curr = new WL_ty ();
  WL_ty* next = new WL_ty ();

  F func_cpy (func);

  typedef impl::FunctorWrapper<F, WL_ty> FWrap;

  typedef Galois::Runtime::ForEachWork<WorkList::ExternalReference<WL_ty>, T, FWrap> ForEachExec_ty;

  ForEachExec_ty exec (curr, FWrap (func_cpy, next), loopname);

  Galois::Substrate::Barrier& barrier = Galois::Substrate::getSystemBarrier ();

  std::atomic<bool> done(false);

  auto loop = [&] (void) {
    exec.initThread(initRange);

    barrier ();

    while (true) {

      if (LL::getTID () == 0) {
        std::swap (curr, next);
        exec.initThread( );

        if (curr->size () == 0) { 
          done = true;
        }
      }

      exec.initThread ( );

      barrier ();

      if (done) { break; }

      next->reset ();

      exec ();

      barrier ();

    }

  };

  exec.init();
  Galois::Substrate::getThreadPool ().run (Galois::getActiveThreads (), loop);
  
  delete curr;
  delete next;
}


template <typename R, typename F>
void for_each_coupled_bs (const R& initRange, const F& func, const char* loopname=nullptr) {
  std::printf ("Running for_each_coupled_bs with executor: %s\n", EXEC_NAMES[execType]);

  switch (execType) {
    case DOALL_WAKEUP:
      for_each_coupled_wake (initRange, func, loopname);
      break;

    case DOALL_EXPLICIT:
      for_each_coupled_explicit (initRange, func, loopname);
      break;

    default:
      std::abort ();
  } 
}
#endif

} // end namespace Runtime
} // end namespace Galois


#endif // GALOIS_RUNTIME_COUPLED_EXECUTOR_H
