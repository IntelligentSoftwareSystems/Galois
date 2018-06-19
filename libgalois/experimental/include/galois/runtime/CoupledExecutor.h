/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_RUNTIME_COUPLEDEXECUTOR_H
#define GALOIS_RUNTIME_COUPLEDEXECUTOR_H

#include "galois/GaloisForwardDecl.h"
#include "galois/PerThreadContainer.h"

#include <tuple>

namespace galois {
namespace runtime {

enum ExecType {
  DOALL_WAKEUP = 0,
  DOALL_EXPLICIT,
};

static const bool DO_STEAL = true;

static const char* EXEC_NAMES[] = {"DOALL_WAKEUP", "DOALL_EXPLICIT"};

static cll::opt<ExecType>
    execType(cll::desc("Executor type"),
             cll::values(clEnumVal(DOALL_WAKEUP, "Wake up thread pool"),
                         clEnumVal(DOALL_EXPLICIT, "Explicit parallel loop"),
                         clEnumValEnd),
             cll::init(DOALL_WAKEUP));

template <typename T>
struct PushWrapper {
  T& container;
  PushWrapper(T& c) : container(c) {}
  template <typename... Args>
  void push(Args&&... args) {
    container.emplace_back(std::forward<Args>(args)...);
  }
};

template <typename R, typename F>
void do_all_coupled_wake(const R& initRange, const F& func,
                         const char* loopname = nullptr) {

  typedef typename R::value_type T;
  typedef PerThreadVector<T> WL_ty;

  WL_ty* curr = new WL_ty();
  WL_ty* next = new WL_ty();

  substrate::getThreadPool().burnPower(galois::getActiveThreads());

  galois::on_each([&initRange, &next](const unsigned tid, const unsigned numT) {
    auto rp = initRange.local_pair();
    for (auto i = rp.first, i_end = rp.second; i != i_end; ++i) {
      next->get().push_back(*i);
    }
  });

  // std::printf ("Initial size: %zd\n", next->size_all ());
  F func_cpy(func);

  while (!next->empty_all()) {
    std::swap(curr, next);

    galois::on_each([&next](const unsigned tid, const unsigned numT) {
      next->get().clear();
    });

    // std::printf ("Current size: %zd\n", curr->size_all ());

    galois::runtime::do_all_gen(makeLocalRange(*curr),
                                [&func_cpy, &next](const T& t) {
                                  PushWrapper<typename WL_ty::Cont_ty> w(
                                      next->get());
                                  func_cpy(t, w);
                                },
                                galois::loopname("do_all_bs"), galois::steal());
  }

  substrate::getThreadPool().beKind();

  delete curr;
  delete next;
}

template <typename R, typename F>
void do_all_coupled_bs(const R& initRange, const F& func,
                       const char* loopname = nullptr) {
  std::printf("Running do_all_coupled_bs with executor: %s\n",
              EXEC_NAMES[execType]);

  switch (execType) {
  case DOALL_WAKEUP:
    do_all_coupled_wake(initRange, func, loopname);
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
  typedef worklists::WLsizeWrapper<typename galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>::template retype<T>::type> WL_ty;

  WL_ty* curr = new WL_ty ();
  WL_ty* next = new WL_ty ();

  F func_cpy (func);

  substrate::getThreadPool ().burnPower (galois::getActiveThreads ());

  galois::on_each(
      [&next, &initRange] (const unsigned tid, const unsigned numT) {
        next->push_initial (initRange);
      });

  while (next->size () != 0) {
    typedef galois::worklists::ExternalReference<WL_ty> WL;
    typedef typename WL_ty::value_type value_type;
    value_type* it = nullptr;

    std::swap (curr, next);
    next->reset_all ();

    galois::for_each(it, it,
        impl::FunctorWrapper<F, WL_ty> (func_cpy, next),
        galois::loopname("for_each_coupled"),
        galois::wl<WL>(curr));
  }

  substrate::getThreadPool ().beKind ();

  delete curr;
  delete next;
}

template <typename R, typename F>
void for_each_coupled_explicit (const R& initRange, const F& func, const char* loopname=nullptr) {
  const unsigned CHUNK_SIZE = 128;

  typedef typename R::value_type T;

  typedef worklists::WLsizeWrapper<typename galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>::template retype<T>::type> WL_ty;

  WL_ty* curr = new WL_ty ();
  WL_ty* next = new WL_ty ();

  F func_cpy (func);

  typedef impl::FunctorWrapper<F, WL_ty> FWrap;

  typedef galois::runtime::ForEachWork<worklists::ExternalReference<WL_ty>, T, FWrap> ForEachExec_ty;

  ForEachExec_ty exec (curr, FWrap (func_cpy, next), loopname);

  galois::substrate::Barrier& barrier = galois::substrate::getSystemBarrier ();

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
  galois::substrate::getThreadPool ().run (galois::getActiveThreads (), loop);

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

} // end namespace runtime
} // end namespace galois

#endif // GALOIS_RUNTIME_COUPLED_EXECUTOR_H
