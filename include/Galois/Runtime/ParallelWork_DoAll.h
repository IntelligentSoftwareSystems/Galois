/** Galois do all runtime -*- C++ -*-
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
 * @section Description
 *
 * Implementation of the Galois doall iterator. Includes various 
 * specializations to operators to reduce runtime overhead.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_PARALLELWORK_DOALL_H
#define GALOIS_RUNTIME_PARALLELWORK_DOALL_H

namespace GaloisRuntime {

template<class WorkListTy, class FunctionTy>
class DoAllWork {
  typedef typename WorkListTy::value_type value_type;

  WorkListTy global_wl;
  FunctionTy& fn;

public:
  DoAllWork(FunctionTy& f): fn(f) { }

  template<typename Iter>
  void AddInitialWork(Iter b, Iter e) {
    global_wl.initializeThread();
    if (b != e)
      global_wl.push_initial(b,e);
  }

  void operator()() {
    boost::optional<value_type> p = global_wl.pop();
    while (p) {
      fn(*p);
      p = global_wl.pop();
    }
#ifdef GALOIS_EXP
    SimpleTaskPool& pool = getSystemTaskPool();
    pool.work();
#endif
  }
};

template<typename WLTy, typename IterTy, typename FunctionTy>
void do_all_impl(IterTy b, IterTy e, FunctionTy f, const char* loopname) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  DoAllWork<WLTy, FunctionTy> W(f);

  W.AddInitialWork(b, e);

  RunCommand w[1];
  w[0].work = Config::ref(W);
  w[0].isParallel = true;
  w[0].barrierAfter = true;
  w[0].profile = false;
  getSystemThreadPool().run(&w[0], &w[1]);

  inGaloisForEach = false;
}

}

#endif
