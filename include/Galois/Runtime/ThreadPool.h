/** Simple thread related classes -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_THREADPOOL_H
#define GALOIS_RUNTIME_THREADPOOL_H

#include "Galois/config.h"
#include GALOIS_CXX11_STD_HEADER(functional)

namespace Galois {
namespace Runtime {

namespace detail {
template<typename tpl, int s, int r>
struct exTupleImpl {
  static inline void execute(tpl& cmds) {
    std::get<s>(cmds)();
    exTupleImpl<tpl,s+1,r-1>::execute(cmds);
  }
};
template<typename tpl, int s>
struct exTupleImpl<tpl, s, 0> {
  static inline void execute(tpl& f) { }
};
}

class ThreadPool {
protected:
  unsigned maxThreads;
  ThreadPool(unsigned m): maxThreads(m) { }

  //!execute work on all threads
  virtual void runInternal(unsigned num, std::function<void (void)>& cmd) = 0;

  //Common implementation stuff

  //! Initialize TID and PTS
  void initThreadCommon(unsigned tid);

public:
  virtual ~ThreadPool() { }

  //! execute work on all threads
  //! a simple wrapper for run
  template<typename... Args>
  void run(unsigned num, Args&&... args) {
    struct exTuple {
      using Ty = std::tuple<Args...>;
      Ty cmds;

      void operator() () {
        detail::exTupleImpl<Ty, 0, std::tuple_size<Ty>::value>::execute(cmds);
      }
      exTuple(Args&&... args) :cmds(std::forward<Args>(args)...) {}
    };
    std::function<void(void)> pf(exTuple(std::forward<Args>(args)...));
    //    std::function<void(void)> pf(std::ref(f));
    runInternal(num, pf);
  }

  //!return the number of threads supported by the thread pool on the current machine
  unsigned getMaxThreads() const { return maxThreads; }
};

//!Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();

} //Runtime
} //Galois

#endif
