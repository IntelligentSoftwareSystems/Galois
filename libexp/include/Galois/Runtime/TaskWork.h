/** Task execution -*- C++ -*-
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
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_TASKWORK_H
#define GALOIS_RUNTIME_TASKWORK_H

#include "Galois/gdeque.h"
#include "Galois/gslist.h"
#include "Galois/Runtime/Substrate.h"
#include "Galois/Runtime/LoopStatistics.h"

// #include <array> if c++11
#include <boost/array.hpp>
#include <boost/mpl/has_xxx.hpp>

namespace Galois {
/**
 * Indicates the operator doesn't need abort support
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_aborts)
template<typename T>
struct needs_aborts : public has_tt_needs_aborts<T> {};

template<
  typename WTy,
  typename Task1Ty,
  typename Task2Ty=void,
  typename Task3Ty=void,
  typename Task4Ty=void,
  typename Task5Ty=void>
struct Pipeline {
  typedef Task1Ty Task1Type;
  typedef Task2Ty Task2Type;
  typedef Task3Ty Task3Type;
  typedef Task4Ty Task4Type;
  typedef Task5Ty Task5Type;
  typedef WTy WorklistType;
};

}

namespace Galois {
namespace Runtime {
namespace Task {

//! User visible portion of task
class UserTask {

};

struct RuntimeTask: public UserTask {
  typedef Galois::gslist<RuntimeTask*,4> OutExtraType;

protected:
  typedef boost::array<RuntimeTask*,2> OutType;
  OutType out;
  OutExtraType outExtra;
  int type;
  bool deferred;
  volatile int incount;
  
  RuntimeTask(int t): type(t), deferred(false), incount(0) { 
    std::fill_n(out.begin(), out.size(), (RuntimeTask*) 0);
  }

public:
  bool isDone() const { return incount < 0; }
  bool isReady() const { return incount == 0 && !deferred; }
  void markDone() { --incount; assert(isDone()); }
  void incrementCount() { __sync_add_and_fetch(&incount, 1); }
  void markDeferred() { deferred = true; }
  bool decrementCount() { 
    return __sync_sub_and_fetch(&incount, 1) == 0;
  }

  void abort() {
    // NB: only called on unpublished tasks
    incount = -1; 
  }

  template<typename HeapTy>
  void addEdge(HeapTy& heap, RuntimeTask* dst) {
    dst->incrementCount();

    for (OutType::iterator ii = out.begin(), ei = out.end(); ii != ei; ++ii) {
      if (*ii)
        continue;
      *ii = dst;
      return;
    }

    outExtra.push_front(heap, dst);
  }

  template<typename HeapTy>
  void clear(HeapTy& heap) {
    outExtra.clear(heap);
    assert(isDone());
  }

  template<typename WorklistPipelineTy>
  void schedule(WorklistPipelineTy& wls) {
    switch (type) {
      case 5: wls.wl5.push(static_cast<typename WorklistPipelineTy::GTask5Type*>(this)); break;
      case 4: wls.wl4.push(static_cast<typename WorklistPipelineTy::GTask4Type*>(this)); break;
      case 3: wls.wl3.push(static_cast<typename WorklistPipelineTy::GTask3Type*>(this)); break;
      case 2: wls.wl2.push(static_cast<typename WorklistPipelineTy::GTask2Type*>(this)); break;
      case 1: wls.wl1.push(static_cast<typename WorklistPipelineTy::GTask1Type*>(this)); break;
      default: abort();
    }
  }
  
  template<typename WorklistPipelineTy>
  void updateSuccessors(WorklistPipelineTy& wls) {
    for (OutType::iterator ii = out.begin(), ei = out.end(); ii != ei; ++ii) {
      if (*ii && (*ii)->decrementCount()) {
        (*ii)->schedule(wls);
      }
    }
    for (OutExtraType::iterator ii = outExtra.begin(), ei = outExtra.end(); ii != ei; ++ii) {
      if ((*ii)->decrementCount())
        (*ii)->schedule(wls);
    }
  }
};

template<typename TaskTy>
struct GTask: public RuntimeTask {
  TaskTy task;

  template<typename... Args>
  GTask(int type, Args&&... args): RuntimeTask(type), task(std::forward<Args>(args)...) { }
};

template<typename TaskTy>
struct TaskTraits {
  typedef GTask<TaskTy> GTaskType;

  typedef char yes[1];
  typedef char no[2];
  template<typename U, void (U::*)()> struct SFINAE { };
  template<typename U> static yes& test(SFINAE<U, &U::inspect>*);
  template<typename> static no& test(...);

  enum {
    hasTask = true,
    hasInspect = sizeof(test<TaskTy>(0)) == sizeof(yes),
    needsAbort = Galois::needs_aborts<TaskTy>::value
  };
};

template<>
struct TaskTraits<void> {
  typedef GTask<int> GTaskType;

  enum {
    hasTask = false,
    hasInspect = false,
    needsAbort = false
  };
};

template<bool>
struct InspectWrapper {
  template<typename T>
  void operator()(T& task) const { task.inspect(); }
};

template<>
struct InspectWrapper<false> {
  template<typename T>
  void operator()(T& task) const { }
};

template<bool>
struct CallWrapper {
  template<typename T,typename U>
  void operator()(T& task, U& u) const { task(u); }
};

template<>
struct CallWrapper<false> {
  template<typename T,typename U>
  void operator()(T& task, U& u) const { }
};

template<typename PipelineTy>
class UserTaskContext: private boost::noncopyable {
protected:
  typedef Galois::gdeque<RuntimeTask*> BufType;

  typedef typename TaskTraits<typename PipelineTy::Task1Type>::GTaskType GTask1Type;
  typedef typename TaskTraits<typename PipelineTy::Task2Type>::GTaskType GTask2Type;
  typedef typename TaskTraits<typename PipelineTy::Task3Type>::GTaskType GTask3Type;
  typedef typename TaskTraits<typename PipelineTy::Task4Type>::GTaskType GTask4Type;
  typedef typename TaskTraits<typename PipelineTy::Task5Type>::GTaskType GTask5Type;

  Galois::gdeque<GTask1Type,64> task1;
  Galois::gdeque<GTask2Type,64> task2;
  Galois::gdeque<GTask3Type,64> task3;
  Galois::gdeque<GTask4Type,64> task4;
  Galois::gdeque<GTask5Type,64> task5;

  BufType buf;
  
  Galois::IterAllocBaseTy IterationAllocatorBase;
  Galois::PerIterAllocTy PerIterationAllocator;
  Galois::Runtime::FixedSizeHeap heap;

  UserTaskContext():
    PerIterationAllocator(&IterationAllocatorBase), 
    heap(sizeof(typename RuntimeTask::OutExtraType::block_type)) { }

public:
  Galois::PerIterAllocTy& getPerIterAlloc() { return PerIterationAllocator; }

  template<typename... Args>
  GTask1Type* addTask1(Args&&... args) {
    task1.emplace_back(1, std::forward<Args>(args)...);
    GTask1Type* t = &task1.back();
    buf.push_back(t);
    return t;
  }

  template<typename... Args>
  GTask2Type* addTask2(Args&&... args) {
    task2.emplace_back(2, std::forward<Args>(args)...);
    GTask2Type* t = &task2.back();
    buf.push_back(t);
    return t;
  }
  
  void addDependence(UserTask* src, UserTask* dst) {
    // NB(ddn): only safe if rsrc is thread-local
    RuntimeTask* rsrc = static_cast<RuntimeTask*>(src);
    RuntimeTask* rdst = static_cast<RuntimeTask*>(dst);
    rsrc->addEdge(heap, rdst);
  }

  //! Deferred tasks are those that dependences on tasks not yet available.
  //! Deferred tasks are not immediately scheduled; instead, they are scheduled
  //! when subsequent tasks add dependence edges and those tasks execute.
  void markDeferred(UserTask* t) {
    RuntimeTask* rt = static_cast<RuntimeTask*>(t);
    rt->markDeferred();
  }
};
}
}
}

// Export user visible types
namespace Galois {
template<typename PipelineTy>
struct TaskContext: public Galois::Runtime::Task::UserTaskContext<PipelineTy> { };
}

namespace Galois {
namespace Runtime {
namespace Task {

template<typename PipelineTy>
class RuntimeTaskContext: public Galois::TaskContext<PipelineTy> {
  typedef Galois::TaskContext<PipelineTy> SuperType;
  typedef typename TaskTraits<typename PipelineTy::Task1Type>::GTaskType GTask1Type;

  template<typename TaskTy,typename DequeTy>
  void gc(DequeTy& deque) {
    if (!TaskTraits<TaskTy>::hasTask) return;
    
    while (!deque.empty()) {
      RuntimeTask* t = &deque.back();
      if (!t->isDone()) break;
      t->clear(this->heap);
      deque.pop_back();
    }

    while (!deque.empty()) {
      RuntimeTask* t = &deque.front();
      if (!t->isDone()) break; 
      t->clear(this->heap);
      deque.pop_front();
    }
  }

public:
  RuntimeTaskContext() { }

  void clear() {
    reset();
    gc();
    assert(this->task1.empty());
    assert(this->task2.empty());
    assert(this->task3.empty());
    assert(this->task4.empty());
    assert(this->task5.empty());
  }

  void gc() {
    gc<typename PipelineTy::Task1Type>(this->task1);
    gc<typename PipelineTy::Task2Type>(this->task2);
    gc<typename PipelineTy::Task3Type>(this->task3);
    gc<typename PipelineTy::Task4Type>(this->task4);
    gc<typename PipelineTy::Task5Type>(this->task5);
  }

  template<typename IterTy,typename WLTy>
  void push_initial(IterTy ii, IterTy ei, WLTy& wl) {
    for (; ii != ei; ++ii) {
      this->task1.push_back(GTask1Type(1, *ii));
      wl.push(&this->task1.back());
    }
  }

  void abort() {
    for (typename SuperType::BufType::iterator ii = this->buf.begin(), ei = this->buf.end(); ii != ei; ++ii) {
      (*ii)->abort();
    }
  }

  void reset() {
    this->buf.clear();
    this->IterationAllocatorBase.clear();
  }

  template<typename WorklistPipelineTy>
  void commit(RuntimeTask* t, WorklistPipelineTy& wls) {
    t->markDone();
    t->updateSuccessors(wls);
    
    for (typename SuperType::BufType::iterator ii = this->buf.begin(), ei = this->buf.end(); ii != ei; ++ii) {
      if ((*ii)->isReady())
        (*ii)->schedule(wls);
    }
  }

  SuperType& data() { return *this; }
};
}
}
}

// Export user visible types
namespace Galois {
typedef Galois::Runtime::Task::UserTask* Task;
}

namespace Galois {
namespace Runtime {
namespace Task {

struct TaskRuntimeContext { }; // XXX

template<typename PipelineTy>
struct WorklistPipeline: private boost::noncopyable {
  typedef typename TaskTraits<typename PipelineTy::Task1Type>::GTaskType GTask1Type;
  typedef typename TaskTraits<typename PipelineTy::Task2Type>::GTaskType GTask2Type;
  typedef typename TaskTraits<typename PipelineTy::Task3Type>::GTaskType GTask3Type;
  typedef typename TaskTraits<typename PipelineTy::Task4Type>::GTaskType GTask4Type;
  typedef typename TaskTraits<typename PipelineTy::Task5Type>::GTaskType GTask5Type;

  typename PipelineTy::WorklistType::template retype<GTask1Type*> wl1;
  typename PipelineTy::WorklistType::template retype<GTask2Type*> wl2;
  typename PipelineTy::WorklistType::template retype<GTask3Type*> wl3;
  typename PipelineTy::WorklistType::template retype<GTask4Type*> wl4;
  typename PipelineTy::WorklistType::template retype<GTask5Type*> wl5;
};

template<typename PipelineTy,typename IterTy>
class Executor {
  typedef TaskRuntimeContext Context;

  typedef typename TaskTraits<typename PipelineTy::Task1Type>::GTaskType GTask1Type;
  typedef typename TaskTraits<typename PipelineTy::Task2Type>::GTaskType GTask2Type;
  typedef typename TaskTraits<typename PipelineTy::Task3Type>::GTaskType GTask3Type;
  typedef typename TaskTraits<typename PipelineTy::Task4Type>::GTaskType GTask4Type;
  typedef typename TaskTraits<typename PipelineTy::Task5Type>::GTaskType GTask5Type;
  
  static const int chunkSize = 32;
  static const int workSize = chunkSize * 2;

  struct ThreadLocalData: private boost::noncopyable {
    RuntimeTaskContext<PipelineTy> facing;
    WorklistPipeline<PipelineTy> ins;
    SimpleRuntimeContext ctx;
    LoopStatistics<true> stat;
    
    ThreadLocalData(const char* ln): stat(ln) { }
  };

  // XXX scheduling type (infer?)
  WorklistPipeline<PipelineTy> wls;

  Substrate::TerminationDetection& term;
  Substrate::Barrier& barrier;
  IterTy initialBegin;
  IterTy initialEnd;
  const char* loopname;

  template<typename TaskTy,typename WLTy>
  GALOIS_ATTRIBUTE_NOINLINE
  void processWithAborts(ThreadLocalData& tld, WLTy& wl, bool& didWork) {
    assert(TaskTraits<TaskTy>::needsAbort);
    setThreadContext(&tld.ctx);

    int count;
    Galois::optional<typename TaskTraits<TaskTy>::GTaskType*> p;
    InspectWrapper<TaskTraits<TaskTy>::hasInspect> inspect;
    CallWrapper<TaskTraits<TaskTy>::hasTask> call;
    int result = 0;

#ifdef GALOIS_USE_LONGJMP
    if ((result = setjmp(hackjmp)) == 0) {
#else
    try {
#endif
    for (count = 0; count < workSize && (p = wl.pop()); ++count) {
 //   for (count = 0; p = wl.pop(); ++count) {
      tld.ctx.startIteration();
      tld.stat.inc_iterations();
      inspect((*p)->task);
      call((*p)->task, tld.facing.data());
      tld.facing.commit(&(**p), wls);
      tld.facing.reset();
      tld.ctx.commitIteration();
    }
#ifdef GALOIS_USE_LONGJMP
    } else { clearConflictLock(); }
#else
    } catch (ConflictFlag const& flag) { clearConflictLock(); result = flag; }
#endif
    //FIXME:    clearReleasable(); 
    switch (result) {
      case 0: break;
      case Galois::Runtime::CONFLICT:
        tld.facing.abort();
        tld.facing.reset();
        tld.stat.inc_conflicts();
        tld.ctx.cancelIteration();
        wl.push(*p);
        didWork = true;
        break;
      default: abort();
    }

    if (count)
      didWork = true;
    setThreadContext(0);
  }

  template<typename TaskTy,typename WLTy>
  void processSimple(ThreadLocalData& tld, WLTy& wl, bool& didWork) {
    int count;
    Galois::optional<typename TaskTraits<TaskTy>::GTaskType*> p;
    InspectWrapper<TaskTraits<TaskTy>::hasInspect> inspect;
    CallWrapper<TaskTraits<TaskTy>::hasTask> call;

    for (count = 0; count < workSize && (p = wl.pop()); ++count) {
 //   for (count = 0; p = wl.pop(); ++count) { XXX
      tld.stat.inc_iterations();
      inspect((*p)->task);
      call((*p)->task, tld.facing.data());
      tld.facing.commit(&(**p), wls);
      tld.facing.reset();
    }

    if (count)
      didWork = true;
  }

  template<typename TaskTy,typename WLTy>
  void process(ThreadLocalData& tld, WLTy& wl, bool& didWork) {
    if (!TaskTraits<TaskTy>::hasTask) return;
    if (TaskTraits<TaskTy>::needsAbort) { 
      processWithAborts<TaskTy>(tld, wl, didWork);
    } else {
      processSimple<TaskTy>(tld, wl, didWork);
    }
  }

public:
  Executor(IterTy b, IterTy e, const char* ln): term(Substrate::getSystemTermination(Galois::getActiveThreads())), barrier(Runtime::getBarrier(Galois::getActiveThreads())), initialBegin(b), initialEnd(e), loopname(ln) { 
    barrier.reinit(Galois::getActiveThreads());
  }

  void initThread(void) {
    term.initializeThread();
  }

  void operator()() {
    ThreadLocalData tld(loopname);
    std::pair<IterTy,IterTy> range = Galois::block_range(initialBegin, initialEnd, Substrate::ThreadPool::getTID(), Galois::getActiveThreads());
    tld.facing.push_initial(range.first, range.second, wls.wl1);

    barrier.wait();

    int rounds = 0;
    int index = 0;

    do {
      bool didWork;
      do {
        didWork = false;
        
        // XXX FIFO versus LIFO etc scheduling measure in/out degrees
        for (int count = 0; count < 5; ++count) {
          switch (index) {
          case 0:
          process<typename PipelineTy::Task1Type>(tld, wls.wl1, didWork);
            process<typename PipelineTy::Task1Type>(tld, tld.ins.wl1, didWork);
            break;
          case 1:
          process<typename PipelineTy::Task2Type>(tld, wls.wl2, didWork);
            process<typename PipelineTy::Task2Type>(tld, tld.ins.wl2, didWork);
            break;
          case 2:
            process<typename PipelineTy::Task3Type>(tld, wls.wl3, didWork);
            process<typename PipelineTy::Task3Type>(tld, tld.ins.wl3, didWork);
            break;
          case 3:
            process<typename PipelineTy::Task4Type>(tld, wls.wl4, didWork);
            process<typename PipelineTy::Task4Type>(tld, tld.ins.wl4, didWork);
            break;
          case 4:
            process<typename PipelineTy::Task5Type>(tld, wls.wl5, didWork);
            process<typename PipelineTy::Task5Type>(tld, tld.ins.wl5, didWork);
            break;
          default:
            abort();
          }
          if (!didWork) {
            if (++index == 5)
              index = 0;
          }
        }

        //term.localTermination(didWork);
        
        if ((++rounds & 7) == 0)
          tld.facing.gc();

      } while (didWork);
      
      term.localTermination(didWork);
    } while (!term.globalTermination());
    
    barrier.wait(); // XXX: certainly will need this if we have multiple wls
    tld.facing.clear();
  }
};

}
}
}

namespace Galois {

//! Task executor
template<typename PipelineTy,typename IterTy>
static inline void for_each_task(IterTy b, IterTy e, const char* loopname = 0) {
  typedef Galois::Runtime::Task::Executor<PipelineTy,IterTy> WorkTy;

  WorkTy W(b, e, loopname);
  
  using namespace Galois::Runtime;
  Substrate::getThreadPool().run(activeThreads, std::bind(&WorkTy::initThread, std::ref(W)), std::ref(W));
}

template<typename PipelineTy,typename TaskTy>
static inline void for_each_task(TaskTy t, const char* loopname = 0) {
  TaskTy tasks[1] = { t };
  for_each_task<PipelineTy>(&tasks[0], &tasks[1], loopname);
}

}
#endif
