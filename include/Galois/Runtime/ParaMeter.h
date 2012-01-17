/** ParaMeter runtime -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * Implementation of ParaMeter runtime
 * Ordered with speculation not supported yet
 *
 * @author ahassaan@ices.utexas.edu
 */
#ifndef GALOIS_RUNTIME_PARAMETER_H_
#define GALOIS_RUNTIME_PARAMETER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <vector>
#include <algorithm>

#include <cstdlib>
#include <ctime>
#include <cstdio>

#include "Galois/UserContext.h"
#include "Galois/TypeTraits.h"
#include "Galois/Mem.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ForeachTraits.h"
#include "Galois/Runtime/Config.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/LoopHooks.h"

namespace GaloisRuntime {


void enableParaMeter ();

void disableParaMeter ();

bool usingParaMeter ();

template <typename WorkListTy, typename FunctionTy>
class ParaMeterExecutor;


class ParaMeter: private boost::noncopyable {
  // Single ParaMeter stats file per run of an app
  // which includes all instances of for_each loops
  // run with ParaMeter Executor
  //
  // basically, from commandline parser calls enableParaMeter
  // and we
  // - set a flag
  // - open a stats file in overwrite mode
  // - print stats header
  // - close file

  // for each for_each loop, we create an instace of ParaMeterExecutor
  // which
  // - opens stats file in append mode
  // - prints stats
  // - closes file when loop finishes
private:

  template <typename WorkListTy, typename FunctionTy>
  friend class ParaMeterExecutor;

  struct StepStats {
    size_t step;
    size_t availParallelism;
    size_t workListSize;

    static std::ostream& printHeader (std::ostream& out) {
      out << "LOOPNAME, STEP, AVAIL_PARALLELISM, WORKLIST_SIZE" << std::endl;
      return out;
    }

    std::ostream& dump (std::ostream& out, const char* loopname) const {
      out << loopname << ", " << step << ", " << availParallelism << ", " << workListSize << std::endl;
      return out;
    }
  };


  struct Init {
    std::string statsFname;

    static const std::string genFname () {

      time_t rawtime;
      struct tm* timeinfo;

      time (&rawtime);
      timeinfo = localtime (&rawtime);

      const size_t STR_SIZE = 256;
      char str [STR_SIZE];
      strftime (str, STR_SIZE, "ParaMeter_Stats_%Y-%m-%d_%H:%M:%S.csv", timeinfo);

      return str;
    }

    Init () {
      statsFname = genFname ();
      std::ofstream statsFile (statsFname.c_str (), std::ios_base::out);
      StepStats::printHeader (statsFile);
      statsFile.close ();
    }
  };


  //////////////////////////////////////////////
  static Init* init;


  static Init& getInit () {
    if (init == NULL) {
      init = new Init ();
    }
    return *init;
  }
    

  ParaMeter () {
  }

  ~ParaMeter () {
    delete init;
    init = NULL;
  }

public:

  static void initialize () {
    getInit ();
  }

  static const std::string& statsFileName () {
    return getInit ().statsFname;
  }

  template <typename WLTy, typename IterTy, typename Func, typename Filter>
  static void for_each_impl (IterTy b, IterTy e, Func func, Filter filter, const char* loopname) {

    typedef typename WLTy::template retype<typename std::iterator_traits<IterTy>::value_type>::WL ActualWLTy;

    ParaMeterExecutor<ActualWLTy, Func> executor (func, loopname);

    executor.addInitialWork (b, e, filter);

    executor.run ();

  }

};


template <typename WorkListTy, typename FunctionTy>
class ParaMeterExecutor: private boost::noncopyable {

private:
  typedef typename WorkListTy::value_type value_type;

  typedef Galois::UserContext<value_type> UserContextTy;

  struct IterationContext {
    UserContextTy facing;
    SimpleRuntimeContext cnx;
  };

  typedef std::deque<IterationContext*> IterQueue;



  class ParaMeterWorkList: private boost::noncopyable {
    WorkListTy* curr;
    WorkListTy* next;

  public:
    ParaMeterWorkList () {
      curr = new WorkListTy ();
      next = new WorkListTy ();
    }

    ~ParaMeterWorkList () {
      delete curr;
      curr = NULL;
      delete next;
      next = NULL;
    }

    WorkListTy& getCurr () {
      return *curr;
    }

    WorkListTy& getNext () {
      return *next;
    }

    void switchWorkLists () {
      delete curr;
      curr = next;
      next = new WorkListTy ();
    }
  };


  FunctionTy body;
  const char* loopname;
  ParaMeterWorkList workList;

  std::ofstream* pstatsFile;

  IterQueue commitQueue;
  // XXX: may turn out to be unnecessary
  std::vector<ParaMeter::StepStats> allSteps;


public:

  ParaMeterExecutor (FunctionTy _body, const char* _loopname)
  : body (_body), loopname (_loopname), pstatsFile (NULL) {

    if (this->loopname == NULL) {
      this->loopname = "foreach";
    }
    
  }

  ~ParaMeterExecutor () {
    delete pstatsFile;
    pstatsFile = NULL;
  }

  template <typename Iter, typename Filter>
  bool addInitialWork (Iter b, Iter e, Filter filter) {
    for (; b != e; ++b) {
      if (filter (*b)) {
        workList.getCurr ().pushi (*b);
      }
    }

    return true;
  }


  void run () {

    beginLoop ();

    size_t currStep = 0;
    bool done = false;
    while (!done) {

      // do initialization for a new step
      //
      // while currWorkList is not empty {
      //  remove an item from current workList
      //  create a new iteration context
      //  run function with iteration and item
      //  if aborted {
      //    add item to nextWorkList
      //  } else {
      //    add iteration to commit queue
      //  }
      // }
      //
      // measure commit queue's size
      // if (size == 0) {
      //  ERROR, no progress?
      // }
      // for each iter in commit queue {
      //  commit iteration
      //    release all locks
      //    add new items to nextWorkList
      //    collect locks/neighborhood stats
      // }
      //
      // log current step
      // move to next step
      //

      size_t numIter = 0;

      for (std::pair<bool, value_type> item = workList.getCurr ().pop ();
          item.first; item = workList.getCurr ().pop ()) {

        IterationContext& it = newIteration ();

        bool doabort = false;
        try {
          body (item.second, it.facing);

        } catch (int a) {
          doabort = true;
        }

        if (doabort) {
          abortIteration (it, item.second);

        } else {

          if (ForeachTraits<FunctionTy>::NeedsBreak) {
            if (it.facing.__breakHappened ()) {
              std::cerr << "ParaMeterExecutor: can't handle breaks yet" << std::endl;
              abort ();
            }
          }

          commitQueue.push_back (&it);
        }

        ++numIter;
      }


      if (numIter == 0) {
        done = true;
        continue;
      }

      size_t numActivities = commitQueue.size ();

      if (numActivities == 0) {
        std::cerr << "ParaMeterExecutor: no progress made in step=" << currStep << std::endl;
        abort ();
      }


      double avgLocks = 0;
      for (typename IterQueue::iterator i = commitQueue.begin (), ei = commitQueue.end ();
          i != ei; ++i) {

        unsigned numLocks = commitIteration (*(*i));
        avgLocks += double (numLocks);
      }

      avgLocks /= numActivities;

      commitQueue.clear ();


      // switch worklists
      // dump stats
      ParaMeter::StepStats stat;
      stat.step = currStep;
      stat.availParallelism = numActivities;
      stat.workListSize = numIter;


      finishStep (stat);
      ++currStep;

    } // end while


    finishLoop ();
  }
    
private:

  void beginLoop () {
    // all instances of ParaMeterExecutor (one per for_each), open the stats
    // file in append mode
    pstatsFile = new std::ofstream (ParaMeter::statsFileName ().c_str (), std::ios_base::app); 
  }

  void finishStep (const ParaMeter::StepStats& stat) {
    allSteps.push_back (stat);
    stat.dump (*pstatsFile, loopname);
    workList.switchWorkLists ();
  }

  void finishLoop () {
    pstatsFile->close ();
  }

  IterationContext& newIteration () const {
    IterationContext* it = new IterationContext ();
    
    setThreadContext (&(it->cnx));

    return *it;
  }

  unsigned retireIteration (IterationContext& it, const bool abort) const {

    if (ForeachTraits<FunctionTy>::NeedsPush) {
      it.facing.__getPushBuffer ().clear ();
    }

    if (ForeachTraits<FunctionTy>::NeedsPIA) {
      it.facing.__resetAlloc ();
    }

    if (ForeachTraits<FunctionTy>::NeedsBreak) {
      it.facing.__resetBreak ();
    }

    unsigned numLocks = 0;
    if (abort) {
      numLocks = it.cnx.cancel_iteration ();

    } else {
      numLocks = it.cnx.commit_iteration ();
    }

    delete &it;

    return numLocks;

  }

  unsigned abortIteration (IterationContext& it, value_type& item) {
    clearConflictLock ();
    workList.getNext ().push (item);
    return retireIteration (it, true);
  }

  unsigned commitIteration (IterationContext& it) {
    if (ForeachTraits<FunctionTy>::NeedsPush) {

      for (typename UserContextTy::pushBufferTy::iterator a = it.facing.__getPushBuffer ().begin (),
          ea = it.facing.__getPushBuffer ().end (); a != ea; ++a) {
        workList.getNext ().push (*a);
      }
    }

    return retireIteration (it, false);
  }





};


} // end namespace

#endif // GALOIS_RUNTIME_PARAMETER_H_
