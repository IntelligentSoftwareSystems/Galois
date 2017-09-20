#ifndef GALOIS_RUNTIME_IKDG_BASE_H
#define GALOIS_RUNTIME_IKDG_BASE_H


#include "galois/AltBag.h"
#include "galois/Accumulator.h"
#include "galois/OrderedTraits.h"
#include "galois/DoAllWrap.h"

#include "galois/runtime/OrderedLockable.h"
#include "galois/runtime/WindowWorkList.h"

#include <boost/iterator/filter_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include "llvm/Support/CommandLine.h"

#include <utility>
#include <functional>

namespace galois {
namespace runtime {

namespace cll = llvm::cl;

static cll::opt<double> commitRatioArg("cratio", cll::desc("target commit ratio for two phase executor, 0.0 to disable windowing"), cll::init(0.80));

// TODO: figure out when to call startIteration

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

    galois::do_all_choice (range,
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
        std::make_tuple(
          galois::loopname("safety_test_loop"),
        galois::chunk_size<DEFAULT_CHUNK_SIZE>()));
  }
};

template <typename Ctxt>
struct SafetyTestLoop<Ctxt, int> {

  SafetyTestLoop (int) {}

  template <typename R>
  void run (const R& range) const { 
  }
};


template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename ArgsTuple, typename Ctxt>
class IKDGbase: public OrderedExecutorBase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt> {

protected:

  using Base = OrderedExecutorBase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt>;
  using CtxtWL = typename Base::CtxtWL;

  using WindowWL = typename std::conditional<Base::NEEDS_PUSH, SetWindowWL<T, Cmp>, SortedRangeWindowWL<T, Cmp> >::type;

  template <typename Outer>
  struct WindowWLwrapper: public WindowWL {
    Outer& outer;

    WindowWLwrapper (Outer& outer, const Cmp& cmp):
      WindowWL (cmp), outer (outer) {}

    void push (const T& x) {
      WindowWL::push (x);
    }

    // TODO: complete this class
    void push (Ctxt* c) {
      assert (c);

      WindowWL::push (c->getActive ());

      // destroy and deallocate c
      outer.ctxtAlloc.destroy (c);
      outer.ctxtAlloc.deallocate (c, 1);
    }

    void poll (CtxtWL& wl, size_t newSize, size_t origSize) {
      WindowWL::poll (wl, newSize, origSize, outer.getCtxtMaker());
    }
  };

  std::unique_ptr<CtxtWL> currWL;
  std::unique_ptr<CtxtWL> nextWL;


  size_t windowSize;
  size_t rounds;
  size_t totalTasks;
  size_t totalCommits;
  double targetCommitRatio;

  GAccumulator<size_t> roundTasks;;
  GAccumulator<size_t> roundCommits;

  TimeAccumulator t_beginRound;
  TimeAccumulator t_expandNhood;
  TimeAccumulator t_executeSources;
  TimeAccumulator t_applyOperator;
  TimeAccumulator t_serviceAborts;
  TimeAccumulator t_performCommits;
  TimeAccumulator t_reclaimMemory;



  IKDGbase (const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple)
    : 
      Base (cmp, nhFunc, exFunc, opFunc, argsTuple),
      currWL (new CtxtWL),
      nextWL (new CtxtWL),
      windowSize (0),
      rounds (0),
      totalTasks (0),
      totalCommits (0),
      targetCommitRatio (commitRatioArg)
  {

    if (targetCommitRatio < 0.0) {
      targetCommitRatio = 0.0;
    }
    if (targetCommitRatio > 1.0) {
      targetCommitRatio = 1.0;
    }

    if (Base::ENABLE_PARAMETER) {
      assert (targetCommitRatio == 0.0);
    }

  }

  ~IKDGbase (void) {
    dumpStats ();
  }


  CtxtWL& getCurrWL (void) { 
    assert (currWL);
    return *currWL; 
  }

  CtxtWL& getNextWL (void) {
    assert (nextWL);
    return *nextWL;
  }

  void dumpStats (void) {
    reportStat_Serial (Base::loopname, "rounds", rounds);
    reportStat_Serial (Base::loopname, "committed", totalCommits);
    reportStat_Serial (Base::loopname, "total", totalTasks);
    // reportStat (loopname, "efficiency", double (totalRetires.reduce ()) / totalTasks);
    // reportStat (loopname, "avg. parallelism", double (totalRetires.reduce ()) / rounds);

    reportStat_Serial (Base::loopname, "t_expandNhood",    t_expandNhood.get());
    reportStat_Serial (Base::loopname, "t_beginRound",     t_beginRound.get());
    reportStat_Serial (Base::loopname, "t_executeSources", t_executeSources.get());
    reportStat_Serial (Base::loopname, "t_applyOperator",  t_applyOperator.get());
    reportStat_Serial (Base::loopname, "t_serviceAborts",  t_serviceAborts.get());
    reportStat_Serial (Base::loopname, "t_performCommits", t_performCommits.get());
    reportStat_Serial (Base::loopname, "t_reclaimMemory",  t_reclaimMemory.get());
  }

  //  TODO: spill range 
  template <typename WinWL, typename WL>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void spillAll (WinWL& winWL, WL& wl) {

    //    dbg::print("Spilling to winWL");

    // TODO: fix this loop, change to do_all_choice
    assert (targetCommitRatio != 0.0);
    on_each(
        [this, &wl, &winWL] (const unsigned tid, const unsigned numT) {
          while (!wl.get ().empty ()) {
            auto e  = wl.get ().back ();
            wl.get ().pop_back ();

            //            dbg::print("Spilling: ", c, " with active: ", c->getActive ());

            winWL.push (c);
          }
        });

    assert (wl.empty_all ());
    assert (!winWL.empty ());
  }

  template <typename WinWL, typename WL>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void refill (WinWL& winWL, WL& wl, size_t currCommits, size_t prevWindowSize) {

    assert (targetCommitRatio != 0.0);

    const size_t INIT_MAX_ROUNDS = 500;
    const size_t THREAD_MULT_FACTOR = 4;
    const double TARGET_COMMIT_RATIO = targetCommitRatio;
    const size_t MIN_WIN_SIZE = OpFunc::CHUNK_SIZE * getActiveThreads ();
    // const size_t MIN_WIN_SIZE = 2000000; // OpFunc::CHUNK_SIZE * getActiveThreads ();
    const size_t WIN_OVER_SIZE_FACTOR = 2;

    if (prevWindowSize == 0) {
      assert (currCommits == 0);

      // initial settings
      if (Base::NEEDS_PUSH) {
        windowSize = std::min (
            (winWL.initSize ()),
            (THREAD_MULT_FACTOR * MIN_WIN_SIZE));

      } else {
        windowSize = std::min (
            (winWL.initSize () / INIT_MAX_ROUNDS),
            (THREAD_MULT_FACTOR * MIN_WIN_SIZE));
      }
    } else {

      assert (windowSize > 0);

      double commitRatio = double (currCommits) / double (prevWindowSize);

      if (commitRatio >= TARGET_COMMIT_RATIO) {
        windowSize *= 2;
        // windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 
        // windowSize = windowSize + windowSize / 2;

      } else {
        windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 

        // if (commitRatio / TARGET_COMMIT_RATIO < 0.90) {
          // windowSize = windowSize - (windowSize / 10);
// 
        // } else {
          // windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 
        // }
      }
    }

    if (windowSize < MIN_WIN_SIZE) { 
      windowSize = MIN_WIN_SIZE;
    }

    assert (windowSize > 0);


    if (Base::NEEDS_PUSH) {
      if (winWL.empty () && (wl.size_all () > windowSize)) {
        // a case where winWL is empty and all the new elements were going into 
        // nextWL. When nextWL becomes bigger than windowSize, we must do something
        // to control efficiency. One solution is to spill all elements into winWL
        // and refill
        //

        spillAll (winWL, wl);

      } else if (wl.size_all () > (WIN_OVER_SIZE_FACTOR * windowSize)) {
        // too many adds. spill to control efficiency
        spillAll (winWL, wl);
      }
    }

    winWL.poll (wl, windowSize, wl.size_all ());
    // std::cout << "Calculated Window size: " << windowSize << ", Actual: " << wl->size_all () << std::endl;
  }

  template <typename WinWL, typename WL>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void refillRound (WinWL& winWL, WL& wl) {


    if (targetCommitRatio != 0.0) {
      size_t currCommits = roundCommits.reduceRO (); 
      size_t prevWindowSize = roundTasks.reduceRO ();
      refill (winWL, wl, currCommits, prevWindowSize);
    }

    roundCommits.reset ();
    roundTasks.reset ();
  }

  template <typename WinWL>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void beginRound (WinWL& winWL) {
    std::swap (currWL, nextWL);
    nextWL->clear_all_parallel ();

    refillRound (winWL, *currWL);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void endRound () {
    ++rounds;
    totalCommits += roundCommits.reduceRO ();
    totalTasks += roundTasks.reduceRO ();

    if (roundTasks.reduceRO () > 0) {
      assert (roundCommits.reduceRO() > 0 && "No commits this round, No progress");
    }

    // std::printf ("Round:%zd, tasks: %zd, commits: %zd\n", 
        // rounds, roundTasks.reduceRO (), roundCommits.reduceRO ());
  }

  // TODO: for debugging only

#ifndef NDEBUG
  const Ctxt* getMinCurrWL (void) const {

    substrate::PerThreadStorage<galois::optional<const Ctxt*> > perThrdMin;

    galois::do_all_choice (makeLocalRange (*currWL),
        [this, &perThrdMin] (const Ctxt* c) {
          galois::optional<const Ctxt*>& m = *(perThrdMin.getLocal ());

          if (!m || Base::ctxtCmp (c, *m)) { // c < *m
            m = c;
          }
        },
        std::make_tuple (
            galois::loopname ("getMinCurrWL"),
            galois::chunk_size<8> ()));

    const Ctxt* ret = nullptr;

    for (unsigned i = 0; i < perThrdMin.size (); ++i) {
      const galois::optional<const Ctxt*>& m = *(perThrdMin.getRemote (i));

      if (m) {
        if (!ret || Base::ctxtCmp (*m, ret)) { // ret < *m
          ret = *m;
        }
      }
    }

    return ret;
  }


  const Ctxt* getMaxCurrWL (void) const {

    substrate::PerThreadStorage<galois::optional<const Ctxt*> > perThrdMax;

    galois::do_all_choice (makeLocalRange (*currWL),
        [this, &perThrdMax] (const Ctxt* c) {
          galois::optional<const Ctxt*>& m = *(perThrdMax.getLocal ());

          if (!m || Base::ctxtCmp (*m, c)) { // *m < c
            m = c;
          } 
        },
        std::make_tuple (
            galois::loopname ("getMaxCurrWL"),
            galois::chunk_size<8> ()));

    const Ctxt* ret = nullptr;

    for (unsigned i = 0; i < perThrdMax.size (); ++i) {
      const galois::optional<const Ctxt*>& m = *(perThrdMax.getRemote (i));

      if (m) {
        if (!ret || Base::ctxtCmp (ret, *m)) { // ret < *m
          ret = *m;
        }
      }
    }

    return ret;
  }
#endif

};



} // end namespace runtime
} // end namespace galois

#endif // GALOIS_RUNTIME_IKDG_BASE_H


