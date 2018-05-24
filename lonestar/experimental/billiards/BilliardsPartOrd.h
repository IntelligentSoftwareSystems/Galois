#ifndef _BILLIARDS_PART_ORD_H_
#define _BILLIARDS_PART_ORD_H_

#include <set>
#include <limits>
#include <iostream>
#include <fstream>

#include <cstdio>
#include <cassert>

#include <boost/iterator/counting_iterator.hpp>


#include "galois/Reduction.h"
#include "galois/Markable.h"
#include "galois/DoAllWrap.h"
#include "galois/PerThreadContainer.h"

#include "galois/runtime/Executor_OnEach.h"
#include "galois/substrate/CompilerSpecific.h"


#include "dependTest.h"
#include "Billiards.h"


class BilliardsPOsortedVec;

class BilliardsPOunsorted: public Billiards<BilliardsPOunsorted, Table<Ball> > {

  typedef galois::Markable<Event> MEvent;
  typedef galois::PerThreadVector<MEvent> WLTy;
  typedef galois::PerThreadVector<Event> ILTy;

  typedef galois::PerThreadVector<Event> AddListTy;

  friend class BilliardsPOsortedVec;

  using Tbl_t = Table<Ball>;


public:


  // static const unsigned CHUNK_SIZE = 1;

  virtual const std::string version () const { return "Parallel Partially Ordered with Unsorted workList"; }


  virtual size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {

    galois::substrate::getThreadPool().burnPower (galois::getActiveThreads ());

    WLTy workList;
    // workList.fill_serial (initEvents.begin (), initEvents.end (), &WLTy::Cont_ty::push_back);
    galois::do_all_choice (
        galois::runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        [&workList] (const Event& e) {
          workList.get ().push_back (MEvent (e));
        },
        std::make_tuple(
          galois::loopname("fill_init"),
          galois::chunk_size<32> ()));


    size_t i = runSimInternal<FindIndepEvents, SimulateIndepEvents, AddNextEvents, RemoveSimulatedEvents> (
        table, workList, endtime, enablePrints);

    galois::substrate::getThreadPool ().beKind ();

    return i;
  }


private:

template <typename _CleanupFunc>
GALOIS_ATTRIBUTE_PROF_NOINLINE static void updateODG_clean (WLTy& workList, const unsigned currStep) {
  galois::on_each (_CleanupFunc (workList, currStep), "remove_simulated_events");
  // galois::runtime::do_all_coupled (
      // boost::counting_iterator<unsigned> (0),
      // boost::counting_iterator<unsigned> (workList.numRows ()), 
      // _CleanupFunc (workList, currStep),
      // "remove_simulated_events", 1);
}

template <typename _FindIndepFunc, typename _SimulateFunc,
          typename _AddNextFunc, typename _CleanupFunc>
static size_t runSimInternal (Tbl_t& table, WLTy& workList, const FP& endtime, bool enablePrints=false) {
    // TODO: Explain separation of simulating events and adding
    // new events

    galois::TimeAccumulator findTimer;
    galois::TimeAccumulator simTimer;
    galois::TimeAccumulator addTimer;
    galois::TimeAccumulator sweepTimer;

    ILTy indepList;

    AddListTy addList;
    unsigned currStep = 0;

    Accumulator findIter;
    size_t iter = 0;


    while (!workList.empty_all ()) {


      // printf ("currStep = %d, workList.size () = %zd\n", currStep, workList.size_all ());

      findTimer.start ();
      galois::do_all_choice (galois::runtime::makeLocalRange (workList),
          _FindIndepFunc (indepList, workList, currStep, findIter), 
          std::make_tuple(
            galois::loopname("find_indep_events"), 
            galois::chunk_size<1> ()));

      findTimer.stop ();

      // printf ("currStep= %d, indepList.size ()= %zd, workList.size ()= %zd\n", 
          // currStep, indepList.size_all (), workList.size_all ());

      assert (!indepList.empty_all () && "couldn't find any independent events");

      simTimer.start ();
      std::for_each (indepList.begin_all (), indepList.end_all (), _SimulateFunc ());
      simTimer.stop ();

      addTimer.start ();
      // galois::runtime::do_all_coupled (indepList, 
      galois::do_all_choice (galois::runtime::makeLocalRange (indepList), 
          _AddNextFunc (workList, addList, table, endtime, enablePrints), 
          std::make_tuple(
            galois::loopname("add_next_events"), 
            galois::chunk_size<1> ()));
      addTimer.stop ();


      sweepTimer.start ();
      updateODG_clean<_CleanupFunc> (workList, currStep);
      sweepTimer.stop ();


      ++currStep;
      iter += indepList.size_all ();
      indepList.clear_all_parallel ();

    } 


    if (false) {
      updateODG_clean<_CleanupFunc> (workList, currStep);

      if (!workList.empty_all ()) {
        std::cerr << "Still valid events that need processing?" << std::endl;
        std::abort ();
      }

    }

    std::cout << "Total number of currStep=" << currStep << std::endl;
    std::cout << "Total number of events=" << iter << std::endl;
    std::cout << "Average parallelism=" << (double (iter)/double (currStep)) << std::endl;
    std::cout << "Total Iterations spent in finding independent evens= " << findIter.reduce () << std::endl;
    std::cout << "Time spent in FINDING independent events= " << findTimer.get () << std::endl;
    std::cout << "Time spent in SIMULATING independent events= " << simTimer.get () << std::endl;
    std::cout << "Time spent in ADDING new events= " << addTimer.get () << std::endl;
    std::cout << "Time spent in REMOVING simulated events= " << sweepTimer.get () << std::endl;

    return iter;
  }

private:



  struct FindIndepEvents {
    ILTy& indepList;
    WLTy& workList;
    unsigned currStep;
    Accumulator& findIter;

    OrderDepTest dt;

    FindIndepEvents (
        ILTy& _indepList,
        WLTy& _workList,
        unsigned _currStep,
        Accumulator& _findIter)
      :
        indepList (_indepList),
        workList (_workList) ,
        currStep (_currStep),
        findIter (_findIter)

    {}


    GALOIS_ATTRIBUTE_PROF_NOINLINE void updateODG_test (MEvent& e) const {


      if (!e.marked ()) {

        bool indep = true;

        for (unsigned r = 0; r < workList.numRows (); ++r) {
          for (WLTy::local_iterator i = workList[r].begin (), ei = workList[r].end ();
              i != ei; ++i) {

            findIter += 1;

            if ((!i->marked () || (i->version () >= currStep))
                && (e.get () > (*i))) { 
              // >= is used to eliminate duplicate events and different events with same
              // time but a common object between them

              if (dt.dependsOn (e, *i)) {
                indep = false;
                break;
              }
            }

          }

          if (!indep) {
            break;
          }
        }


        if (indep) {
          indepList.get ().push_back (e);
          e.mark  (currStep);
        }
      }

    }

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (MEvent& e) const {
      updateODG_test (e);
    }

  };

  struct SimulateIndepEvents {

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Event& event) const {
      event.simulate();
    }
  };

  struct AddNextEvents {

    WLTy& workList;
    AddListTy& addList;
    Tbl_t& table;
    const FP& endtime;
    bool enablePrints;

    AddNextEvents (
        WLTy& _workList,
        AddListTy& _addList,
        Tbl_t& _table,
        const FP& _endtime,
        bool _enablePrints)
      :
        workList (_workList),
        addList (_addList),
        table (_table),
        endtime (_endtime),
        enablePrints (_enablePrints)
    {}


    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Event& event) const {
      addList.get().clear ();

      table.addNextEvents (event, addList.get (), endtime);

      for (auto a = addList.get ().begin (), ea = addList.get ().end ();
          a != ea; ++a ) {

        workList.get ().push_back (MEvent (*a));
      }

    }


  };


  struct RemoveSimulatedEvents {
    WLTy& workList;
    unsigned currStep;

    RemoveSimulatedEvents (
        WLTy& _workList,
        unsigned _currStep)
      :
        workList (_workList),
        currStep (_currStep)
    {}

    GALOIS_ATTRIBUTE_PROF_NOINLINE void updateODG_clean (const unsigned r) {

      for (WLTy::local_iterator i = workList[r].begin (), ei = workList[r].end (); i != ei;) {

        if (i->marked () && i->version () <= currStep) {
          WLTy::local_iterator tmp = workList[r].end ();
          --tmp; // last element

          std::swap (*i, *tmp);

          workList[r].erase (tmp);

          if (i == tmp) { break; }

          ei = workList[r].end ();

        } else {
          ++i;
        }
      }
    }

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const unsigned tid, const unsigned numT) {
      assert (tid < workList.numRows ());
      updateODG_clean (tid);
    }
  
  };



};


class BilliardsPOsortedVec: public Billiards<BilliardsPOsortedVec, Table<Ball> > {

  typedef BilliardsPOunsorted::MEvent MEvent;
  typedef BilliardsPOunsorted::WLTy WLTy;
  typedef BilliardsPOunsorted::ILTy ILTy;

  using Tbl_t = Table<Ball>;

public:

  virtual const std::string version () const { return "Parallel Partially Ordered with sorted vector  workList"; }



  virtual size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {

    galois::substrate::getThreadPool ().burnPower (galois::getActiveThreads ());

    WLTy workList;
    // workList.fill_serial (initEvents.begin (), initEvents.end (), &WLTy::Cont_ty::push_back);
    galois::do_all_choice (
        galois::runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        [&workList] (const Event& e) {
          workList.get ().push_back (MEvent (e));
        },
        std::make_tuple(
          galois::loopname("fill_init"), 
          galois::chunk_size<32> ()));

    // sort events
    // for (unsigned r = 0; r < workList.numRows (); ++r) {
      // std::sort (workList[r].begin (), workList[r].end (), Event::Comparator ());
    // }
    galois::on_each (
        [&workList] (const unsigned tid, const unsigned numT) {
          unsigned r = tid;
          assert (r < workList.numRows ());
           std::sort (workList[r].begin (), workList[r].end (), Event::Comparator ());

        },
        galois::loopname("initsort"));



    size_t i =  BilliardsPOunsorted::runSimInternal
      <FindIndepEvents, BilliardsPOunsorted::SimulateIndepEvents, 
           BilliardsPOunsorted::AddNextEvents, RemoveAndSortEvents> 
             (table, workList, endtime, enablePrints);

    galois::substrate::getThreadPool ().beKind ();

    return i;
  }

private:

  struct FindIndepEvents {
    ILTy& indepList;
    WLTy& workList;
    unsigned currStep;
    Accumulator& findIter;

    OrderDepTest dt;

    FindIndepEvents (
        ILTy& _indepList,
        WLTy& _workList,
        unsigned _currStep, 
        Accumulator& _findIter) 
      :
        indepList (_indepList),
        workList (_workList),
        currStep (_currStep),
        findIter (_findIter)
    {} 

    GALOIS_ATTRIBUTE_PROF_NOINLINE void updateODG_test (MEvent& e) const {
      if (!e.marked ()) {

        bool indep = true;

        for (unsigned r = 0; r < workList.numRows (); ++r) {
          for (WLTy::local_iterator i = workList[r].begin (), ei = workList[r].end ();
              (i != ei) && (i->get () < e.get ()); ++i) {

            findIter += 1;

            if (!i->marked () || (i->version () >= currStep))  {

              if (dt.dependsOn (e, *i)) {
                indep = false;
                break;
              }
            }
            
          }

          if (!indep) {
            break;
          }

        }


        if (indep) {
          indepList.get ().push_back (e);
          e.mark (currStep);
        }
        
      } // end outer if
    }

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (MEvent& e) const {
      updateODG_test (e);
    }
  };


  struct RemoveAndSortEvents: public BilliardsPOunsorted::RemoveSimulatedEvents {
    typedef BilliardsPOunsorted::RemoveSimulatedEvents SuperTy;

    RemoveAndSortEvents (
        WLTy& _workList,
        unsigned _currStep)
      :
        SuperTy (_workList, _currStep)
    {}

    GALOIS_ATTRIBUTE_PROF_NOINLINE void updateODG_clean (const unsigned r) {
      // first remove simulated events
      SuperTy::updateODG_clean (r);

      // now sort the rest
      std::sort (workList[r].begin (), workList[r].end (), Event::Comparator ());
    }

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const unsigned tid, const unsigned numT) {
      assert (tid < workList.numRows ());
      updateODG_clean (tid);
    }

  };

};



#endif // _BILLIARDS_PART_ORD_H_

