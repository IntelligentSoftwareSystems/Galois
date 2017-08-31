/** Billiards Simulation Finding Partial Order -*- C++ -*-
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
 * Billiards Simulation Finding Partial Order
 *
 * @author <ahassaan@ices.utexas.edu>
 */


#ifndef _BILLIARDS_PART_ORD_H_
#define _BILLIARDS_PART_ORD_H_

#include <set>
#include <limits>
#include <iostream>
#include <fstream>

#include <cstdio>
#include <cassert>

#include <boost/iterator/counting_iterator.hpp>


#include "Galois/Accumulator.h"
#include "Galois/Markable.h"
#include "Galois/DoAllWrap.h"
#include "Galois/PerThreadContainer.h"

#include "Galois/Runtime/Executor_OnEach.h"
#include "Galois/Substrate/CompilerSpecific.h"


#include "dependTest.h"
#include "Billiards.h"


class BilliardsPOsortedVec;

class BilliardsPOunsorted: public Billiards<BilliardsPOunsorted, Table<Ball> > {

  typedef Galois::Markable<Event> MEvent;
  typedef Galois::PerThreadVector<MEvent> WLTy;
  typedef Galois::PerThreadVector<Event> ILTy;

  typedef Galois::PerThreadVector<Event> AddListTy;

  friend class BilliardsPOsortedVec;

  using Tbl_t = Table<Ball>;


public:


  // static const unsigned CHUNK_SIZE = 1;

  virtual const std::string version () const { return "Parallel Partially Ordered with Unsorted workList"; }


  virtual size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {

    Galois::Substrate::getThreadPool().burnPower (Galois::getActiveThreads ());

    WLTy workList;
    // workList.fill_serial (initEvents.begin (), initEvents.end (), &WLTy::Cont_ty::push_back);
    Galois::do_all_choice (
        Galois::Runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        [&workList] (const Event& e) {
          workList.get ().push_back (MEvent (e));
        },
        std::make_tuple(
          Galois::loopname("fill_init"),
          Galois::chunk_size<32> ()));


    size_t i = runSimInternal<FindIndepEvents, SimulateIndepEvents, AddNextEvents, RemoveSimulatedEvents> (
        table, workList, endtime, enablePrints);

    Galois::Substrate::getThreadPool ().beKind ();

    return i;
  }


private:

template <typename _CleanupFunc>
GALOIS_ATTRIBUTE_PROF_NOINLINE static void updateODG_clean (WLTy& workList, const unsigned currStep) {
  Galois::Runtime::on_each_impl (_CleanupFunc (workList, currStep), "remove_simulated_events");
  // Galois::Runtime::do_all_coupled (
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

    Galois::TimeAccumulator findTimer;
    Galois::TimeAccumulator simTimer;
    Galois::TimeAccumulator addTimer;
    Galois::TimeAccumulator sweepTimer;

    ILTy indepList;

    AddListTy addList;
    unsigned currStep = 0;

    Accumulator findIter;
    size_t iter = 0;


    while (!workList.empty_all ()) {


      // printf ("currStep = %d, workList.size () = %zd\n", currStep, workList.size_all ());

      findTimer.start ();
      Galois::do_all_choice (Galois::Runtime::makeLocalRange (workList),
          _FindIndepFunc (indepList, workList, currStep, findIter), 
          std::make_tuple(
            Galois::loopname("find_indep_events"), 
            Galois::chunk_size<1> ()));

      findTimer.stop ();

      // printf ("currStep= %d, indepList.size ()= %zd, workList.size ()= %zd\n", 
          // currStep, indepList.size_all (), workList.size_all ());

      assert (!indepList.empty_all () && "couldn't find any independent events");

      simTimer.start ();
      std::for_each (indepList.begin_all (), indepList.end_all (), _SimulateFunc ());
      simTimer.stop ();

      addTimer.start ();
      // Galois::Runtime::do_all_coupled (indepList, 
      Galois::do_all_choice (Galois::Runtime::makeLocalRange (indepList), 
          _AddNextFunc (workList, addList, table, endtime, enablePrints), 
          std::make_tuple(
            Galois::loopname("add_next_events"), 
            Galois::chunk_size<1> ()));
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

    Galois::Substrate::getThreadPool ().burnPower (Galois::getActiveThreads ());

    WLTy workList;
    // workList.fill_serial (initEvents.begin (), initEvents.end (), &WLTy::Cont_ty::push_back);
    Galois::do_all_choice (
        Galois::Runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        [&workList] (const Event& e) {
          workList.get ().push_back (MEvent (e));
        },
        std::make_tuple(
          Galois::loopname("fill_init"), 
          Galois::chunk_size<32> ()));

    // sort events
    // for (unsigned r = 0; r < workList.numRows (); ++r) {
      // std::sort (workList[r].begin (), workList[r].end (), Event::Comparator ());
    // }
    Galois::on_each (
        [&workList] (const unsigned tid, const unsigned numT) {
          unsigned r = tid;
          assert (r < workList.numRows ());
           std::sort (workList[r].begin (), workList[r].end (), Event::Comparator ());

        },
        Galois::loopname("initsort"));



    size_t i =  BilliardsPOunsorted::runSimInternal
      <FindIndepEvents, BilliardsPOunsorted::SimulateIndepEvents, 
           BilliardsPOunsorted::AddNextEvents, RemoveAndSortEvents> 
             (table, workList, endtime, enablePrints);

    Galois::Substrate::getThreadPool ().beKind ();

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

