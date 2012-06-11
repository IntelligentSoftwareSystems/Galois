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

#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/DoAll.h"
#include "Galois/util/Marked.h"


#include "dependTest.h"
#include "Billiards.h"


typedef Galois::GSimpleReducible<size_t, std::plus<size_t> > Accumulator;




// METHOD 1
// =========
//
// O (B^2 * number of rounds (or iterations of outermost while loop)) parallel algorithm
//
// workList = initial events
//
// while (!workList.empty ()) {
//  indepList.clear ();
//
//  for each item i in workList {
//     isindep = true;
//     for each item j in workList {
//       if (i !=j && i > j) {
//         test i against j
//         if (i is dependent of j) {
//            put i in remainingList
//            isindep = false;
//            break;
//         }
//       }
//     }
//
//     if isindep {
//       put i in indepList
//     }
//  }
//
//  for each item i in indepList {
//    simulate i
//    for each newly generated event e {
//      add e to remainingList;
//    }
//  }
//
//  workList = remainingList;
//  remainingList = new ...
// }
//
// Method 1B:
// Typically, a few elements will make it to the indepList. Therefore,
// we can keep on removing (and adding) items from workList and get rid of 
// remainingList. We'll have to check every element against existing items
// in the indepList
//
// Method 1C:
// Mark independent events in the workList for removal, and 
// copy independent to indepList. Now workList contains 
// marked events, which can be removed easily (e.g. by swapping with last
// element of vector). This gets rid of the remainingList 
//
//
//
//
//    
//

class BilliardsPOunsortedSerial: public Billiards {
  typedef std::vector<Event> WLTy;
  typedef std::vector<Event> ILTy;

public:

  virtual const std::string version () const { return "Partially Ordered with Unsorted workList"; }

  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) {

    Galois::TimeAccumulator findTimer;
    Galois::TimeAccumulator simTimer;
    
    WLTy* workList = new WLTy (initEvents.begin (), initEvents.end ());
    WLTy* remainingList = new WLTy ();

    ILTy indepList;

    std::vector<Event> addList;

    size_t iter = 0;
    unsigned currStep = 0;
    size_t findIter = 0;

    while (!workList->empty ()) {

      findTimer.start ();
      findIndepEvents (indepList, *remainingList, *workList, findIter, enablePrints);
      findTimer.stop ();

      if (indepList.empty ()) {
        std::cerr <<  "No independent events?? No progress??" << std::endl;
        std::abort ();
      }
      

      simTimer.start ();
      simulateIndepEvents (indepList, *remainingList, addList,
          table, endtime, enablePrints);
      simTimer.stop ();


      if (enablePrints) {
        std::cout << "step=" << currStep << ", indepList.size=" << indepList.size ()  
          << ", workList.size=" << workList->size () << std::endl;
      }

      iter += indepList.size ();
      ++currStep;

      std::swap (workList, remainingList);
      remainingList->clear ();
      indepList.clear ();


    }

    std::cout << "Total number of currStep=" << currStep << std::endl;
    std::cout << "Total number of events=" << iter << std::endl;
    std::cout << "Average parallelism=" << (double (iter)/double (currStep)) << std::endl;
    std::cout << "Total Iterations spent in finding independent evens= " << findIter << std::endl;
    std::cout << "Time spent in FINDING independent events= " << findTimer.get () << std::endl;
    std::cout << "Time spent in SIMULATING independent events= " << simTimer.get () << std::endl;

    delete workList;
    delete remainingList;

    return iter;
  }

private:

  void findIndepEvents (ILTy& indepList, WLTy& remainingList, WLTy& workList, 
      size_t& findIter, bool enablePrints) {
    for (WLTy::const_iterator i = workList.begin ()
        , ei = workList.end (); i != ei; ++i) {


      bool indep = true;

      for (WLTy::const_iterator j = workList.begin ()
          , ej = workList.end (); j != ej; ++j) {

        ++findIter;

        // if event i happens after j, then check for order independence
        if (i != j 
            && (*i) > (*j)) {

          // DEBUG: std::cout << "Testing e1=" << i->str () << ", e2=" << j->str () << std::endl;
          if (OrderDepTest::dependsOn (*i, *j)) {
            //DEBUG: std::cout << "Are dependent" << std::endl;
            indep = false;
            break;

          } else {
            //DEBUG: std::cout << "Are independent" << std::endl;
          }
        }
      }

      if (indep) {
        indepList.push_back (*i);

      } else {
        remainingList.push_back (*i);
      }


    } // end for
  }


  void simulateIndepEvents (ILTy& indepList, WLTy& remainingList, 
      std::vector<Event>& addList, Table& table, const double endtime, bool enablePrints) {

    for (WLTy::iterator i = indepList.begin (), ei = indepList.end ();
        i != ei; ++i) {

      if (enablePrints) {
        std::cout << "Processing event=" << i->str () << std::endl;
      }

      addList.clear ();
      i->simulate (addList, table, endtime);

      for (std::vector<Event>::iterator a = addList.begin (), ea = addList.end ();
          a != ea; ++a) {

        remainingList.push_back (*a);

        if (enablePrints) {
          std::cout << "Adding event=" << a->str () << std::endl;
        }
      }

      if (enablePrints) {
        table.printState (std::cout);
      }

    }
  }



};

// Method 2:
// =========
//
// O ((BlogB + B^2) * number of rounds) parallel algorithm
// Order workList by priority in an iterable e.g. set or sorted vector
//
//
// workList = initial events 
//
//
// while (!workList.empty ()) {
//  sort workList if not already sorted
//
//  readList.clear ();
//
//  add first item to indepList
//
//  for each item i after first in sorted workList {
//     indep = true;
//
//     for each item j in indepList {
//       test i against j
//       if (i not indep of j) {
//         indep = false
//         break;
//       }
//     }
//
//     if (!indep) { 
//        continue;
//     }
//
//     for each item j upto i-1 in sorted workList do {
//       test i against higher priority item j
//       if i is not indep of j {
//         indep = false;
//         break;
//       }
//     }
//
//     if (indep) {
//       add i to indepList
//       remove i from workList
//     }
//  }
//
//  for each item i in indepList {
//    simulate i
//    for each newly generated event e {
//      add e to workList;
//    }
//    
//  }
//
//  workList = remainingList;
//  remainingList = new ...
// }
//      
// Typically, a few elements will make it to the indepList. Therefore,
// it's more efficient to make the workList an iterable priority queue 
// and remove the items from priority queue that make it to the indepList.
//
// Possible implementations
// RB tree from stl
// skiplist
//
//
//        
// METHOD 3:
// =========
//

class BilliardsPOsortedSerial: public Billiards {

  typedef std::set<Event, Event::Comparator> WLTy;
  typedef std::vector<Event> ILTy;

public:

  virtual const std::string version () const { return "Partially Ordered with Unsorted workList"; }

  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) {

    const bool SHOW_PARAMETER = false;

    std::ofstream* statsFile = NULL;
    if (SHOW_PARAMETER) {
      statsFile = new std::ofstream ("parameter_billiards.csv");
      (*statsFile) << "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE" << std::endl;
    }


    Galois::TimeAccumulator findTimer;
    Galois::TimeAccumulator simTimer;

    WLTy workList (initEvents.begin (), initEvents.end ());

    ILTy indepList;

    std::vector<Event> addList;

    size_t iter = 0;
    unsigned currStep = 0;
    size_t findIter = 0;

    while (!workList.empty ()) {

      findTimer.start ();
      findIndepEvents (indepList, workList, findIter, enablePrints);
      findTimer.stop ();
      
      if (indepList.empty ()) {
        std::cerr <<  "No independent events?? No progress??" << std::endl;
        std::abort ();
      }

      simTimer.start ();
      simulateIndepEvents (indepList, workList, addList, 
          table, endtime, enablePrints);
      simTimer.stop ();

      if (SHOW_PARAMETER) {
        (*statsFile) << "foreach, " << currStep << ", " << indepList.size ()
          << ", " << (workList.size () + indepList.size ()) << std::endl;
      }

      iter+= indepList.size ();
      ++currStep;

      indepList.clear ();
    }

    std::cout << "Total number of currStep=" << currStep << std::endl;
    std::cout << "Total number of events=" << iter << std::endl;
    std::cout << "Average parallelism=" << (double (iter)/double (currStep)) << std::endl;
    std::cout << "Total Iterations spent in finding independent evens= " << findIter << std::endl;
    std::cout << "Time spent in FINDING independent events= " << findTimer.get () << std::endl;
    std::cout << "Time spent in SIMULATING independent events= " << simTimer.get () << std::endl;


    if (SHOW_PARAMETER) {
      delete statsFile;
      statsFile = NULL;
    }


    return iter;

  }

protected:

  void findIndepEvents (ILTy& indepList, WLTy& workList, size_t& findIter, bool enablePrints) {

    indepList.clear ();

    for (WLTy::iterator i = workList.begin (), ei = workList.end ();
        i != ei;) {

      bool indep = true;

      for (ILTy::const_iterator j = indepList.begin (), ej = indepList.end ();
          j != ej; ++j) {

        ++findIter;

        assert ((*i) > (*j));

        if (OrderDepTest::dependsOn (*i, *j)) {
          indep = false;
          break;
        }
      } // end for indepList



      if (indep) {

        // from start upto i in priority order
        for (WLTy::iterator j = workList.begin (), ej = workList.end (); 
            (j != i) && ((*j) < (*i)); ++j) {

          ++findIter;

          if (OrderDepTest::dependsOn (*i, *j)) {
            indep = false;
            break;
          }
        }
      }



      if (indep) {
        // add to indepList
        indepList.push_back (*i);

        // remove from workList
        WLTy::iterator tmp = i;
        ++i;
        workList.erase (tmp);

      } else {
        ++i;
      }

    }

  }


  void simulateIndepEvents (ILTy& indepList, WLTy& workList, 
      std::vector<Event>& addList, Table& table, const double endtime, bool enablePrints) { 


    for (ILTy::iterator i = indepList.begin (), ei = indepList.end ();
        i != ei; ++i) {

      addList.clear ();
      i->simulate (addList, table, endtime);

      for (std::vector<Event>::const_iterator a = addList.begin (), ea = addList.end ();
          a != ea; ++a) {

        workList.insert (*a);
      }
    }
  }



};


class BilliardsPOsortedVec;

class BilliardsPOunsorted: public Billiards {

  typedef GaloisRuntime::PerThreadWLfactory<Markable<Event> >::PerThreadVector WLTy;
  typedef GaloisRuntime::PerThreadWLfactory<Event>::PerThreadVector ILTy;

  typedef GaloisRuntime::PerCPU< std::vector<Event> > AddListTy;

  friend class BilliardsPOsortedVec;


public:

  virtual const std::string version () const { return "Parallel Partially Ordered with Unsorted workList"; }


  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) {

    WLTy workList;
    workList.fill_init (initEvents.begin (), initEvents.end (), &WLTy::Cont_ty::push_back);

    return runSimInternal<FindIndepEvents, SimulateIndepEvents, AddNextEvents, RemoveSimulatedEvents> (
        table, workList, endtime, enablePrints);
  }


private:

template <typename _FindIndepFunc, typename _SimulateFunc,
          typename _AddNextFunc, typename _CleanupFunc>
static size_t runSimInternal (Table& table, WLTy& workList, const double endtime, bool enablePrints=false) {
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


    do {

      indepList.clear_all ();


      findTimer.start ();
      GaloisRuntime::do_all_coupled (workList, 
          _FindIndepFunc (indepList, workList, currStep, findIter), "find_indep_events");
      findTimer.stop ();

      // printf ("currStep= %d, indepList.size ()= %zd, workList.size ()= %zd\n", 
          // currStep, indepList.size_all (), workList.size_all ());

      simTimer.start ();
      GaloisRuntime::do_all_serial (indepList, _SimulateFunc (), "simulate_indep_events");
      simTimer.stop ();

      addTimer.start ();
      GaloisRuntime::do_all_coupled (indepList, 
          _AddNextFunc (workList, addList, table, endtime, enablePrints), "add_next_events");
      addTimer.stop ();


      sweepTimer.start ();

      GaloisRuntime::do_all_coupled (
          boost::counting_iterator<unsigned> (0),
          boost::counting_iterator<unsigned> (workList.numRows ()), 
          _CleanupFunc (workList, currStep),
          "remove_simulated_events");
      sweepTimer.stop ();

      ++currStep;
      iter += indepList.size_all ();


    } while (!indepList.empty_all ()); 


    if (true) {
      GaloisRuntime::do_all_coupled (
          boost::counting_iterator<unsigned> (0),
          boost::counting_iterator<unsigned> (workList.numRows ()), 
          _CleanupFunc (workList, currStep),
          "remove_simulated_events");

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


    void operator () (Markable<Event>& e) {


      if (!e.marked ()) {

        bool indep = true;

        for (unsigned r = 0; r < workList.numRows (); ++r) {
          for (WLTy::iterator i = workList.begin (r), ei = workList.end (r);
              i != ei; ++i) {

            findIter.get () += 1;

            if ((!i->marked () || (i->version () >= currStep))
                && (e > (*i))) { 
              // >= is used to eliminate duplicate events and different events with same
              // time but a common object between them

              if (OrderDepTest::dependsOn (e, *i)) {
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

  };

  struct SimulateIndepEvents {

    void operator () (Event& event) {
      event.simulateCollision ();
    }
  };

  struct AddNextEvents {

    WLTy& workList;
    AddListTy& addList;
    Table& table;
    double endtime;
    bool enablePrints;

    AddNextEvents (
        WLTy& _workList,
        AddListTy& _addList,
        Table& _table,
        double _endtime,
        bool _enablePrints)
      :
        workList (_workList),
        addList (_addList),
        table (_table),
        endtime (_endtime),
        enablePrints (_enablePrints)
    {}


    void operator () (Event& event) {
      addList.get ().clear ();

      event.addNextEvents (addList.get (), table, endtime);

      for (std::vector<Event>::iterator a = addList.get ().begin (), ea = addList.get ().end ();
          a != ea; ++a ) {

        workList.get ().push_back (Markable<Event> (*a));
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

    void operator () (unsigned r) {
      assert (r < workList.numRows ());

      for (WLTy::iterator i = workList.begin (r), ei = workList.end (r); i != ei;) {

        if (i->marked () && i->version () <= currStep) {
          WLTy::iterator tmp = workList.end (r);
          --tmp; // last element

          std::swap (*i, *tmp);

          workList[r].erase (tmp);

          ei = workList.end (r);

        } else {
          ++i;
        }
      }
    }
  
  };



};


class BilliardsPOsortedVec: public Billiards {

  typedef BilliardsPOunsorted::WLTy WLTy;
  typedef BilliardsPOunsorted::ILTy ILTy;

public:

  virtual const std::string version () const { return "Parallel Partially Ordered with sorted vector  workList"; }



  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) {

    WLTy workList;
    workList.fill_init (initEvents.begin (), initEvents.end (), &WLTy::Cont_ty::push_back);

    // sort events
    for (unsigned r = 0; r < workList.numRows (); ++r) {
      std::sort (workList.begin (r), workList.end (r), Event::Comparator ());
    }



    return BilliardsPOunsorted::runSimInternal
      <FindIndepEvents, BilliardsPOunsorted::SimulateIndepEvents, 
           BilliardsPOunsorted::AddNextEvents, RemoveAndSortEvents> 
             (table, workList, endtime, enablePrints);
  }

private:

  struct FindIndepEvents {
    ILTy& indepList;
    WLTy& workList;
    unsigned currStep;
    Accumulator& findIter;

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


    void operator () (Markable<Event>& e) {

      if (!e.marked ()) {

        bool indep = true;

        for (unsigned r = 0; r < workList.numRows (); ++r) {
          for (WLTy::iterator i = workList.begin (r), ei = workList.end (r);
              (i != ei) && ((*i) < e); ++i) {

            findIter.get () += 1;

            if (!i->marked () || (i->version () >= currStep))  {

              if (OrderDepTest::dependsOn (e, *i)) {
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
  };


  struct RemoveAndSortEvents: public BilliardsPOunsorted::RemoveSimulatedEvents {
    typedef BilliardsPOunsorted::RemoveSimulatedEvents SuperTy;

    RemoveAndSortEvents (
        WLTy& _workList,
        unsigned _currStep)
      :
        SuperTy (_workList, _currStep)
    {}

    void operator () (unsigned r) {
      // first remove simulated events
      SuperTy::operator () (r);

      // now sort the rest
      std::sort (workList.begin (r), workList.end (r), Event::Comparator ());
    }

  };

};



#endif // _BILLIARDS_PART_ORD_H_

