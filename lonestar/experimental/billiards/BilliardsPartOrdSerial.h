/** Billiards Simulation Finding Partial Order Serially -*- C++ -*-
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
 * Billiards Simulation Finding Partial Order Serially
 *
 * @author <ahassaan@ices.utexas.edu>
 */


#ifndef BILLIARDS_PART_ORD_SERIAL_H
#define BILLIARDS_PART_ORD_SERIAL_H

#include <set>
#include <limits>
#include <iostream>
#include <fstream>

#include <cstdio>
#include <cassert>

#include <boost/iterator/counting_iterator.hpp>


#include "galois/Accumulator.h"
#include "galois/Markable.h"
#include "galois/PerThreadContainer.h"

#include "galois/runtime/Executor_DoAll.h"
#include "galois/substrate/CompilerSpecific.h"


#include "dependTest.h"
#include "Billiards.h"
#include "SimLogger.h"

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

  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false) {

    galois::TimeAccumulator findTimer;
    galois::TimeAccumulator simTimer;
    
    WLTy* workList = new WLTy (initEvents.begin (), initEvents.end ());
    WLTy* remainingList = new WLTy ();

    ILTy indepList;

    std::vector<Event> addList;

    size_t iter = 0;
    unsigned step = 0;
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
        std::cout << "step=" << step << ", indepList.size=" << indepList.size ()  
          << ", workList.size=" << workList->size () << std::endl;
      }

      iter += indepList.size ();
      ++step;

      std::swap (workList, remainingList);
      remainingList->clear ();
      indepList.clear ();


    }

    std::cout << "Total number of step=" << step << std::endl;
    std::cout << "Total number of events=" << iter << std::endl;
    std::cout << "Average parallelism=" << (double (iter)/double (step)) << std::endl;
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
      std::vector<Event>& addList, Table& table, const FP& endtime, bool enablePrints) {

    for (WLTy::iterator i = indepList.begin (), ei = indepList.end ();
        i != ei; ++i) {

      if (enablePrints) {
        std::cout << "Processing event=" << i->str () << std::endl;
      }

      addList.clear ();
      i->simulate ();
      table.addNextEvents (*i, addList, endtime);

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

  static const bool SHOW_PARAMETER = false;

  static const bool PRODUCE_LOG = true;
public:

  virtual const std::string version () const { return "Partially Ordered with Unsorted workList"; }

  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false) {


    std::ofstream* statsFile = NULL;
    if (SHOW_PARAMETER) {
      statsFile = new std::ofstream ("parameter_billiards.csv");
      (*statsFile) << "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE" << std::endl;
    }

    SimLogger* logger = nullptr;

    if (PRODUCE_LOG) {
      logger = new SimLogger ();
      table.writeConfig ();
      table.ballsToCSV ();
    }

    galois::TimeAccumulator findTimer;
    galois::TimeAccumulator simTimer;

    WLTy workList (initEvents.begin (), initEvents.end ());

    ILTy indepList;

    std::vector<Event> addList;

    size_t iter = 0;
    unsigned step = 0;
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
          table, endtime, logger, enablePrints);
      simTimer.stop ();

      if (SHOW_PARAMETER) {
        (*statsFile) << "foreach, " << step << ", " << indepList.size ()
          << ", " << (workList.size () + indepList.size ()) << std::endl;
      }

      iter+= indepList.size ();
      ++step;

      if (PRODUCE_LOG) { logger->incStep (); }

      indepList.clear ();
    }

    std::cout << "Total number of step=" << step << std::endl;
    std::cout << "Total number of events=" << iter << std::endl;
    std::cout << "Average parallelism=" << (double (iter)/double (step)) << std::endl;
    std::cout << "Total Iterations spent in finding independent evens= " << findIter << std::endl;
    std::cout << "Time spent in FINDING independent events= " << findTimer.get () << std::endl;
    std::cout << "Time spent in SIMULATING independent events= " << simTimer.get () << std::endl;


    if (SHOW_PARAMETER) {
      delete statsFile;
      statsFile = NULL;
    }

    if (PRODUCE_LOG) {
      delete logger;
      logger = nullptr;
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
        for (WLTy::iterator j = workList.begin ()/*, ej = workList.end ()*/; 
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
      std::vector<Event>& addList, Table& table, const FP& endtime, SimLogger* logger, bool enablePrints) { 


    for (ILTy::iterator i = indepList.begin (), ei = indepList.end ();
        i != ei; ++i) {

      addList.clear ();
      const bool notStale = i->notStale ();


      i->simulate ();
      table.addNextEvents (*i, addList, endtime);

      for (std::vector<Event>::const_iterator a = addList.begin (), ea = addList.end ();
          a != ea; ++a) {

        workList.insert (*a);
      }

      if (PRODUCE_LOG && notStale) {
        logger->log (*i);
      }
    }
  }



};



#endif //  BILLIARDS_PART_ORD_SERIAL_H

