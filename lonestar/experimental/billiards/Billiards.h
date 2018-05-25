/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef _BILLIARDS_H_
#define _BILLIARDS_H_


#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <queue>

#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/DoAllWrap.h"
#include "galois/Reduction.h"
#include "galois/runtime/Profile.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "TableSectored.h"

namespace cll = llvm::cl;

static const char* const name = "Billiards Simulation";
static const char* const desc = "Simulates elastic collisions between billiard balls on a table";
static const char* const url = "billiards";

static cll::opt<unsigned>   sectorSize("secsize", cll::desc("size of a (square) sector"), cll::init(100));
static cll::opt<unsigned>   xSectors("xsec", cll::desc("number of sectors along length"), cll::init(1));
static cll::opt<unsigned>   ySectors("ysec", cll::desc("number of sectors along height"), cll::init(1));
static cll::opt<unsigned> numballs("balls", cll::desc("number of balls on the table"), cll::init(100));
static cll::opt<unsigned>   endtime("end", cll::desc("simulation end time"), cll::init(100));

// static cll::opt<bool> runFlat("flat", cll::desc("Run simulation without introducing sectors/partitions"), cll::init(false));

static cll::opt<bool> veriFlat ("vflat", cll::desc ("Verify against serial flat simulation"), cll::init (false));


typedef galois::GAccumulator<size_t> Accumulator;


static const unsigned DEFAULT_CHUNK_SIZE = 4;

template <typename Derived, typename Tbl_t>
class Billiards {

  using Ball_t = typename Tbl_t::Ball_t;

public:

  virtual const std::string version () const = 0;
  //! @return number of events processed
  size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {
    GALOIS_DIE ("Derived classes must provide the implementation");
  }

private:
  template <typename ST>
  void verify (const Tbl_t& initial, Tbl_t& final, size_t numEvents, const FP& endtime);

// #define CUSTOM_TESTS
#ifdef CUSTOM_TESTS

  void testA (std::vector<Ball_t>& balls) {

    sectorSize = 1;
    xSectors = 2;
    ySectors = 1;
    endtime = 10.0;

    balls.clear ();

    balls.emplace_back (0, Vec2 (0.5, 0.5), Vec2 (1.0, 1.0), 1.0, 0.25);
  }

  void testB (std::vector<Ball_t>& balls) {

    sectorSize = 1;
    xSectors = 2;
    ySectors = 1;
    endtime = 10.0;

    balls.clear ();

    balls.emplace_back (0, Vec2 (0.5, 0.5), Vec2 (1.0, 1.0), 1.0, 0.25);
    balls.emplace_back (1, Vec2 (1.5, 0.5), Vec2 (-1.0, 0.0), 1.0, 0.25);
  }

  void testC (std::vector<Ball_t>& balls) {

    sectorSize = 1;
    xSectors = 2;
    ySectors = 2;
    endtime = 10.0;

    balls.clear ();

    balls.emplace_back (0, Vec2 (0.5, 0.5), Vec2 (1.0, 1.0), 1.0, 0.25);
    // balls.emplace_back (1, Vec2 (1.5, 0.5), Vec2 (-1.0, 0.0), 1.0, 0.25);
  }

  void testD (std::vector<Ball_t>& balls) {

    sectorSize = 1;
    xSectors = 2;
    ySectors = 2;
    endtime = 10.0;

    balls.clear ();

    balls.emplace_back (0, Vec2 (0.5, 0.5), Vec2 (1.0, -1.0), 1.0, 0.25);
    // balls.emplace_back (1, Vec2 (1.5, 0.5), Vec2 (-1.0, 0.0), 1.0, 0.25);
  }

  void testE (std::vector<Ball_t>& balls) {

    sectorSize = 1;
    xSectors = 2;
    ySectors = 2;
    endtime = 10.0;

    balls.clear ();

    balls.emplace_back (0, Vec2 (0.5, 0.5), Vec2 (1.0, 1.0), 1.0, 0.25);
    balls.emplace_back (1, Vec2 (1.5, 1.5), Vec2 (-1.0, -1.0), 1.0, 0.25);
  }

  void testF (std::vector<Ball_t>& balls) {

    sectorSize = 10;
    xSectors = 3;
    ySectors = 3;
    endtime = 100.0;

    balls.clear ();

    balls.emplace_back (0, Vec2 (0.5, 0.5), Vec2 (1.0, 1.0), 1.0, 0.25);
    balls.emplace_back (1, Vec2 (1.5, 1.5), Vec2 (-1.0, -1.0), 1.0, 0.25);
    balls.emplace_back (2, Vec2 (2.5, 2.5), Vec2 (-1.0, -1.0), 1.0, 0.25);
  }
#endif // CUSTOM_TESTS


  void runImpl (void) {

#ifdef CUSTOM_TESTS

    std::vector<Ball_t> balls;

    testF (balls);

    Tbl_t table (balls.begin (), balls.end (), sectorSize, xSectors, ySectors);


#else // CUSTOM_TESTS
    Tbl_t table (numballs, sectorSize, xSectors, ySectors);
#endif // CUSTOM_TESTS

    Tbl_t verCopy (table);

    bool enablePrints = false;
    bool logEvents = true;


    if (enablePrints) {
      table.printState (std::cout);
    }

    std::vector<Event> initEvents;
    table.genInitialEvents (initEvents, unsigned (endtime));

    std::cout << "Number of initial events = " << initEvents.size () << std::endl;

    galois::preAlloc ((galois::getActiveThreads () * unsigned(endtime) * numballs * 10)/galois::runtime::pagePoolSize());
    galois::reportPageAlloc("MeminfoPre");

    galois::StatTimer timer;

    timer.start ();
    size_t numEvents = static_cast<Derived*> (this)->runSim (table, initEvents, unsigned (endtime), enablePrints, logEvents);
    timer.stop ();
    galois::reportPageAlloc("MeminfoPost");

    std::cout << "Billiards " << version () << ", number of events processed=" << numEvents << std::endl;

    if (!skipVerify) {
      if (veriFlat) {
        verify <typename Tbl_t::SerialFlatTable> (static_cast<const Table<Ball_t>&> (verCopy), static_cast<Table<Ball_t>&> (table), numEvents, unsigned (endtime));

      } else {
        verify <typename Tbl_t::SerialTable> (verCopy, table, numEvents, unsigned (endtime));

      }
    }

  }

public:
  void run (int argc, char* argv[]) {
    
    galois::StatManager sm;
    LonestarStart (argc, argv, name, desc, url);
    runImpl ();
  }

};


template <typename Tbl_t=Table<Ball> >
class BilliardsSerialPQ: public Billiards<BilliardsSerialPQ<Tbl_t>, Tbl_t> {

  using PriorityQueue =  std::priority_queue<Event, std::vector<Event>, Event::ReverseComparator>; 

public:

  virtual const std::string version () const { return "Serial Ordered with Priority Queue"; }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void processEvent (Event& e, Tbl_t& table, std::vector<Event>& addList, const FP& endtime) {

      addList.clear ();
      e.simulate ();
      table.addNextEvents (e, addList, endtime);
  }

  size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {

    std::printf ("BilliardsSerialPQ: number of initial events: %zd\n", initEvents.size ());

    PriorityQueue pq;

    for (std::vector<Event>::iterator i = initEvents.begin (), ei = initEvents.end ();
        i != ei; ++i) {

        pq.push (*i);
    }

    size_t iter = 0;
    std::vector<Event> addList;

    while (!pq.empty ()) {

      Event e = pq.top ();
      pq.pop ();

      if (enablePrints) {
        std::cout << "Processing event=" << e.str () << std::endl;
      }

      processEvent (e, table, addList, endtime);

      if (logEvents) {
        table.logCollisionEvent (e);
      }

      for (std::vector<Event>::iterator i = addList.begin (), ei = addList.end ();
          i != ei; ++i) {

        pq.push (*i);

        if (enablePrints) {
          std::cout << "Adding event=" << i->str () << std::endl;
        }
      }

      if (enablePrints) {
        table.printState (std::cout);
      }

      ++iter;
    }

    return iter;

  }
};


template <typename Derived, typename Tbl_t>
template <typename T>
void Billiards<Derived, Tbl_t>::verify (const Tbl_t& initial, Tbl_t& final, size_t numEvents, const FP& endtime) {

  FP initEnergy = initial.sumEnergy ();
  FP finalEnergy = final.sumEnergy ();

  FPutils::checkError (initEnergy, finalEnergy);


  BilliardsSerialPQ<T> serial;
  T serialTable(initial);

  std::vector<Event> initEvents;
  serialTable.genInitialEvents (initEvents, endtime);

  galois::StatTimer timer ("Verfication time (Serial PQ simulation)= ");
  
  timer.start ();
  size_t serEvents = serial.runSim(serialTable, initEvents, endtime, false, true);
  timer.stop ();

  std::cout << "Serial ordered numEvents=" << serEvents << std::endl;

  if (serEvents != numEvents) {
    std::cerr << "WARNING: number of events differ from verification run, normal=" 
      << numEvents <<  " != verification=" << serEvents << std::endl;
  }

  FP serEnergy = serialTable.sumEnergy ();

  FPutils::checkError (serEnergy, initEnergy);
  FPutils::checkError (serEnergy, finalEnergy);


  //advance both tables to same endtime
  final.advance (endtime);
  serialTable.advance (endtime);

  // check the end state of both simulations
  // pass true to print differences 
  bool result = serialTable.template cmpState<true> (final);

  if (!result) {
    std::cerr << "ERROR, Comparison against serial ordered simulation failed due to above differences" << std::endl;

    // std::cout << "<<<<<<<<<<< Event logs from Original Run >>>>>>>>>>>>>" << std::endl;
    // final.printEventLogs ();
// 
    // std::cout << "<<<<<<<<<<< Event logs from Verification Run >>>>>>>>>>>>>" << std::endl;
    // serialTable.printEventLogs ();

    final.diffEventLogs (serialTable, "Serial Flat Run");

    abort ();

  } else {
    std::cout << "OK, Result verified against serial ordered simulation" << std::endl;
  }


}

#endif // _BILLIARDS_H_

