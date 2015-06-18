/** Billiards Simulation top level clas  -*- C++ -*-
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
 * Billiards Simulation top level clas.
 *
 * @author <ahassaan@ices.utexas.edu>
 */


#ifndef _BILLIARDS_H_
#define _BILLIARDS_H_


#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <queue>

#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Runtime/Sampling.h"
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
static cll::opt<double>   endtime("end", cll::desc("simulation end time"), cll::init(100.0));


class Billiards {

public:
  typedef Galois::GAccumulator<size_t> Accumulator;

  virtual const std::string version () const = 0;
  //! @return number of events processed
  virtual size_t runSim (TableSectored& table, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) = 0;

  virtual void run (int argc, char* argv[]) {
    
    Galois::StatManager sm;
    LonestarStart (argc, argv, name, desc, url);

    TableSectored table (numballs, sectorSize, xSectors, ySectors);
    TableSectored verCopy (table);

    bool enablePrints = false;


    if (enablePrints) {
      table.printState (std::cout);
    }

    std::vector<Event> initEvents;
    table.genInitialEvents (initEvents, endtime);

    std::cout << "Number of initial events = " << initEvents.size () << std::endl;

    Galois::StatTimer timer ("Simulation time: ");

    timer.start ();
    Galois::Runtime::beginSampling ();
    size_t numEvents = runSim (table, initEvents, endtime, enablePrints);
    Galois::Runtime::endSampling ();
    timer.stop ();

    std::cout << "Billiards " << version () << ", number of events processed=" << numEvents << std::endl;

    if (!skipVerify) {
      verify (verCopy, table, numEvents, endtime);
    }

  }

private:

  void verify (const TableSectored& initial, TableSectored& final, size_t numEvents, const double endtime);
};


class BilliardsSerialPQ: public Billiards {
  typedef std::priority_queue<Event, std::vector<Event>, Event::ReverseComparator> PriorityQueue;
public:

  virtual const std::string version () const { return "Serial Ordered with Priority Queue"; }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void processEvent (Event& e, Table& table, std::vector<Event>& addList, const double endtime) {
      addList.clear ();
      e.simulate ();
      table.addNextEvents (e, addList, endtime);
  }

  virtual size_t runSim (TableSectored& tableSectored, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) {

    Table& table = tableSectored; // remove sectoring functionality

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


void Billiards::verify (const TableSectored& initial, TableSectored& final, size_t numEvents, const double endtime) {

  double initEnergy = initial.sumEnergy ();
  double finalEnergy = final.sumEnergy ();

  FPutils::checkError (initEnergy, finalEnergy);


  BilliardsSerialPQ serial;
  TableSectored serialTable(initial);

  std::vector<Event> initEvents;
  serialTable.genInitialEvents (initEvents, endtime);

  Galois::StatTimer timer ("Verfication time (Serial PQ simulation)= ");
  
  timer.start ();
  size_t serEvents = serial.runSim(serialTable, initEvents, endtime, false);
  timer.stop ();

  std::cout << "Serial ordered numEvents=" << serEvents << std::endl;

  if (serEvents != numEvents) {
    std::cerr << "WARNING: number of events differ from verification run, normal=" 
      << numEvents <<  " != verification=" << serEvents << std::endl;
  }

  double serEnergy = serialTable.sumEnergy ();

  FPutils::checkError (serEnergy, initEnergy);
  FPutils::checkError (serEnergy, finalEnergy);


  // check the end state of both simulations
  // pass true to print differences 
  bool result = serialTable.cmpState<true> (final);

  if (!result) {
    std::cerr << "ERROR, Comparison against serial ordered simulation failed due to above differences" << std::endl;
    abort ();

  } else {
    std::cout << "OK, Result verified against serial ordered simulation" << std::endl;
  }


}

#endif // _BILLIARDS_H_

