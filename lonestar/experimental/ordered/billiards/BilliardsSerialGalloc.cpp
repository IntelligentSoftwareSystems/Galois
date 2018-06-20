/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "Billiards.h"

#include "galois/PriorityQueue.h"

template <typename Tbl_t = Table<Ball>>
class BilliardsSerialGalloc
    : public Billiards<BilliardsSerialGalloc<Tbl_t>, Tbl_t> {

  // using PriorityQueue =  std::priority_queue<Event, std::vector<Event>,
  // Event::ReverseComparator>;
  using PriorityQueue = galois::ThreadSafeOrderedSet<Event, Event::Comparator>;

public:
  virtual const std::string version() const {
    return "Serial Ordered with Priority Queue";
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void
  processEvent(Event& e, Tbl_t& table, std::vector<Event>& addList,
               const FP& endtime) {

    addList.clear();
    e.simulate();
    table.addNextEvents(e, addList, endtime);
  }

  size_t runSim(Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime,
                bool enablePrints = false, bool logEvents = false) {

    std::printf("BilliardsSerialGalloc: number of initial events: %zd\n",
                initEvents.size());

    PriorityQueue pq;

    for (std::vector<Event>::iterator i  = initEvents.begin(),
                                      ei = initEvents.end();
         i != ei; ++i) {

      pq.push(*i);
    }

    size_t iter = 0;
    std::vector<Event> addList;

    while (!pq.empty()) {

      Event e = pq.pop();

      if (enablePrints) {
        std::cout << "Processing event=" << e.str() << std::endl;
      }

      processEvent(e, table, addList, endtime);

      if (logEvents) {
        table.logCollisionEvent(e);
      }

      for (std::vector<Event>::iterator i = addList.begin(), ei = addList.end();
           i != ei; ++i) {

        pq.push(*i);

        if (enablePrints) {
          std::cout << "Adding event=" << i->str() << std::endl;
        }
      }

      if (enablePrints) {
        table.printState(std::cout);
      }

      ++iter;
    }

    return iter;
  }
};

int main(int argc, char* argv[]) {
  BilliardsSerialGalloc<> s;
  s.run(argc, argv);
  return 0;
}
