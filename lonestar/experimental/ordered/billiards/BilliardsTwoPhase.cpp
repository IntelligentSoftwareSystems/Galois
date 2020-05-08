/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/runtime/KDGtwoPhase.h"

#include "Billiards.h"
#include "dependTest.h"
#include "BilliardsParallel.h"

class BilliardsTwoPhase : public Billiards<BilliardsTwoPhase, Table<Ball>> {

public:
  using Tbl_t = Table<Ball>;

  virtual const std::string version() const { return "using IKDG"; }

  size_t runSim(Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime,
                bool enablePrints = false, bool logEvents = false) {

    AddListTy addList;
    Accumulator iter;

    galois::runtime::for_each_ordered_ikdg(
        galois::runtime::makeStandardRange(initEvents.begin(),
                                           initEvents.end()),
        Event::Comparator(), VisitNhoodSafetyTest(), ExecSources(),
        AddEvents<Tbl_t>(table, endtime, addList, iter, enablePrints),
        std::make_tuple(galois::loopname("billiards-ikdg")));

    return iter.reduce();
  }
};

int main(int argc, char* argv[]) {
  BilliardsTwoPhase s;
  s.run(argc, argv);
  return 0;
}
