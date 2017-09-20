/** BilliardsSectoredIKDG -*- C++ -*-
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
 * @author <ahassaan@ices.utexas.edu>
 */

#include "Billiards.h"
#include "dependTest.h"
#include "BilliardsParallel.h"
#include "BilliardsParallelSectored.h"

#include "Galois/Runtime/KDGtwoPhase.h"

class BilliardsSectoredIKDG: public Billiards<BilliardsSectoredIKDG>  {

public:

  virtual const std::string version () const { return "IKDG with custom safety test"; }

  template <typename Tbl_t>
  size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {

    AddListTy addList;
    Accumulator iter;

    galois::runtime::for_each_ordered_ikdg_custom_safety (
        galois::runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        Event::Comparator (),
        SectorLocalTest<Tbl_t> {table},
        ExecSources (),
        AddEvents<Tbl_t> (table, endtime, addList, iter),
        "sectored-ikdg-custom-safety");

    return iter.reduce ();

  }
};

int main (int argc, char* argv[]) {
  BilliardsSectoredIKDG s;
  s.run (argc, argv);
  return 0;
}
