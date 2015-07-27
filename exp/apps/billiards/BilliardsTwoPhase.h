/** Billiards Simulation using two phase executor-*- C++ -*-
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

#ifndef BILLIARDS_TWO_PHASE_H
#define BILLIARDS_TWO_PHASE_H

#include "Galois/PerThreadContainer.h"

#include "Galois/Graphs/Graph.h"

#include "Galois/Runtime/KDGtwoPhase.h"

#include "Billiards.h"
#include "dependTest.h"

class BilliardsTwoPhase: public Billiards {

  using AddListTy = Galois::PerThreadVector<Event>;

  struct VisitNhood {
    static const unsigned CHUNK_SIZE = 1;

    template <typename C, typename I>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& event, const C& c, const I beg, const I end) {
      bool indep = true;

      for (I i = beg; i != end; ++i) {
        if (event > *i) {
          if (OrderDepTest::dependsOn (event, *i)) {
            indep = false;
            break;
          }
        }
      }

      if (!indep) {
        Galois::Runtime::signalConflict ();
      }
    }
  };


  struct SerialPart {

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& e) const {
      const_cast<Event&> (e).simulate ();
    }
  };

  struct OpFunc {

    static const unsigned CHUNK_SIZE = 1;

    Table& table;
    const double endtime;
    AddListTy& addList;
    Accumulator& iter;

    OpFunc (
        Table& table,
        double endtime,
        AddListTy& addList,
        Accumulator& iter)
      :
        table (table),
        endtime (endtime),
        addList (addList),
        iter (iter)
    {}


    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& e, C& ctx) {

      addList.get ().clear ();

      // TODO: use locks to update balls' state atomically 
      // and read atomically
      // const_cast<Event&>(e).simulate ();
      table.addNextEvents (e, addList.get (), endtime);

      for (auto i = addList.get ().begin ()
          , endi = addList.get ().end (); i != endi; ++i) {

        ctx.push (*i);
      }

      iter += 1;
    }
  };

  // void createLocks (const Table& table, Graph& graph, VecNodes& nodes) {
    // nodes.reserve (table.getNumBalls ());
// 
    // for (unsigned i = 0; i < table.getNumBalls (); ++i) {
      // nodes.push_back (graph.createNode (nullptr));
    // }
// 
  // };

public:

  virtual const std::string version () const { return "using IKDG"; }

  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) {

    AddListTy addList;
    Accumulator iter;

    // createLocks (table, graph, nodes);

    Galois::Runtime::for_each_ordered_2p_win (
        Galois::Runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        Event::Comparator (),
        VisitNhood (),
        OpFunc (table, endtime, addList, iter),
        SerialPart ());

    return iter.reduce ();

  }
};


#endif // BILLIARDS_TWO_PHASE_H
