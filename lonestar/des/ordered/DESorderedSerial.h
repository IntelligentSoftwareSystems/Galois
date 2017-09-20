/** DES serial ordered version -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef DES_ORDERED_SERIAL_H
#define DES_ORDERED_SERIAL_H

#include <deque>
#include <functional>
#include <queue>
#include <set>

#include <cassert>

#include "galois/Substrate/CompilerSpecific.h"

#include "abstractMain.h"
#include "SimInit.h"
#include "TypeHelper.h"

namespace des_ord {

class DESorderedSerial: 
  public des::AbstractMain<TypeHelper<>::SimInit_ty>, public TypeHelper<> {

  typedef std::priority_queue<Event_ty, std::vector<Event_ty>, des::EventRecvTimeLocalTieBrkCmp<Event_ty>::RevCmp> MinHeap;
  typedef std::set<Event_ty, des::EventRecvTimeLocalTieBrkCmp<Event_ty> > OrdSet;

  std::vector<GNode> nodes;

protected:


  virtual std::string getVersion () const { return "Ordered serial"; }

  virtual void initRemaining (const SimInit_ty& simInit, Graph& graph) {
    nodes.clear ();
    nodes.resize (graph.size ());

    for (Graph::iterator n = graph.begin ()
        , endn = graph.end (); n != endn; ++n) {

      BaseSimObj_ty* so = graph.getData (*n, galois::MethodFlag::UNPROTECTED);
      nodes[so->getID ()] = *n;
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static Event_ty removeMin (MinHeap& pq) {
    Event_ty ret = pq.top ();
    pq.pop ();
    return ret;
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static Event_ty removeMin (OrdSet& pq) {
    Event_ty ret = *pq.begin ();
    pq.erase (pq.begin ());
    return ret;
  }


  template <typename PQ, typename _ignore>
  struct AddNewWrapper {
    PQ& pq;

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event_ty& e) {
      pq.push (e);
    }
  };

  template <typename _ignore>
  struct AddNewWrapper<OrdSet, _ignore> {
    OrdSet& pq;

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event_ty& e) {
      pq.insert (e);
    }
  };

  template <typename PQ>
  AddNewWrapper<PQ, char> makeAddNewFunc (PQ& pq) {
    return AddNewWrapper<PQ, char> {pq};
  }

  virtual void runLoop (const SimInit_ty& simInit, Graph& graph) {


    // MinHeap pq;
    OrdSet pq;

    auto addNewFunc = makeAddNewFunc (pq);

    for (std::vector<Event_ty>::const_iterator i = simInit.getInitEvents ().begin ()
        , endi = simInit.getInitEvents ().end (); i != endi; ++i) {
      addNewFunc (*i);
    }

    size_t numEvents = 0;;
    while (!pq.empty ()) {
      ++numEvents;

      Event_ty event = removeMin (pq);

      SimObj_ty* recvObj = static_cast<SimObj_ty*> (event.getRecvObj ());
      GNode recvNode = nodes[recvObj->getID ()];

      recvObj->execEvent (event, graph, recvNode, addNewFunc);

    }

    std::cout << "Number of events processed = " << numEvents << std::endl;

  }

};

} // namespace des_ord
#endif // DES_ORDERED_SERIAL_H
