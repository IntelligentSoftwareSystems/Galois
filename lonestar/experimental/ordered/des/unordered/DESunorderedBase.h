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

#include "SimInit.h"
#include "abstractMain.h"
#include "SimObject.h"

namespace des_unord {

namespace cll = llvm::cl;
static cll::opt<unsigned>
    eventsPerIter("epi",
                  cll::desc("number of events processed per iteration (max.)"),
                  cll::init(0));

struct TypeHelper {
  typedef des::Event<des::LogicUpdate> Event_ty;
  typedef Event_ty::BaseSimObj_ty BaseSimObj_ty;
  typedef des_unord::SimObject<Event_ty> SimObj_ty;

  typedef des::SimGate<SimObj_ty> SimGate_ty;
  typedef des::Input<SimObj_ty> Input_ty;
  typedef des::Output<SimObj_ty> Output_ty;

  typedef des::SimInit<NEEDS_NULL_EVENTS, SimGate_ty, Input_ty, Output_ty>
      SimInit_ty;
};

class DESunorderedBase
    : public des::AbstractMain<des_unord::TypeHelper::SimInit_ty>,
      public des_unord::TypeHelper {

protected:
  typedef des::AbstractMain<des_unord::TypeHelper::SimInit_ty> AbstractBase;

  virtual void initRemaining(const SimInit_ty& simInit, Graph& graph) {

    SimObj_ty::NEVENTS_PER_ITER = eventsPerIter;
    if (SimObj_ty::NEVENTS_PER_ITER == 0) {
      SimObj_ty::NEVENTS_PER_ITER = AbstractBase::DEFAULT_EPI;
    }

    // post the initial events on their stations
    for (std::vector<Event_ty>::const_iterator
             i    = simInit.getInitEvents().begin(),
             endi = simInit.getInitEvents().end();
         i != endi; ++i) {

      SimObj_ty* so = static_cast<SimObj_ty*>(i->getRecvObj());
      so->recv(*i);
    }
  }

  template <typename WL, typename B>
  void initWorkList(Graph& graph, WL& workList, std::vector<B>& onWLflags) {
    onWLflags.clear();
    workList.clear();

    onWLflags.resize(graph.size(), B(false));

    // set onWLflags for input objects
    for (Graph::iterator n = graph.begin(), endn = graph.end(); n != endn;
         ++n) {

      SimObj_ty* so = static_cast<SimObj_ty*>(
          graph.getData(*n, galois::MethodFlag::UNPROTECTED));

      if (so->isActive()) {
        workList.push_back(*n);
        onWLflags[so->getID()] = true;
      }
    }

    std::cout << "Initial workList size = " << workList.size() << std::endl;
  }
};

} // end namespace des_unord
