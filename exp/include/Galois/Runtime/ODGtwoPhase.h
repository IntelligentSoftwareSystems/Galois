/** TODO -*- C++ -*-
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
 * TODO 
 *
 * @author <ahassaan@ices.utexas.edu>
 */


#ifndef GALOIS_RUNTIME_ODGTWOPHASE_H
#define GALOIS_RUNTIME_ODGTWOPHASE_H

#include <iostream>
#include <iterator>
#include <new>

#include "Galois/Accumulator.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"


#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/DoAll.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/Neighborhood.h"

namespace Galois::Runtime {


template <typename T, typename OperFunc, typename NhoodFunc, typename Cmp, typename Ctxt >
class TwoPhaseODGexecutor {

  typedef Galois::GAccumulator<size_t> Accumulator;

  typedef Galois::Runtime::MM::FSBGaloisAllocator<Ctxt> CtxtAlloc_ty;
  typedef Galois::Runtime::PerThreadVector<Ctxt*> CtxtWL_ty;

  typedef Galois::Runtime::PerThreadVector<Ctxt*> SrcWL_ty;

  typedef Galois::Runtime::PerThreadVector<T> AddList;

  struct CreateCtxt {
    CtxtAlloc_ty& ctxtAlloc;
    CtxtWL_ty& contexts;

    CreateCtxt (
        CtxtAlloc_ty& ctxtAlloc,
        CtxtWL_ty& contexts)
      :
        ctxtAlloc (ctxtAlloc),
        contexts (contexts)
    {}

    Ctxt* operator () (const T& active) {
      Ctxt* ctxt = ctxtAlloc.allocate (1);
      ctxtAlloc.construct (ctxt, Ctxt (active));
      // Ctxt* ctxt = new Ctxt (&active);

      contexts.get ().push_back (ctxt);

      return ctxt;
    };
  };

  struct ExpandNhood {
    NhoodFunc& nhoodVisitor;
    Accumulator& findIter;

    ExpandNhood (
        NhoodFunc& nhoodVisitor,
        Accumulator& findIter)
      :
        nhoodVisitor (nhoodVisitor),
        findIter (findIter)
    {}

    void operator () (Ctxt* ctxt) {
      findIter += 1;

      Galois::Runtime::setThreadContext (ctxt);
      ctxt->removeFromNhood ();
      nhoodVisitor (ctxt->active);
      Galois::Runtime::setThreadContext (NULL);
    }
  };


  struct CreateCtxtExpandNhood: public CreateCtxt, public ExpandNhood {

    CreateCtxtExpandNhood (
        NhoodFunc& nhoodVisitor,
        CtxtAlloc_ty& ctxtAlloc,
        CtxtWL_ty& contexts,
        Accumulator& expIter)
      :
        CreateCtxt (ctxtAlloc, contexts),
        ExpandNhood (nhoodVisitor, expIter)
    {}


    // TODO: change to ref
    void operator () (const T& active) {

      Ctxt* ctxt = CreateCtxt::operator () (active);

      ExpandNhood::operator () (ctxt);

    }

  };


  struct FindSourcesByNeighborhood {
    Cmp& cmp;
    SrcWL_ty& sources;
    Accumulator& findIter;

    FindSourcesByNeighborhood (
        Cmp& cmp, 
        SrcWL_ty& sources,
        Accumulator& findIter)
      : 
        cmp (cmp), 
        sources (sources),
        findIter (findIter)
    {}


    template <typename NItem>
    void operator () (const NItem* ni) {
      findIter += 1;

      Ctxt* ctxt = ni->getHighestPriority (cmp);


      
      if ((ctxt != NULL) && ctxt->isLeader (ni)) {
        if (ctxt->isSrc (cmp)) {
          sources.get ().push_back (ctxt);
          // ctxt->printNhood ();
        }
      }
    }

  };

// 
  // struct FindSourcesByActive {
// 
    // Cmp& cmp;
    // SrcWL_ty& sources;
    // Accumulator& findIter;
// 
    // FindSourcesByActive (
        // Cmp& cmp, 
        // SrcWL_ty& sources,
        // Accumulator& findIter)
      // : 
        // cmp (cmp), 
        // sources (sources),
        // findIter (findIter)
    // {}
// 
    // void operator () (Ctxt* ctxt) {
      // assert (ctxt != NULL);
// 
      // ++(findIter.get ());
// 
      // if (ctxt.isSrc (cmp)) {
        // sources.get ().push_back (ctxt);
      // }
    // }
  // };


  struct ApplyOperatorShareList: public CreateCtxtExpandNhood {

    OperFunc& op;
    AddList& addList;
    Accumulator& opIter;

    ApplyOperatorShareList (
        NhoodFunc& nhoodVisitor,
        CtxtAlloc_ty& ctxtAlloc,
        CtxtWL_ty& contexts,
        Accumulator& expIter,
        OperFunc& op,
        AddList& addList,
        Accumulator& opIter)
      :
        CreateCtxtExpandNhood (nhoodVisitor, ctxtAlloc, contexts, expIter),
        op (op),
        addList (addList),
        opIter (opIter)
    {}


    void operator () (Ctxt* src) {
      opIter += 1;

      addList.get ().clear ();
      op (src->active, addList.get ());
      src->removeFromNhood ();

      CreateCtxtExpandNhood::ctxtAlloc.destroy (src);
      CreateCtxtExpandNhood::ctxtAlloc.deallocate (src, 1);
      // delete src; src = NULL;

      for (typename AddList::local_iterator i = addList.get ().begin ()
          , endi = addList.get ().end (); i != endi; ++i) {

        CreateCtxtExpandNhood::operator () (*i);
      }
    }
  };


  struct ApplyOperatorPriorityLock: public CreateCtxt  {

    Cmp& cmp;
    OperFunc& op;
    AddList& addList;
    Accumulator& opIter;

    ApplyOperatorPriorityLock (
        CtxtAlloc_ty& ctxtAlloc,
        SrcWL_ty& nextWL,
        Cmp& cmp,
        OperFunc& op,
        AddList& addList,
        Accumulator& opIter)
      :
        CreateCtxt (ctxtAlloc, nextWL),
        cmp (cmp),
        op (op),
        addList (addList),
        opIter (opIter)
    {}

    void operator () (Ctxt* ctxt) {

      if (ctxt->isSrc (cmp)) {
        opIter += 1;

        // std::cout << "Found source: " << ctxt << ", Active: " << ctxt->active 
          // << std::endl;

        addList.get ().clear ();
        op (ctxt->active, addList.get ());
        ctxt->removeFromNhood ();

        CreateCtxt::ctxtAlloc.destroy (ctxt);
        CreateCtxt::ctxtAlloc.deallocate (ctxt, 1);

        for (typename AddList::local_iterator i = addList.get ().begin ()
            , endi = addList.get ().end (); i != endi; ++i) {

          CreateCtxt::operator () (*i);
        }


      } else {
        ctxt->removeFromNhood ();
        CreateCtxt::contexts.get ().push_back (ctxt);
      }
    }

  };



  Cmp cmp;
  OperFunc operFunc;
  NhoodFunc nhoodVisitor;

public:

  TwoPhaseODGexecutor (
      Cmp& _cmp, 
      OperFunc& _operFunc, 
      NhoodFunc& _nhoodVisitor)
    :
      cmp (_cmp),
      operFunc (_operFunc),
      nhoodVisitor (_nhoodVisitor) 
  {}


  template <typename AI, typename NI>
  void execute (AI abeg, AI aend, NI nbeg, NI nend) {

    CtxtWL_ty contexts;
    CtxtAlloc_ty ctxtAlloc;
    SrcWL_ty sources;
    AddList addList;

    Accumulator expIter;
    Accumulator findIter;
    Accumulator opIter;

    Galois::TimeAccumulator findTime;
    Galois::TimeAccumulator opTime;

    // Galois::Runtime::do_all_coupled (
    Galois::do_all (
        abeg, aend, 
        CreateCtxtExpandNhood (nhoodVisitor, ctxtAlloc, contexts, expIter),
        "initial_expand_nhood");

    std::cout << "Iterations spent in initial expansion of nhood: " << expIter.reduce () << std::endl;

    size_t round = 0;
    while (true) {

      ++round;
      addList.clear_all ();
      sources.clear_all ();

      findTime.start ();
      // Galois::Runtime::do_all_coupled ( 
      Galois::do_all ( 
          nbeg, nend,
          FindSourcesByNeighborhood (cmp, sources, findIter),
          "find_src_by_nhood");
      findTime.stop ();




      // std::cout << "Number of sources found: " << sources.size_all () << std::endl;

      if (sources.empty_all ()) {
        break;
      }

      opTime.start ();
      // Galois::Runtime::do_all_coupled (
      Galois::do_all (
          sources.begin_all (), sources.end_all (),
          ApplyOperatorShareList (
            nhoodVisitor, 
            ctxtAlloc, 
            contexts, 
            expIter, 
            operFunc, 
            addList, 
            opIter),
          "exec_src");
      opTime.stop ();
          

    }

    std::cout << "Number of rounds: " << round << std::endl;
    std::cout << "Iterations spent in finding sources: " << findIter.reduce () << std::endl;
    std::cout << "Time spent in finding sources: " << findTime.get () << std::endl;

    std::cout << "Iterations spent in processing sources: " << opIter.reduce () << std::endl;
    std::cout << "Time spent in processing sources: " << opTime.get () << std::endl;


  }

  template <typename AI>
  void execute (AI abeg, AI aend) {

    CtxtAlloc_ty ctxtAlloc;
    CtxtWL_ty* currWL = new CtxtWL_ty ();
    CtxtWL_ty* nextWL = new CtxtWL_ty ();
    AddList addList;

    Accumulator findIter;
    Accumulator opIter;

    Galois::TimeAccumulator findTime;
    Galois::TimeAccumulator opTime;


    bool first = true;
    size_t round = 0;

    //size_t prev = 0;
    while (true) {

      ++round;
      if (first) {
        first = false;

        findTime.start ();
        Galois::do_all (abeg, aend,
            CreateCtxtExpandNhood (nhoodVisitor, ctxtAlloc, *currWL, findIter),
            "initial_expand_nhood");
        findTime.stop ();

      } else {

        findTime.start ();
        Galois::do_all (currWL->begin_all (), currWL->end_all (),
            ExpandNhood (nhoodVisitor, findIter),
            "find_sources");
        findTime.stop ();
      }



      opTime.start ();
      Galois::do_all (currWL->begin_all (), currWL->end_all (),
          ApplyOperatorPriorityLock (ctxtAlloc, *nextWL, cmp, operFunc, addList, opIter),
          "apply_operator");
      opTime.stop ();

      // std::cout << "Number of sources found: " << (opIter.reduce () - prev) << std::endl;
      //prev =
      opIter.reduce ();

      std::swap (currWL, nextWL);
      nextWL->clear_all ();

      if (currWL->empty_all ()) {
        break;
      }
    }

    std::cout << "Number of rounds: " << round << std::endl;
    std::cout << "Iterations spent in finding sources: " << findIter.reduce () << std::endl;
    std::cout << "Time spent in finding sources: " << findTime.get () << std::endl;

    std::cout << "Iterations spent in processing sources: " << opIter.reduce () << std::endl;
    std::cout << "Time spent in processing sources: " << opTime.get () << std::endl;

    delete currWL; currWL = NULL;
    delete nextWL; nextWL = NULL;

  }

};


template <typename AI, typename NI, typename OperFunc, typename NhoodFunc, typename Cmp>
void for_each_ordered (AI abeg, AI aend, NI nbeg, NI nend, OperFunc operFunc, NhoodFunc nhoodVisitor, Cmp cmp) {

  typedef typename std::iterator_traits<AI>::value_type T;

  typedef typename std::iterator_traits<NI>::value_type NItemPtr;

  typedef typename std::iterator_traits<NItemPtr>::value_type NItem;

  typedef NhoodListContext<T, NItem> Ctxt; 

  typedef TwoPhaseODGexecutor<T, OperFunc, NhoodFunc, Cmp, Ctxt> Exec;

  Exec exec (cmp, operFunc, nhoodVisitor);

  exec.execute (abeg, aend, nbeg, nend); 
}

template <typename AI, typename OperFunc, typename NhoodFunc, typename Cmp>
void for_each_ordered (AI abeg, AI aend, OperFunc operFunc, NhoodFunc nhoodVisitor, Cmp cmp) {

  typedef typename std::iterator_traits<AI>::value_type T;

  typedef NhoodItemPriorityLock<T> NItem;

  typedef NhoodListContext<T, NItem> Ctxt;

  typedef TwoPhaseODGexecutor<T, OperFunc, NhoodFunc, Cmp, Ctxt> Exec;

  Exec exec (cmp, operFunc, nhoodVisitor);

  exec.execute (abeg, aend);
}


} // end namespace Galois::Runtime

#endif // GALOIS_RUNTIME_ODG_TWO_PHASE_H


