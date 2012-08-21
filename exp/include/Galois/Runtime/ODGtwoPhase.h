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


#ifndef GALOIS_RUNTIME_ODG_TWO_PHASE_H
#define GALOIS_RUNTIME_ODG_TWO_PHASE_H

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

namespace GaloisRuntime {


template <typename T, typename OperFunc, typename NhoodFunc, typename Cmp, typename Ctxt >
class TwoPhaseShareListExecutor {

  typedef Galois::GAccumulator<size_t> Accumulator;

  typedef GaloisRuntime::MM::FSBGaloisAllocator<Ctxt> CtxtAlloc_ty;
  typedef GaloisRuntime::PerThreadVector<Ctxt*> CtxtWL_ty;

  typedef GaloisRuntime::PerThreadVector<Ctxt*> SrcWL_ty;

  typedef GaloisRuntime::PerThreadVector<T> AddList;

  struct ExpandNhood {

    Cmp& cmp;
    NhoodFunc& nhoodVisitor;
    CtxtAlloc_ty& ctxtAlloc;
    CtxtWL_ty& contexts;
    Accumulator& expIter;

    ExpandNhood (
        Cmp& cmp,
        NhoodFunc& nhoodVisitor,
        CtxtAlloc_ty& ctxtAlloc,
        CtxtWL_ty& contexts,
        Accumulator& expIter)
      :
        cmp (cmp),
        nhoodVisitor (nhoodVisitor),
        ctxtAlloc (ctxtAlloc),
        contexts (contexts),
        expIter (expIter)
    {}


    // TODO: change to ref
    void operator () (const T& active) {
      expIter += 1;

      Ctxt* ctxt = ctxtAlloc.allocate (1);
      new (ctxt) Ctxt (active);
      // Ctxt* ctxt = new Ctxt (&active);

      contexts.get ().push_back (ctxt);

      GaloisRuntime::setThreadContext (ctxt);

      nhoodVisitor (active);

      GaloisRuntime::setThreadContext (NULL);
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


  struct ApplyOperator: public ExpandNhood {

    OperFunc& op;
    AddList& addList;
    Accumulator& opIter;

    ApplyOperator (
        Cmp& cmp,
        OperFunc& op,
        AddList& addList,
        NhoodFunc& nhoodVisitor,
        CtxtAlloc_ty& ctxtAlloc,
        CtxtWL_ty& contexts,
        Accumulator& expIter,
        Accumulator& opIter)
      :
        ExpandNhood (cmp, nhoodVisitor, ctxtAlloc, contexts, expIter),
        op (op),
        addList (addList),
        opIter (opIter)
    {}


    void operator () (Ctxt*& src) {
      opIter += 1;

      addList.get ().clear ();
      op (src->active, addList.get ());
      src->removeFromNhood ();

      ExpandNhood::ctxtAlloc.deallocate (src, 1);
      // delete src; src = NULL;

      for (typename AddList::local_iterator i = addList.get ().begin ()
          , endi = addList.get ().end (); i != endi; ++i) {

        ExpandNhood::operator () (*i);
      }
    }
  };




  Cmp cmp;
  OperFunc operFunc;
  NhoodFunc nhoodVisitor;

public:

  TwoPhaseShareListExecutor (
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

    // GaloisRuntime::do_all_coupled (
    Galois::do_all (
        abeg, aend, 
        ExpandNhood (cmp, nhoodVisitor, ctxtAlloc, contexts, expIter),
        "initial_expand_nhood");

    std::cout << "Iterations spent in initial expansion of nhood: " << expIter.reduce () << std::endl;

    size_t round = 0;
    while (true) {

      ++round;
      addList.clear_all ();
      sources.clear_all ();

      findTime.start ();
      // GaloisRuntime::do_all_coupled ( 
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
      // GaloisRuntime::do_all_coupled (
      Galois::do_all (
          sources.begin_all (), sources.end_all (),
          ApplyOperator (
            cmp, 
            operFunc, 
            addList, 
            nhoodVisitor, 
            ctxtAlloc, 
            contexts, 
            expIter, 
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

};


template <typename AI, typename NI, typename OperFunc, typename NhoodFunc, typename Cmp>
void for_each_ordered (AI abeg, AI aend, NI nbeg, NI nend, OperFunc operFunc, NhoodFunc nhoodVisitor, Cmp cmp) {

  typedef typename std::iterator_traits<AI>::value_type T;

  typedef typename std::iterator_traits<NI>::value_type NItemPtr;

  typedef typename std::iterator_traits<NItemPtr>::value_type NItem;

  typedef NhoodListContext<T, NItem> Ctxt; 

  typedef TwoPhaseShareListExecutor<T, OperFunc, NhoodFunc, Cmp, Ctxt> Exec;

  Exec exec (cmp, operFunc, nhoodVisitor);

  exec.execute (abeg, aend, nbeg, nend); 
}


} // end namespace GaloisRuntime

#endif // GALOIS_RUNTIME_ODG_TWO_PHASE_H


