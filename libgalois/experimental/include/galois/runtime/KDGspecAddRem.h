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

#ifndef GALOIS_RUNTIME_KDG_SPEC_LOCAL_MIN_H
#define GALOIS_RUNTIME_KDG_SPEC_LOCAL_MIN_H

#include "OrderedSpeculation.h"

namespace galois {
namespace runtime {

template <typename Ctxt, typename CtxtCmp>
struct SpecAddRemNhoodItem
    : public OrdLocBase<SpecAddRemNhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {

protected:
  using Base    = OrdLocBase<SpecAddRemNhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp>;
  using Factory = OrdLocFactoryBase<SpecAddRemNhoodItem, Ctxt, CtxtCmp>;

  using PQ = galois::ThreadSafeMinHeap<Ctxt*, CtxtCmp>;

  using NF = OptimNItemFunctions<SpecAddRemNhoodItem, Ctxt, CtxtCmp>;

  const CtxtCmp& ctxtCmp;
  PQ sharers;
  HistList histList;

  NhoodItem(Lockable* l, const CtxtCmp& ctxtcmp) : Base(l), sharers(ctxtcmp) {}

  void addSharer(const Ctxt* ctxt) {

    assert(!sharers.find(const_cast<Ctxt*>(ctxt)));
    Ctxt* prevTop = sharers.top();
    sharers.push(const_cast<Ctxt*>(ctxt));

    if (isMin(ctxt)) {
      detectAborts(ctxt);

      if (prevTop) {
        prevTop.disableSrc();
      }
    }
  }

  bool isMin(const Ctxt* ctxt) const {
    assert(ctxt);
    assert(!sharers.empty());
    return (sharers.top() == ctxt);
  }

  Ctxt* getMin() const {
    if (sharers.empty()) {
      return NULL;

    } else {
      return sharers.top();
    }
  }

  void removeSharerHead(Ctxt* ctxt) {
    assert(sharers.top() == ctxt);
    sharers.pop();
  }

  void addToHistory(Ctxt* ctxt) { NF::addToHistory(this, ctxt); }

  Ctxt* getHistHead(void) const { return NF::getHistHead(this); }

  Ctxt* getHistTail(void) const { return NF::getHistTail(this); }

  template <typename WL>
  void findAborts(Ctxt* ctxt, WL& abortRoots) {
    NF::findAborts(this, ctxt, abortRoots);
  }

  template <typename WL>
  void markForAbort(Ctxt* ctxt, WL& abortRoots) {
    NF::markForAbort(this, ctxt, abortRoots);
  }

  void removeAbort(Ctxt* ctxt) { NF::removeAbort(this, ctxt); }

  void removeCommit(Ctxt* ctxt) { NF::removeCommit(this, ctxt); }

  void detectAborts(Ctxt* ctxt) { NF::detectAborts(this, ctxt); }
};

template <typename T, typename Cmp, typename Exec>
struct SpecAddRemContext : public SpecContextBase<T, Cmp, Exec> {

  using Base = SpecContextBase<T, Cmp, Exec>;

  using CtxtCmp   = typename Base::CtxtCmp;
  using NItem     = SpecAddRemNhoodItem<OptimContext, CtxtCmp>;
  using NhoodMgr  = PtrBasedNhoodMgr<NItem>;
  using NhoodList = typename galois::Vector<NItem*>;
  using ChildList = typename galois::Vector<OptimContext*>;

  using CF = OptimContextFunctions<SpecAddRemContext, SpecAddRemNhoodItem>;

  galois::GAtomic<bool> onWL;
  bool addBack; // set to false by parent when parent is marked for abort, see
                // markAbortRecursive
  NhoodList nhood;
  ChildList children;

  explicit SpecAddRemContext(const T& x, const ContextState& s, Exec& exec)
      : Base(x, s, exec), onWL(false), addBack(true) {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire(Lockable* l, galois::MethodFlag m) {

    NItem& nitem = Base::exec.getNhoodMgr().getNhoodItem(l);
    assert(NItem::getOwner(l) == &nitem);

    if (std::find(nhood.begin(), nhood.end(), &nitem) == nhood.end()) {
      nhood.push_back(&nitem);
      nitem.addSharer(this);
    }
  }

  bool isSrcSlowCheck(void) const {

    if (CF::isSrcSlowCheck(this)) {
      this->enableSrc();
      return true;
    } else {
      return false;
    }
  }

  template <typename WL>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void findNewSources(WL& wl) {

    for (const NItem* ni : nhood) {

      SpecAddRemContext* highest = ni->getHighestPriority();
      if ((highest != NULL) && highest->isSrcSlowCheck() &&
          !bool(highest->onWL) && highest->onWL.cas(false, true)) {

        // GALOIS_DEBUG ("Adding found source: %s\n", highest->str ().c_str ());
        highest->enableSrc();
        wl.push(highest);
      }
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void removeExecutedSource(void) {
    assert(isSrc());

    for (NItem* ni : nhood) {
      assert(ni->isMin(this));
      ni->removeSharerHead(this);
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void addToHistory(void) {
    assert(isSrc());

    CF::addToHistory(this);

    removeExecutedSource();
  }

  void addChild(SpecAddRemContext* child) { CF::addChild(this, child); }

  void doCommit(void) { CF::doCommit(this); }

  void doAbort(void) { CF::doAbort(this); }

  bool isCommitSrc(void) const { return CF::isCommitSrc(this); }

  template <typename WL>
  void findCommitSrc(const SpecAddRemContext* gvt, WL& wl) const {
    CF::findCommitSrc(this, gvt, wl);
  }

  bool isAbortSrc(void) const { return CF::isAbortSrc(this); }

  template <typename WL>
  void findAbortSrc(WL& wl) const {
    CF::findAbortSrc(this, wl);
  }

  bool isSrcSlowCheck(void) const { return CF::isSrcSlowCheck(this); }

  void addAbortLocation(NItem* ni) { CF::addAbortLocation(this, ni); }

  template <typename WL>
  void markForAbortRecursive(WL& abortRoots) {
    CF::markForAbortRecursive(this, abortRoots);
  }
};

template <typename T, typename Cmp, typename NhFunc, typename ExFunc,
          typename OpFunc, typename ArgsTuple>
class KDGspecAddRemExecutor
    : public OrdSpecExecBase<
          T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple,
          SpecAddRemContext<T, Cmp,
                            KDGspecAddRemExecutor<T, Cmp, NhFunc, ExFunc,
                                                  OpFunc, ArgsTuple>>> {

  void expandNhoodPending(void) {
    galois::runtime::do_all_gen(
        makeLocalRange(pendingWL),
        [this](Ctxt* c) {
          typename Base::UserCtxt& uhand = c->userHandle;

          // nhFunc (c, uhand);
          runCatching(Base::nhFunc, c, uhand);

          Base::roundTasks += 1;
        },
        std::make_tuple(galois::loopname("include-pending"),
                        galois::chunk_size<NhFunc::CHUNK_SIZE>()));

    galois::runtime::do_all_gen(
        makeLocalRange(pendingWL),
        [this](Ctxt* c) {
          if (c->isSrc()) {
            Base::currWL->push(c);
          }
        },
        std::make_tuple(galois::loopname("collect-sources"),
                        galois::chunk_size<Base::DEFAULT_CHUNK_SIZE>()));
  }

  void applyOperator(void) {

    Ctxt* minWinWL = Base::getMinWinWL();

    galois::runtime::do_all_gen(
        makeLocalRange(Base::currWL),
        [this](Ctxt* c) {
          typename Base::UserCtxt& uhand = c->userHandle;

          assert(c->isSrc());
          assert(!c->hasState(ContextState::RECLAIM));
          assert(!c->hasState(ContextState::ABORTED_CHILD));

          Base::opFunc(c->getActive(), uhand);

          if (Base::NEEDS_PUSH) {

            for (auto i    = uhand.getPushBuffer().begin(),
                      endi = uhand.getPushBuffer().end();
                 i != endi; ++i) {

              // Ctxt* child = Base::push_commit (*i, minWinWL);
              Ctxt* child = Base::ctxtMaker(*i);
              c->addChild(child);
            }
            uhand.getPushBuffer().clear();
          } else {

            assert(uhand.getPushBuffer().begin() ==
                   uhand.getPushBuffer().end());
          }

          bool b = c->casState(ContextState::SCHEDULED,
                               ContextState::READY_TO_COMMIT);

          assert(b && "CAS shouldn't have failed");
          Base::roundCommits += 1;

          c->addToHistory();
          Base::commitQ.get().push_back(c);

          if (Base::ENABLE_PARAMETER) {
            c->markExecRound(Base::rounds);
          }
        },
        std::make_tuple(galois::loopname("applyOperator"),
                        galois::chunk_size<OpFunc::CHUNK_SIZE>()));

    galois::runtime::do_all_gen(
        makeLocalRange(Base::getCurrWL()),
        [this, minWinWL](Ctxt* c) {
          for (Ctxt* child : c->children) {

            if (!minWinWL || (Base::targetCommitRatio == 0.0) ||
                ctxtCmp(child, minWinWL)) {
              typename Base::UserCtxt& uhand = child->userHandle;
              child->schedule();

              // nhFunc (c, uhand);
              runCatching(Base::nhFunc, child, uhand);

              Base::roundTasks += 1;
            } else {
              Base::winWL.push(child);
            }
          }
        },
        std::make_tuple(galois::loopname("add-children"),
                        galois::chunk_size<Base::DEFAULT_CHUNK_SIZE>()));

    galois::runtime::do_all_gen(
        makeLocalRange(Base::getCurrWL()),
        [this](Ctxt* c) {
          for (Ctxt* child : c->children) {

            if (child->hasState(ContextState::SCHEDULED) &&
                child->isSrcSlowCheck() && child->onWL.cas(false, true)) {

              Base::nextWL->push(child);
            }
          }

          c->findNewSources(Base::getNextWL());
        },
        std::make_tuple(galois::loopname("add-children"),
                        galois::chunk_size<Base::DEFAULT_CHUNK_SIZE>()));
  }

public:
  // Algorithm 1:
  //  pick a window from pending
  //  expandNhoodPending
  //    collect sources from pending
  //  applyOperator
  //
  //  serviceAborts // pending may cause aborts on executed sources
  //
  //    for all sources:
  //      if isSrc and not aborted
  //      execute src
  //      remove from shares add to hist
  //    for all sources:
  //      expand nhood for new tasks
  //    for all sources:
  //      discover new  sources
  //      add all new sources to nextWL
  //
  //  serviceAborts // pending may cause aborts on executed sources
  //
  //  gvt is min of nextWL
  //  repeat

  // Algorithm 2:
  //  pick a window from pending
  //  expandNhoodPending
  //  serviceAborts // pending may cause aborts on executed sources
  //  collect sources
  //  applyOperator
  //    if a task is not source, ignore and look for new sources
  //    expand nhood for new tasks that are in the window
  //
  template <typename R>
  void execute(const R& range) {

    push_initial(range);

    while (true) {

      beginRound();

      expandNhoodPending();

      serviceAborts();

      if (Base::currWL->empty_all()) {
        assert(pending.empty_all());
        break;
      }

      applyOperator();

      performCommits();

      Base::endRound();
    }
  }
};

} // namespace runtime
} // namespace galois

#endif // GALOIS_RUNTIME_KDG_SPEC_LOCAL_MIN_H
