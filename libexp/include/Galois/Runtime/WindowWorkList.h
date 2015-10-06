/** Window WorkList -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_WINDOW_WORKLIST_H
#define GALOIS_RUNTIME_WINDOW_WORKLIST_H

#include "Galois/Accumulator.h"
#include "Galois/RangePQ.h"
#include "Galois/PerThreadContainer.h"

#include "Galois/Substrate/gio.h"


namespace Galois {
namespace Runtime { 

template <typename T, typename Cmp>
class SortedRangeWindowWL {
  using PerThrdWL = Galois::PerThreadVector<T>;
  using Iter = typename PerThrdWL::local_iterator;
  using Range = std::pair<Iter, Iter>;

  Cmp cmp;
  PerThrdWL m_wl;
  Substrate::PerThreadStorage<Range> wlRange;

  size_t init_sz = 0;

public:

  explicit SortedRangeWindowWL (const Cmp& cmp=Cmp ()): cmp (cmp) {
    Substrate::gPrint("Using SortedRangeWindowWL\n");
  }

  template <typename R>
  void initfill (const R& range) {

    GAccumulator<size_t> count;

    Galois::Runtime::do_all_impl (range,
        [this, &count] (const T& x) {
          m_wl.get ().push_back (x);
          count += 1;
        }
        , "initfill");

    init_sz = count.reduce ();

    Galois::Runtime::on_each_impl (
        [this] (const unsigned tid, const unsigned numT) {
          std::sort (m_wl[tid].begin (), m_wl[tid].end (), cmp);
        }
        , "initsort");

    for (unsigned i = 0; i < m_wl.numRows (); ++i) {
      *(wlRange.getRemote (i)) = std::make_pair (m_wl[i].begin (), m_wl[i].end ());
    }
  }

  size_t initSize () const { return init_sz; }

  bool empty () const { 
    bool e = true;

    for (unsigned i = 0; i < wlRange.size (); ++i) {
      const Range& r = *wlRange.getRemote (i);
      if (r.first != r.second) {
        e = false;
        break;
      }
    }

    return e;
  }


  const T* getMin (void) const {
    unsigned numT = getActiveThreads ();

    const T* minElem = nullptr;

    for (unsigned i = 0; i < numT; ++i) {
      const Range& r = *wlRange.getRemote (i);

      if (r.first != r.second) {
        if (minElem == nullptr || cmp (*minElem, *r.first)) {
          minElem = &(*r.first);
        }
      }
    }

    return minElem;
  }


  template <typename WL, typename CtxtMaker>
  void poll (WL& workList, const size_t newSize, const size_t origSize, CtxtMaker& ctxtMaker ) {

    if (origSize >= newSize) { 
      return;
    }

    const size_t numT = Galois::getActiveThreads ();

    const size_t numPerThrd = (newSize - origSize) / numT;

    const T* windowLim = nullptr;

    for (size_t i = 0; i < numT; ++i) {
      Range& r = *(wlRange.getRemote (i));
      const T* lim = nullptr;

      if (std::distance (r.first, r.second) <= ptrdiff_t (numPerThrd)) {

        if (r.first != r.second) {
          assert (std::distance (r.first, r.second) >= 1);

          Iter it = r.first;
          std::advance (it, std::distance (r.first, r.second) - 1);
          lim = &(*it);
        }

      } else {
        Iter it = r.first;
        std::advance (it, numPerThrd);

        lim = &(*it);
      }

      if (lim != nullptr) {
        if ((windowLim == nullptr) || cmp (*windowLim, *lim)) {
          windowLim = lim;
        }
      }

    }

    if (windowLim != nullptr) {
      Galois::Runtime::on_each_impl (
          [this, &workList, &ctxtMaker, numPerThrd, windowLim] (const unsigned tid, const unsigned numT) {
            Range& r = *(wlRange.getLocal (tid));

            for (size_t i = 0; (i < numPerThrd) 
              && (r.first != r.second); ++r.first) {

              workList.get ().push_back (ctxtMaker (*(r.first)) );
              ++i;

            }

            for (; r.first != r.second 
              && cmp (*(r.first), *windowLim); ++r.first) {

                workList.get ().push_back (ctxtMaker (*(r.first)));
            }
          }
          , "poll");

    } else {

      for (unsigned i = 0; i < numT; ++i) {
        Range& r = *(wlRange.getRemote (i));
        assert (r.first == r.second);
      }
    }


  }
    

  void push (const T& x) {
    GALOIS_DIE("not implemented for range based WindowWL");
  }


};


template <typename T, typename Cmp>
class PQbasedWindowWL {

  using PerThrdWL = Galois::PerThreadMinHeap<T, Cmp>;

  Cmp cmp;
  PerThrdWL m_wl;

public:

  explicit PQbasedWindowWL (const Cmp& cmp=Cmp ())
    : cmp (cmp), m_wl (cmp) 
  {
    Substrate::gPrint("Using PQbasedWindowWL\n");
  }


  template <typename R>
  void initfill (const R& range) {

    Galois::Runtime::do_all_impl (range,
        [this] (const T& x) {
          m_wl.get ().push (x);
        }
        , "initfill");

  }

  const T* getMin (void) const {
    const T* windowLim = nullptr;

    unsigned numT = getActiveThreads ();

    // compute a lowest priority limit
    for (unsigned i = 0; i < numT; ++i) {

      if (!m_wl[i].empty ()) {
        if (windowLim == nullptr || cmp (*windowLim, m_wl[i].top ())) {
          windowLim = &(m_wl[i].top ());
        }
      }
    }

    return windowLim;
  }

  void push (const T& x) {
    m_wl.get ().push (x);
  }

  size_t initSize (void) const {
    return m_wl.size_all ();
  }

  bool empty (void) const {
    return m_wl.empty_all ();
  }

  template <typename WL, typename CtxtMaker>
  void poll (WL& workList, const size_t newSize, const size_t origSize, CtxtMaker& ctxtMaker ) {

    if (origSize >= newSize) { 
      return;
    }

    const size_t numT = Galois::getActiveThreads ();

    const size_t numPerThrd = (newSize - origSize) / numT;

    // part 1 of poll
    // windowLim is calculated by computing the max of max element pushed by each
    // thread. In this case, the max element is the one pushed in last 

    Galois::Runtime::on_each_impl (
        [this, &workList, &ctxtMaker, numPerThrd] (const unsigned tid, const unsigned numT) {


          unsigned lim = std::min (m_wl.get ().size (), numPerThrd);

          for (unsigned i = 0; i < lim; ++i) {
            workList.get ().push_back (ctxtMaker (m_wl.get ().top ()));
            m_wl.get ().pop ();
          }
        }
        , "poll_part_1");

    const T* windowLim = nullptr;
    // compute the max of last element pushed into any workList rows
    for (unsigned i = 0; i < numT; ++i) {


      if (!workList[i].empty ()) {
        const T* last = &(workList[i].back ()->getElem ());
        assert (last != nullptr);

        if (windowLim == nullptr || cmp (*windowLim, *last)) {
          windowLim = last;
        }
      }
    }

    // for (unsigned i = 0; i < numT; ++i) {
      // const T* const lim = m_wl[i].empty () ? nullptr : &(m_wl[i].top ());
      // if (lim != nullptr) {
        // if ((windowLim == nullptr) || cmp (*windowLim, *lim)) { // *windowLim < *lim
          // windowLim = lim;
        // }
      // }
    // }

    // part 2 of poll
    // windowLim is the max of the min of each thread's
    // container,
    // every thread must select elements lesser than windowLim
    // for processing, 
    // in order to ensure that an element B from Thread i is not scheduled ahead 
    // of elment A from Thread j, such that A and B have a dependence
    // and A < B. 

    if (windowLim != NULL) {

      T limCopy (*windowLim);

      Galois::Runtime::on_each_impl (
          [this, &workList, &ctxtMaker, &limCopy] (const unsigned tid, const unsigned numT) {
            for (const T* t = &m_wl.get ().top (); 
              !m_wl.get ().empty () && cmp (*t, limCopy); t = &m_wl.get ().top ()) {

                workList.get ().push_back (ctxtMaker (*t));
                m_wl.get ().pop ();
            }
            
          }
          , "poll_part_2");




      for (unsigned i = 0; i < numT; ++i) {
        if (!m_wl[i].empty ()) {
          assert (!cmp (m_wl[i].top (), limCopy) && "poll gone wrong");
        }
      }
    }

  }

};

template <typename T, typename Cmp> 
class PartialPQbasedWindowWL {

  using PerThrdWL = Substrate::PerThreadStorage<TreeBasedPartialPQ<T, Cmp> >;

  Cmp cmp;
  PerThrdWL m_wl;


public:

  explicit PartialPQbasedWindowWL (const Cmp& cmp=Cmp ()): cmp (cmp) {}

  template <typename R>
  void initfill (const R& range) {
    Galois::Runtime::on_each_impl (
        [this, range] (const unsigned tid, const unsigned numT) {
          m_wl.getLocal ()->initfill (range.local_begin (), range.local_end ());
        }, "initfill");
  }

  template <typename WL>
  void poll (WL& workList, const size_t numElems) {

    Galois::Runtime::on_each_impl (
        [this, &workList, numElems] (const unsigned tid, const unsigned numT) {
          const size_t numPerThrd = numElems / numT;
          m_wl.getLocal ()->poll (workList, numPerThrd);
        }, "poll_part_1");


    const T* windowLim = nullptr;

    for (unsigned i = 0; i < m_wl.size (); ++i) {
      const T* lim = nullptr;
      if (!m_wl.getRemote (i)->empty ()) {
        lim = m_wl.getRemote (i)->getMin ();
      }

      if (lim != nullptr) {
        if (windowLim == nullptr || cmp (*windowLim, *lim)) {
          windowLim = lim;
        }
      }
    }

    if (windowLim != nullptr) {
      Galois::Runtime::on_each_impl (
          [this, &workList, windowLim] (const unsigned tid, const unsigned numT) {
            m_wl.getLocal ()->partition (workList, *windowLim);
          }, "poll_part_2");


      for (unsigned i = 0; i < m_wl.size (); ++i) {
        const T* lim = m_wl.getRemote (i)->getMin ();
        if (lim != nullptr) {
          assert (!cmp (*lim, *windowLim) && "prefix invariant violated");
        }
      }
    }
    
  }

  void push (const T& x) {
    assert (false && "not implemented yet");
  }
  
};


} // end namespace Runtime
} // end namespace Galois


#endif // GALOIS_RUNTIME_WINDOW_WORKLIST_H

