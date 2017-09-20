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

#include "Galois/gIO.h"

#include <boost/noncopyable.hpp>


namespace galois {
namespace runtime { 

template <typename T, typename Cmp>
class SortedRangeWindowWL: private boost::noncopyable {
  using PerThrdWL = galois::PerThreadVector<T>;
  using Iter = typename PerThrdWL::local_iterator;
  using Range = std::pair<Iter, Iter>;

  Cmp cmp;
  PerThrdWL m_wl;
  Substrate::PerThreadStorage<Range> wlRange;

  size_t init_sz = 0;

public:

  explicit SortedRangeWindowWL (const Cmp& cmp=Cmp ()): cmp (cmp) {
    gPrint("Using SortedRangeWindowWL\n");
  }

  template <typename R>
  void initfill (const R& range) {

    GAccumulator<size_t> count;

    galois::do_all_choice (range,
        [this, &count] (const T& x) {
          m_wl.get ().push_back (x);
          count += 1;
        }, 
        std::make_tuple (
        galois::loopname ("initfill"),
        galois::chunk_size<16> ()));

    init_sz = count.reduce ();

    galois::runtime::on_each_impl (
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


  galois::optional<T> getMin (void) const {
    unsigned numT = getActiveThreads ();

    galois::optional<T> minElem;

    for (unsigned i = 0; i < numT; ++i) {
      const Range& r = *wlRange.getRemote (i);

      if (r.first != r.second) {
        if (!minElem || cmp (*minElem, *r.first)) {
          minElem = *r.first;
        }
      }
    }

    return minElem;
  }

  template <typename WL>
  void poll (WL& workList, const size_t newSize, const size_t origSize) {
    poll (workList, newSize, origSize, [] (const T& x) { return x; });
  }

  template <typename WL, typename Wrap>
  void poll (WL& workList, const size_t newSize, const size_t origSize, Wrap& wrap) {

    if (origSize >= newSize) { 
      return;
    }

    const size_t numT = galois::getActiveThreads ();

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
      galois::runtime::on_each_impl (
          [this, &workList, &wrap, numPerThrd, windowLim] (const unsigned tid, const unsigned numT) {
            Range& r = *(wlRange.getLocal ());

            for (size_t i = 0; (i < numPerThrd) 
              && (r.first != r.second); ++r.first) {

              workList.get ().push_back (wrap (*(r.first)) );
              ++i;

            }

            for (; r.first != r.second 
              && cmp (*(r.first), *windowLim); ++r.first) {

                workList.get ().push_back (wrap (*(r.first)));
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

// TODO: add functionality to spill complete bag or range efficiently 
//
template <typename T, typename Cmp, typename PerThrdWL, typename Derived>
class WindowWLbase: private boost::noncopyable {

protected:

  using dbg = galois::debug<0>;

  Cmp cmp;
  PerThrdWL m_wl;


public:
  explicit WindowWLbase (const Cmp& cmp=Cmp ())
    : cmp (cmp), m_wl (cmp)
  {
  }


  template <typename R>
  void initfill (const R& range) {

    galois::do_all_choice (range,
        [this] (const T& x) {
          push (x);
        }, 
        std::make_tuple (
          galois::loopname ("initfill"),
          galois::chunk_size<16> ()));

  }

  galois::optional<T> getMin (void) const {
    galois::optional<T> ret;

    unsigned numT = getActiveThreads ();

    // compute a lowest priority limit
    for (unsigned i = 0; i < numT; ++i) {

      if (!m_wl[i].empty ()) {
        const T& top = static_cast<const Derived*> (this)->getTop (i);
        if (!ret || cmp (top, *ret)) {
          ret = top;
        }
      }
    }

    return ret;
  }

  void push (const T& x) {
    static_cast<Derived*> (this)->pushImpl (x);
  }

  size_t initSize (void) const {
    return m_wl.size_all ();
  }

  bool empty (void) const {
    return m_wl.empty_all ();
  }

  template <typename WL>
  void poll (WL& workList, const size_t newSize, const size_t origSize) {

    auto f = [] (const T& x) { return x; };
    poll (workList, newSize, origSize, f);
  }

  template <typename WL, typename Wrap>
  void poll (WL& workList, const size_t newSize, const size_t origSize, Wrap& wrap) {

    if (origSize >= newSize) { 
      return;
    }

    const size_t numT = galois::getActiveThreads ();

    const size_t numPerThrd = (newSize - origSize) / numT;

    // part 1 of poll
    // windowLim is calculated by computing the max of max element pushed by each
    // thread. In this case, the max element is the one pushed in last 

    Substrate::PerThreadStorage<galois::optional<T> > perThrdLastPop;

    Derived* d = static_cast<Derived*> (this);
    assert(d);

    galois::runtime::on_each_impl (
        [this, d, &workList, &wrap, numPerThrd, &perThrdLastPop] (const unsigned tid, const unsigned numT) {

          galois::optional<T>& lastPop = *(perThrdLastPop.getLocal ());


          int lim = std::min (m_wl.get ().size (), numPerThrd);

          for (int i = 0; i < lim; ++i) {
            if (i == (lim - 1)) {
              lastPop = d->getTop ();
            }

            dbg::print("Removing and adding to window: ", d->getTop ());

            workList.get ().push_back (wrap (d->getTop ()));
            d->popMin();
          }
        }
        , "poll_part_1");



    // // compute the max of last element pushed into any workList rows
    //
    galois::optional<T> windowLim;

    for (unsigned i = 0; i < numT; ++i) {
      const galois::optional<T>& lp = *(perThrdLastPop.getRemote (i));

      if (lp) {
        if (!windowLim || cmp (*windowLim, *lp)) {
          windowLim = *lp;
        }
      }
    }
 
    // for (unsigned i = 0; i < numT; ++i) {
      // if (!workList[i].empty ()) {
        // const T& last = unwrap (workList[i].back ());
        // assert (&last != nullptr);
// 
        // if (windowLim == nullptr || cmp (*windowLim, last)) {
          // windowLim = &last;
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

    if (windowLim) {

      Derived* d = static_cast<Derived*> (this);
      assert(d);

      galois::runtime::on_each_impl (
          [this, d, &workList, &wrap, &windowLim] (const unsigned tid, const unsigned numT) {

            while (!m_wl.get ().empty ()) {
              const T& t = d->getTop ();

              if (cmp (*windowLim, t)) { // windowLim < t
                break;
              }

              dbg::print("Removing and adding to window: ", t, ", windowLim: ", *windowLim);
              workList.get ().push_back (wrap (t));
              d->popMin ();
            }
          }
          , "poll_part_2");


      galois::optional<T> min  = d->getMin ();
      if (min) {
        assert (cmp (*windowLim, *min));
      }


      for (unsigned i = 0; i < numT; ++i) {
        if (!m_wl[i].empty ()) {
          assert (cmp (*windowLim, d->getTop (i)) && "poll gone wrong");
        }
      }

      
    }

  }


};

template <typename T, typename Cmp>
class PQwindowWL: public WindowWLbase<T, Cmp, PerThreadMinHeap<T, Cmp>, PQwindowWL<T, Cmp> > {


  using PerThrdWL = galois::PerThreadMinHeap<T, Cmp>;
  using Base = WindowWLbase<T, Cmp, PerThrdWL, PQwindowWL>;

  template <typename, typename, typename, typename> 
  friend class WindowWLbase;

  const T& getTop (void) const {
    assert (!Base::m_wl.get ().empty ());
    return Base::m_wl.get ().top ();
  }

  const T& getTop (const unsigned i) const {
    assert (!Base::m_wl[i].empty ());
    return Base::m_wl[i].top ();
  }

  void pushImpl (const T& x) {
    Base::m_wl.get ().push (x);
  }

  void popMin (void) {
    Base::m_wl.get ().pop ();
  }

public:

  explicit PQwindowWL (const Cmp& cmp=Cmp ())
    : Base (cmp)
  {
    gPrint("Using PQwindowWL\n");
  }


};


template <typename T, typename Cmp>
class SetWindowWL: public WindowWLbase<T, Cmp, PerThreadSet<T, Cmp>, SetWindowWL<T, Cmp> > {


  // using PerThrdWL = galois::PerThreadMinHeap<T, Cmp>;
  using PerThrdWL = galois::PerThreadSet<T, Cmp>;
  using Base = WindowWLbase<T, Cmp, PerThrdWL, SetWindowWL>;

  template <typename, typename, typename, typename> 
  friend class WindowWLbase;

  const T& getTop (void) const {
    assert (!Base::m_wl.get ().empty ());
    return *(Base::m_wl.get ().begin ());
  }

  const T& getTop (const unsigned i) const {
    assert (!Base::m_wl[i].empty ());
    return *(Base::m_wl[i].begin ());
  }

  void pushImpl (const T& x) {
    Base::m_wl.get ().insert (x);
  }

  void popMin (void) {
    Base::m_wl.get ().erase (Base::m_wl.get ().begin ());
  }

public:

  explicit SetWindowWL (const Cmp& cmp=Cmp ())
    : Base (cmp)
  {
    gPrint("Using SetwindowWL\n");
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
    galois::runtime::on_each_impl (
        [this, range] (const unsigned tid, const unsigned numT) {
          m_wl.getLocal ()->initfill (range.local_begin (), range.local_end ());
        }, "initfill");
  }

  template <typename WL>
  void poll (WL& workList, const size_t numElems) {

    galois::runtime::on_each_impl (
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
      galois::runtime::on_each_impl (
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


} // end namespace runtime
} // end namespace galois


#endif // GALOIS_RUNTIME_WINDOW_WORKLIST_H

