/** Range based Partial PQ -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RANGE_PQ_H
#define GALOIS_RANGE_PQ_H

#include <utility>
#include <type_traits>
#include <map>

#include "galois/BoundedVector.h"

namespace galois {

template <typename T, typename Cmp, unsigned N=1024, typename WL=galois::BoundedVector<T, N> >
class RangeBuffer: public WL {

  using Super = WL;

  Cmp m_cmp;
  const T* m_min;
  const T* m_max;

public:

  RangeBuffer (const Cmp& cmp=Cmp ()): WL (), m_cmp (cmp), m_min (nullptr), m_max (nullptr)  {}

  void push_back (const T& x) {
    Super::push_back (x);
    if (m_min == nullptr || m_cmp (x, *m_min)) {
      m_min = &Super::back ();
    }

    if (m_max == nullptr || m_cmp (*m_max, x)) {
      m_max = &Super::back ();
    }

    assert (!m_cmp (*m_max, *m_min));
  }

  void updateLimits () {
    for (auto i = Super::begin (), endi = Super::end (); i != endi; ++i) {
      if (m_min == nullptr || m_cmp (*i, *m_min)) {
        m_min = &(*i);
      }

      if (m_max == nullptr || m_cmp (*m_max, *i)) {
        m_max = &(*i);
      }
    }
  }

  const T* getMin () const { return m_min; }

  const T* getMax () const { return m_max; }

  // TODO: hide pop_back

public:

  struct Comparator {

    Cmp cmp;

    explicit Comparator (const Cmp& _cmp): cmp (_cmp) {}

    template <typename WL1, typename WL2, unsigned N1, unsigned N2>
    int compare (const RangeBuffer<T, Cmp, N1, WL1>& left, const RangeBuffer<T, Cmp, N2, WL2>& right) const {
      assert (!left.empty ());
      assert (!right.empty ());
      assert (left.m_min != nullptr && left.m_max != nullptr);
      assert (right.m_min != nullptr && right.m_max != nullptr);

      if (cmp (*left.m_max, *right.m_min)) {
        return -1;
      }
      else if (cmp (*right.m_max, *left.m_min)) {
        return 1;
      }
      else {
        // a-----b
        //    c------d
        // or
        // c------d
        //    a-------b
        // or
        // a------c----d------b
        // or
        // c------a----b------d

        // (a < c && b < d) || (c < a && d < b) || (a < c && d < b) || (c < a && b
        // < d)
        // or
        // (c < b && a < d)

        assert (cmp (*right.m_min, *left.m_max) && cmp (*left.m_min, *right.m_max));

        return 0;

      }
    }

    template <typename WL1, typename WL2, unsigned N1, unsigned N2>
    bool operator () (const RangeBuffer<T, Cmp, N1, WL1>& left, const RangeBuffer<T, Cmp, N2, WL2>& right) const {
      return compare (left, right) < 0;
    }
  };

  struct PtrComparator {

    Comparator c;

    explicit PtrComparator (const Cmp& cmp): c (cmp) {}

    template <typename WL1, typename WL2, unsigned N1, unsigned N2>
    bool operator () (const RangeBuffer<T, Cmp, N1, WL1>* const lp, const RangeBuffer<T, Cmp, N2, WL2>* const rp) const {
      assert (lp != nullptr);
      assert (rp != nullptr);
      return c (*lp, *rp);
    }
  };

};


namespace internal {

  template <typename T>
  struct Identity {
    const T& operator () (const T& x) {
      return x;
    }
  };

  template <typename T, typename Cmp>
  struct TypeHelper {
    using RBuf = RangeBuffer<T, Cmp>;
    using UnitBuf = RangeBuffer<T, Cmp, 1>;
    using RBufAlloc = galois::runtime::FixedSizeAllocator<RBuf>;
    using RBufPtrAlloc = galois::runtime::FixedSizeAllocator<RBuf*>;
    using Tree = std::map<RBuf*, RBuf*, typename RBuf::PtrComparator, RBufPtrAlloc>;
    using Set = std::set<RBuf*, typename RBuf::PtrComparator, RBufPtrAlloc>;
  };
}


template <typename T, typename Cmp>
class RangePQTreeBased: public internal::TypeHelper<T,Cmp>::Tree {

  using THelper = internal::TypeHelper<T, Cmp>;
  using RBuf = typename THelper::RBuf;
  using RBufAlloc = typename THelper::RBufAlloc;
  using RBufPtrAlloc = typename THelper::RBufPtrAlloc;
  using Tree = typename THelper::Tree;
  using UnitBuf = typename THelper::UnitBuf;

  using RBufCmp = typename RBuf::Comparator;
  using RBufPtrCmp = typename RBuf::PtrComparator;

  RBufPtrCmp ptrcmp;

public:


  explicit RangePQTreeBased (const Cmp& cmp=Cmp ())
    : Tree (RBufPtrCmp (cmp), RBufPtrAlloc ()), ptrcmp (cmp)
  {}

  typename Tree::iterator mergePoint (const T& item) {
    assert (!Tree::empty ());

    UnitBuf unitRange;
    unitRange.push_back (item);

    auto node = Tree::_M_begin ();

    while (true) {

      T* nv = Tree::_S_value (node);
      auto left = Tree::_S_left (node);
      auto right = Tree::_S_right (node);

      if (ptrcmp (&unitRange, nv)) { // item < nv
        if (left != nullptr) {
          node = left;

        } else {
          break;
        }

      } else if (ptrcmp (nv, &unitRange)) { // nv < item

        if (right != nullptr) {
          node = right;

        } else {
          break;

        }

      } else { // nv == item
        break;

      }

    }

    return Tree::iterator (node);
  }

  std::pair<typename Tree::iterator, bool> insert (RBuf* x) {
    return Tree::_M_insert_unique (x);
  }


};

template <typename T, typename Cmp>
class RangePQSetBased: public internal::TypeHelper<T, Cmp>::Set {

  using THelper = internal::TypeHelper<T, Cmp>;
  using RBuf = typename THelper::RBuf;
  using RBufAlloc = typename THelper::RBufAlloc;
  using RBufPtrAlloc = typename THelper::RBufPtrAlloc;
  using Set = typename THelper::Set;
  using UnitBuf = typename THelper::UnitBuf;

  using RBufCmp = typename RBuf::Comparator;
  using RBufPtrCmp = typename RBuf::PtrComparator;

  RBufPtrCmp ptrcmp;

public:

  explicit RangePQSetBased (const Cmp& cmp=Cmp())
    : Set (RBufPtrCmp (cmp), RBufPtrAlloc ()), ptrcmp (cmp)
  {}

  typename Set::iterator mergePoint (const T& item) {
    assert (!Set::empty ());

    UnitBuf unitRange;
    unitRange.push_back (item);

    auto i = Set::find (&unitRange);

    if (i != Set::end ()) {
      return i;

    } else {

      auto j = Set::begin ();
      auto endj = Set::end ();

      for (; j != endj; ++j) {
        if (ptrcmp (&unitRange, *j)) {
          break;
        }
      }

      if (j == Set::end ()) {
        --j;
      }

      return j;
    }
  }

};



template <typename T, typename Cmp, typename PQImpl>
class PartialPQBase {

  using THelper = internal::TypeHelper<T, Cmp>;
  using RBuf = typename THelper::RBuf;
  using RBufAlloc = typename THelper::RBufAlloc;

  Cmp cmp;
  RBufAlloc bufAlloc;
  PQImpl pq;

public:

  PartialPQBase (const Cmp& cmp=Cmp ()): cmp (cmp), bufAlloc (), pq (cmp)
  {}

  bool empty () const {
    return pq.empty ();
  }

  const T* getMin () const {
    if (pq.empty ()) {
      return nullptr;

    }  else {
      RBuf* head = *pq.begin ();
      return head->getMin ();

    }
  }

  template <typename I>
  void initfill (I b, I e) {
    partition_recursive (b, e);
  }

  template <typename WL>
  void poll (WL& workList, const size_t numPerThrd) {
    size_t numChunks = (numPerThrd + RBuf::capacity () - 1) / RBuf::capacity ();

    for (size_t i = 0; i < numChunks && !pq.empty (); ++i) {
      RBuf* head = *pq.begin ();
      pq.erase (pq.begin ());

      copyOut (workList, head);
    }
  }

  template <typename WL>
  void push_back (const T& item) {
    auto mp = pq.mergePoint (item);
    merge (mp, item);
  }

  template <typename WL>
  void partition (WL& workList, const T& windowLim) {

    assert (&windowLim != nullptr);

    while (!pq.empty ()) {

      RBuf* head = *pq.begin ();

      if (cmp (*head->getMin (), windowLim)) {
        // head has min less than windowLim
        //
        // first remove
        pq.erase (pq.begin ());

        if (cmp (*head->getMax (), windowLim)) {
          copyOut (workList, head);

        } else {
          auto splitPt = partition_reverse (head->begin (), head->end (), windowLim);
          assert (splitPt != head->end ());

          for (auto i = splitPt, endi = head->end (); i != endi; ++i) {
            workList.get ().push_back (*i);
          }

          for (ptrdiff_t i = 0, lim = (head->end () - splitPt); i < lim; ++i) {
            head->pop_back ();
          }

          head->updateLimits ();

          // add back if head was partitioned
          if (!head->empty ()) {
            auto r = pq.insert (head);
            assert (r.second);

            assert (!cmp (*head->getMin(), windowLim));
            assert (cmp (windowLim, *head->getMax()));
          }

        } // end else

      } else {
        // head has a min greater than (or eq.) windowLim
        break;
      }

    }

  }

private:

  template <typename I>
  I partition_reverse (const I beg, const I end, const T& pivot) {

    I b = beg;
    I e = end;

    assert (b != e);

    --e;
    while (b != e) {
      while (b != e && cmp (pivot, *b)) {
        ++b;
      }

      while (b != e && cmp (*e, pivot)) {
        --e;
      }

      if (b != e) {
        std::swap (*b, *e);
      }
    }

    return b;
  }

  template <typename WL>
  void copyOut (WL& workList, RBuf* head) {
    for (auto i = head->begin (), endi = head->end (); i != endi; ++i) {
      workList.get ().push_back (*i);
    }

    bufAlloc.destroy (head);
    bufAlloc.deallocate (head, 1);

  }

  template <typename I>
  void partition_recursive (const I beg, const I end) {

    if (std::distance (beg, end) < ptrdiff_t (RBuf::capacity ())) {
      if (std::distance (beg, end) > 0) {
        RBuf* buf = makeBuf (beg, end);
        pq.insert (buf);

      }
    } else {
      I b = beg;
      I e = end;
      using V = typename std::remove_reference<decltype (*b)>::type;
      V pivot = *(b + std::distance (b, e) / 2);

      --e;
      while (b != e) {

        while (b != e && cmp (*b, pivot)) {
          ++b;
        }

        while (b != e && cmp (pivot, *e)) {
          --e;
        }

        std::swap (*b, *e);
      }

      partition_recursive (beg, b);
      partition_recursive (e, end);

    }

  }

  template <typename I>
  RBuf* makeBuf (const I beg, const I end) {

    RBuf* buf = bufAlloc.allocate (1);
    bufAlloc.construct (buf, RBuf ());

    for (I i = beg; i != end; ++i) {
      buf->push_back (*i);
    }

    pq.insert (buf);

    return buf;
  }

  template <typename PI>
  void merge (PI mp, const T& item) {

    RBuf& nv = *mp;

    if (nv.full ()) {
      pq.erase (mp);

      auto middle = nv.begin () + nv.size () / 2;
      std::nth_element (nv.begin (), middle, nv.end (), nv.comparator ());

      RBuf* lower = makeBuf (nv.begin (), middle);
      RBuf* upper = makeBuf (middle, nv.end ());

      if (cmp (item, *upper)) { // item < upper
        lower->push_back (item);

      } else {
        upper->push_back (item);

      }

    } else {
      nv.push_back (item);
    }
  }


};

template <typename T, typename Cmp>
struct TreeBasedPartialPQ: public PartialPQBase<T, Cmp, RangePQTreeBased<T, Cmp> > {

  TreeBasedPartialPQ (const Cmp& cmp=Cmp ())
    : PartialPQBase<T, Cmp, RangePQTreeBased<T, Cmp> > (cmp)
  {}
};

template <typename T, typename Cmp>
struct SetBasedPartialPQ: public PartialPQBase<T, Cmp, RangePQSetBased<T, Cmp> > {

  SetBasedPartialPQ (const Cmp& cmp=Cmp ())
    : PartialPQBase<T, Cmp, RangePQSetBased<T, Cmp> > (cmp)
  {}

};



} // end namespace galois

#endif // GALOIS_RANGE_PQ_H
