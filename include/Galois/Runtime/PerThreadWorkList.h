/** Per Thread workLists-*- C++ -*-
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
 * a thread local stl container for each thread
 *
 * @author <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_
#define GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_

#include <vector>
#include <deque>
#include <list>
#include <set>
#include <limits>
#include <iterator>
#include <iostream>

#include <cstdio>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "Galois/Threads.h"
#include "Galois/PriorityQueue.h"
#include "Galois/TwoLevelIterator.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/ll/gio.h"

namespace Galois {
namespace Runtime {

namespace {

enum GlobalPos {
  GLOBAL_BEGIN, GLOBAL_END
};

// #define ADAPTOR_BASED_OUTER_ITER

// XXX: use a combination of boost::transform_iterator and
// boost::counting_iterator to implement the following OuterPerThreadWLIter
#ifdef ADAPTOR_BASED_OUTER_ITER

template <typename PerThrdWL>
struct WLindexer: 
  public std::unary_function<unsigned, typename PerThrdWL::Cont_ty&> 
{

  using Ret_ty = typename PerThrdWL::Cont_ty;

  PerThrdWL* wl;

  WLindexer (): wl (NULL) {}

  WLindexer (PerThrdWL& _wl): wl(&_wl) {}

  Ret_ty& operator () (unsigned i) const {
    assert (wl != NULL);
    assert (i < wl->numRows ());
    return const_cast<Ret_ty&> (wl->get (i));
  }
};

template <typename PerThrdWL>
struct TypeFactory {
  using OuterIter = boost::transform_iterator<WLindexer<PerThrdWL>, boost::counting_iterator<unsigned> >;

  using RvrsOuterIter = std::reverse_iterator<OuterIter>;
};


template <typename PerThrdWL>
typename TypeFactory<PerThrdWL>::OuterIter make_outer_begin (PerThrdWL& wl) {
  return boost::make_transform_iterator (
      boost::counting_iterator<unsigned> (0), WLindexer<PerThrdWL> (wl));
}

template <typename PerThrdWL>
typename TypeFactory<PerThrdWL>::OuterIter make_outer_end (PerThrdWL& wl) {
  return boost::make_transform_iterator (
      boost::counting_iterator<unsigned> (wl.numRows ()), WLindexer<PerThrdWL> (wl));
}

template <typename PerThrdWL>
typename TypeFactory<PerThrdWL>::RvrsOuterIter make_outer_rbegin (PerThrdWL& wl) {
  return typename TypeFactory<PerThrdWL>::RvrsOuterIter (make_outer_end (wl));
}

template <typename PerThrdWL>
typename TypeFactory<PerThrdWL>::RvrsOuterIter make_outer_rend (PerThrdWL& wl) {
  return typename TypeFactory<PerThrdWL>::RvrsOuterIter (make_outer_begin (wl));
}

#else

template <typename PerThrdWL>
class OuterPerThreadWLIter: public std::iterator<std::random_access_iterator_tag, typename PerThrdWL::Cont_ty> {

  using Cont_ty = typename PerThrdWL::Cont_ty;
  using Super_ty = std::iterator<std::random_access_iterator_tag, Cont_ty>;
  using Diff_ty = typename Super_ty::difference_type;

  PerThrdWL* workList;
  // using Diff_ty due to reverse iterator, whose 
  // end is -1, and,  begin is numRows - 1
  Diff_ty row;

  void assertInRange () const {
    assert ((row >= 0) && (row < workList->numRows ()));
  }

  Cont_ty& getWL () {
    assertInRange ();
    return (*workList)[row];
  }

  const Cont_ty& getWL () const {
    assertInRange ();
    return (*workList)[row];
  }


public:

  OuterPerThreadWLIter (): Super_ty (), workList (NULL), row (0) {}

  OuterPerThreadWLIter (PerThrdWL& wl, const GlobalPos& pos)
    : Super_ty (), workList (&wl), row (0) {

    switch (pos) {
      case GLOBAL_BEGIN:
        row = 0;
        break;
      case GLOBAL_END:
        row = wl.numRows ();
        break;
      default:
        std::abort ();
    }
  }

  typename Super_ty::reference operator * () { return getWL (); }

  const typename Super_ty::reference operator * () const { return getWL (); }

  typename Super_ty::pointer operator -> () { return &(getWL ()); }

  const typename Super_ty::value_type* operator -> () const { return &(getWL ()); }

  OuterPerThreadWLIter& operator ++ () {
    ++row;
    return *this;
  }

  OuterPerThreadWLIter operator ++ (int) {
    OuterPerThreadWLIter tmp (*this);
    operator ++ ();
    return tmp;
  }

  OuterPerThreadWLIter& operator -- () {
    --row;
    return *this;
  }

  OuterPerThreadWLIter operator -- (int) {
    OuterPerThreadWLIter tmp (*this);
    operator -- ();
    return tmp;
  }

  OuterPerThreadWLIter& operator += (Diff_ty d) {
    row = unsigned (Diff_ty (row) + d);
    return *this;
  }

  OuterPerThreadWLIter& operator -= (Diff_ty d) {
    row = unsigned (Diff_ty (row) - d);
    return *this;
  }

  friend OuterPerThreadWLIter operator + (const OuterPerThreadWLIter& it, Diff_ty d) {
    OuterPerThreadWLIter tmp (it);
    tmp += d;
    return tmp;
  }

  friend OuterPerThreadWLIter operator + (Diff_ty d, const OuterPerThreadWLIter& it) {
    return it + d;
  }

  friend OuterPerThreadWLIter operator - (const OuterPerThreadWLIter& it, Diff_ty d) {
    OuterPerThreadWLIter tmp (it);
    tmp -= d;
    return tmp;
  }

  friend Diff_ty operator - (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {
    return Diff_ty (left.row) - Diff_ty (right.row);
  }

  typename Super_ty::reference operator [] (Diff_ty d) {
    return *((*this) + d);
  }

  friend bool operator == (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    assert (left.workList == right.workList);
    return (left.row == right.row);
  }

  friend bool operator != (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {
    return !(left == right);
  }

  friend bool operator < (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    assert (left.workList == right.workList);

    return (left.row < right.row);
  }

  friend bool operator <= (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    return (left == right) || (left < right);
  }

  friend bool operator > (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    return !(left <= right);
  }

  friend bool operator >= (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    return !(left < right);
  }

};


template <typename PerThrdWL>
OuterPerThreadWLIter<PerThrdWL> make_outer_begin (PerThrdWL& wl) {
  return OuterPerThreadWLIter<PerThrdWL> (wl, GLOBAL_BEGIN);
}

template <typename PerThrdWL>
OuterPerThreadWLIter<PerThrdWL> make_outer_end (PerThrdWL& wl) {
  return OuterPerThreadWLIter<PerThrdWL> (wl, GLOBAL_END);
}

template <typename PerThrdWL>
std::reverse_iterator<OuterPerThreadWLIter<PerThrdWL> > 
  make_outer_rbegin (PerThrdWL& wl) {
  using Ret_ty = std::reverse_iterator<OuterPerThreadWLIter<PerThrdWL> >;
  return Ret_ty (make_outer_end (wl));
}

template <typename PerThrdWL>
std::reverse_iterator<OuterPerThreadWLIter<PerThrdWL> > 
  make_outer_rend (PerThrdWL& wl) {
  using Ret_ty = std::reverse_iterator<OuterPerThreadWLIter<PerThrdWL> >;
  return Ret_ty (make_outer_begin (wl));
}

#endif

} // end namespace 


template <typename Cont_tp> 
class PerThreadWorkList {

public:
  using Cont_ty = Cont_tp;
  using value_type = typename Cont_ty::value_type;
  using reference = typename Cont_ty::reference;
  using pointer = typename Cont_ty::pointer;
  using size_type = typename Cont_ty::size_type;

  using local_iterator = typename Cont_ty::iterator;
  using local_const_iterator = typename Cont_ty::const_iterator;
  using local_reverse_iterator = typename Cont_ty::reverse_iterator;
  using local_const_reverse_iterator = typename Cont_ty::const_reverse_iterator;

  using This_ty = PerThreadWorkList;

#ifdef ADAPTOR_BASED_OUTER_ITER

  using OuterIter = typename TypeFactory<This_ty>::OuterIter;
  using RvrsOuterIter = typename TypeFactory<This_ty>::RvrsOuterIter;

#else

  using OuterIter = OuterPerThreadWLIter<This_ty>;
  using RvrsOuterIter = std::reverse_iterator<OuterIter>;

#endif

  using global_iterator = typename Galois::ChooseStlTwoLevelIterator<OuterIter, typename Cont_ty::iterator>::type;

  using global_const_iterator = typename Galois::ChooseStlTwoLevelIterator<OuterIter, typename Cont_ty::const_iterator>::type;

  using global_reverse_iterator = typename Galois::ChooseStlTwoLevelIterator<RvrsOuterIter, typename Cont_ty::reverse_iterator>::type;

  using global_const_reverse_iterator = typename Galois::ChooseStlTwoLevelIterator<RvrsOuterIter, typename Cont_ty::const_reverse_iterator>::type;



private:

  // XXX: for testing only

#if 0
  struct FakePTS {
    std::vector<Cont_ty*> v;

    FakePTS () { 
      v.resize (size ());
    }

    Cont_ty** getLocal () const {
      return getRemote (Galois::Runtime::LL::getTID ());
    }

    Cont_ty** getRemote (size_t i) const {
      assert (i < v.size ());
      return const_cast<Cont_ty**> (&v[i]);
    }

    size_t size () const { return Galois::Runtime::LL::getMaxThreads(); }

  };
#endif



  // typedef FakePTS PerThrdCont_ty;
  using PerThrdCont_ty = Galois::Runtime::PerThreadStorage<Cont_ty*>;
  PerThrdCont_ty perThrdCont;

  void destroy () {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      delete *perThrdCont.getRemote (i);
      *perThrdCont.getRemote (i) = NULL;
    }
  }

protected:
  PerThreadWorkList (): perThrdCont () {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      *perThrdCont.getRemote (i) = NULL;
    }
  }

  void init (const Cont_ty& cont) {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      *perThrdCont.getRemote (i) = new Cont_ty (cont);
    }
  }


  ~PerThreadWorkList () { 
    destroy ();
  }


public:
  unsigned numRows () const { return perThrdCont.size (); }

  Cont_ty& get () { return **(perThrdCont.getLocal ()); }

  const Cont_ty& get () const { return **(perThrdCont.getLocal ()); }

  Cont_ty& get (unsigned i) { return **(perThrdCont.getRemote (i)); }

  const Cont_ty& get (unsigned i) const { return **(perThrdCont.getRemote (i)); }

  Cont_ty& operator [] (unsigned i) { return get (i); }

  const Cont_ty& operator [] (unsigned i) const { return get (i); }


  global_iterator begin_all () { 
    return Galois::stl_two_level_begin (
        make_outer_begin (*this), make_outer_end (*this)); 
  }

  global_iterator end_all () { 
    return Galois::stl_two_level_end (
        make_outer_begin (*this), make_outer_end (*this)); 
  }

  global_const_iterator begin_all () const { 
    return Galois::stl_two_level_cbegin (
        make_outer_begin (*this), make_outer_end (*this));
  }

  global_const_iterator end_all () const { 
    return Galois::stl_two_level_cend (
        make_outer_begin (*this), make_outer_end (*this));
  }

  global_const_iterator cbegin_all () const { 
    return Galois::stl_two_level_cbegin (
        make_outer_begin (*this), make_outer_end (*this));
  }

  global_const_iterator cend_all () const { 
    return Galois::stl_two_level_cend (
        make_outer_begin (*this), make_outer_end (*this));
  }

  global_reverse_iterator rbegin_all () { 
    return Galois::stl_two_level_rbegin (
        make_outer_rbegin (*this), make_outer_rend (*this)); 
  }

  global_reverse_iterator rend_all () { 
    return Galois::stl_two_level_rend (
        make_outer_rbegin (*this), make_outer_rend (*this)); 
  }

  global_const_reverse_iterator rbegin_all () const { 
    return Galois::stl_two_level_crbegin (
        make_outer_rbegin (*this), make_outer_rend (*this));
  }

  global_const_reverse_iterator rend_all () const { 
    return Galois::stl_two_level_crend (
        make_outer_rbegin (*this), make_outer_rend (*this));
  }

  global_const_reverse_iterator crbegin_all () const { 
    return Galois::stl_two_level_crbegin (
        make_outer_rbegin (*this), make_outer_rend (*this));
  }

  global_const_reverse_iterator crend_all () const { 
    return Galois::stl_two_level_crend (
        make_outer_rbegin (*this), make_outer_rend (*this));
  }


  size_type size_all () const {
    size_type sz = 0;

    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      sz += get (i).size ();
    }

    return sz;
  }


  void clear_all () {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      get (i).clear ();
    }
  }

  bool empty_all () const {
    bool res = true;
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      res = res && get (i).empty ();
    }

    return res;
  }


  // TODO: fill parallel

  template <typename Iter, typename R>
  void fill_serial (Iter begin, Iter end,
      R (Cont_ty::*pushFn) (const value_type&)=&Cont_ty::push_back) {

    const unsigned P = Galois::getActiveThreads ();

    using Diff_ty = typename std::iterator_traits<Iter>::difference_type;

    // integer division, where we want to round up. So adding P-1
    Diff_ty block_size = (std::distance (begin, end) + (P-1) ) / P;

    assert (block_size >= 1);

    Iter block_begin = begin;

    for (unsigned i = 0; i < P; ++i) {

      Iter block_end = block_begin;

      if (std::distance (block_end, end) < block_size) {
        block_end = end;

      } else {
        std::advance (block_end, block_size);
      }

      for (; block_begin != block_end; ++block_begin) {
        // workList[i].push_back (Marked<Value_ty> (*block_begin));
        ((*this)[i].*pushFn) (value_type (*block_begin));
      }

      if (block_end == end) {
        break;
      }
    }
  }

};


namespace PerThreadFactory {

  using BasicHeap =  MM::SimpleBumpPtrWithMallocFallback<MM::FreeListHeap<MM::SystemBaseAlloc> >;
  
  using Heap = MM::ThreadAwarePrivateHeap<BasicHeap>;

  template <typename T>
  using Alloc = MM::ExternRefGaloisAllocator<T, Heap>;

  template <typename T>
  using FSBAlloc = MM::FSBGaloisAllocator<T>;

  template <typename T>
  using Vector =  std::vector<T, Alloc<T> >;

  template <typename T>
  using Deque = std::deque<T, Alloc<T> >;

  template <typename T>
  using List = std::list<T, FSBAlloc<T> >;

  template <typename T, typename C>
  using Set = std::set<T, C, FSBAlloc<T> >;

  template <typename T, typename C>
  using PQ = MinHeap<T, C, Vector<T> >;

};


template <typename T>
class PerThreadVector: 
  public PerThreadWorkList<PerThreadFactory::Vector<T> > {

public:
  using Heap_ty = PerThreadFactory::Heap;
  using Alloc_ty = PerThreadFactory::Alloc<T>;

protected:
  using Cont_ty = PerThreadFactory::Vector<T>;
  using Super_ty = PerThreadWorkList<Cont_ty>;

  Heap_ty heap;
  Alloc_ty alloc;

public:
  PerThreadVector (): Super_ty (), heap (), alloc (&heap) {

    Super_ty::init (Cont_ty (alloc));
  }

  void reserve_all (size_t sz) {
    size_t numT = Galois::getActiveThreads ();
    size_t perT = (sz + numT - 1) / numT; // round up

    for (unsigned i = 0; i < numT; ++i) {
      Super_ty::get (i).reserve (perT);
    }
  }
  

};


template <typename T>
class PerThreadDeque: 
  public PerThreadWorkList<PerThreadFactory::Deque<T> > {

public:
  using Heap_ty = PerThreadFactory::Heap;
  using Alloc_ty = PerThreadFactory::Alloc<T>;

protected:
  using Cont_ty = PerThreadFactory::Deque<T>;
  using Super_ty = PerThreadWorkList<Cont_ty>;

  Heap_ty heap;
  Alloc_ty alloc;

public:
  PerThreadDeque (): Super_ty (), heap (), alloc (&heap) {

    Super_ty::init (Cont_ty (alloc));

  }

};

template <typename T>
class PerThreadList:
  public PerThreadWorkList<PerThreadFactory::List<T> > {

public:
  using Heap_ty = PerThreadFactory::Heap;
  using Alloc_ty = PerThreadFactory::Alloc<T>;

protected:
  using Cont_ty = PerThreadFactory::List<T>;
  using Super_ty = PerThreadWorkList<Cont_ty>;

  Heap_ty heap;
  Alloc_ty alloc;

public:
  PerThreadList (): Super_ty (), heap (), alloc (&heap) {

    Super_ty::init (Cont_ty (alloc));

  }

};

template <typename T, typename C=std::less<T> >
class PerThreadSet: 
  public PerThreadWorkList<PerThreadFactory::Set<T, C> > {

public:
  using Alloc_ty = PerThreadFactory::FSBAlloc<T>;

protected:
  using Cont_ty = PerThreadFactory::Set<T, C>;
  using Super_ty = PerThreadWorkList<Cont_ty>;

  Alloc_ty alloc;

public:
  explicit PerThreadSet (const C& cmp = C ()): Super_ty (), alloc () {

    Super_ty::init (Cont_ty (cmp, alloc));
  }

  using typename Super_ty::global_const_iterator;
  using typename Super_ty::global_const_reverse_iterator;

  // hiding non-const (and const) versions in Super_ty
  global_const_iterator begin_all () const { return Super_ty::cbegin_all (); }
  global_const_iterator end_all () const { return Super_ty::cend_all (); }

  // hiding non-const (and const) versions in Super_ty
  global_const_reverse_iterator rbegin_all () const { return Super_ty::crbegin_all (); }
  global_const_reverse_iterator rend_all () const { return Super_ty::crend_all (); }

};


template <typename T, typename C=std::less<T> >
class PerThreadMinHeap:
  public PerThreadWorkList<PerThreadFactory::PQ<T, C> > {

public:
  using Heap_ty = PerThreadFactory::Heap;
  using Alloc_ty = PerThreadFactory::Alloc<T>;

protected:
  using Vec_ty = PerThreadFactory::Vector<T>;
  using Cont_ty = PerThreadFactory::PQ<T, C>;
  using Super_ty = PerThreadWorkList<Cont_ty>;

  Heap_ty heap;
  Alloc_ty alloc;

public:
  explicit PerThreadMinHeap (const C& cmp = C ()): Super_ty (), heap (), alloc (&heap) {
    Super_ty::init (Cont_ty (cmp, Vec_ty (alloc)));
  }

  using typename Super_ty::global_const_iterator;
  using typename Super_ty::global_const_reverse_iterator;

  // hiding non-const (and const) versions in Super_ty
  global_const_iterator begin_all () const { return Super_ty::cbegin_all (); }
  global_const_iterator end_all () const { return Super_ty::cend_all (); }

  // hiding non-const (and const) versions in Super_ty
  global_const_reverse_iterator rbegin_all () const { return Super_ty::crbegin_all (); }
  global_const_reverse_iterator rend_all () const { return Super_ty::crend_all (); }


};


}
} // end namespace Galois

#endif // GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_
